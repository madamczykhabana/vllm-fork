###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import habana_frameworks.torch as htorch
import habana_frameworks.torch.utils.experimental as htexp
from typing import List, Optional, Tuple

import vllm.hpu.utils as hpu_utils

# FIXME: For some reason splitting value causes DFAs on G3. This needs to be debugged
PA_SPLIT_VALUE_DEFAULT = '0' if (htexp._get_device_type() == htexp.synDeviceType.synDeviceGaudi3) else '1'
PA_SPLIT_VALUE = (os.environ.get('PA_SPLIT_VALUE', PA_SPLIT_VALUE_DEFAULT) == '1')


def silu_and_mul(output, input):
    d = input.shape[-1] // 2
    silu = torch.nn.SiLU().to(input.device)
    x, y = torch.split(input, d, dim=-1)
    output.copy_(silu(x) * y)


def gelu_new(output, input):
    raise NotImplementedError


def gelu_fast(output, input):
    raise NotImplementedError


def batch2block(tensor, block_mapping):
    shape = tuple(tensor.shape)
    return (block_mapping @ tensor.view(shape[0], -1)).view(-1, *shape[1:])


def block2batch(tensor, block_mapping):
    shape = tuple(tensor.shape)
    return (block_mapping.t() @ tensor.view(shape[0], -1)).view(-1, *shape[1:])


def block_softmax(batch_size, attn, block_mapping):
    attn = attn.exp_()
    sums = attn.sum(dim=-1).unsqueeze(-1)
    sums = block2batch(sums, block_mapping)
    sums = batch2block(sums, block_mapping)
    attn.div_(sums)
    return attn


#@hpu_utils.with_mark_steps
def flat_pa(query, key_cache, value_cache, block_list, block_mapping, block_bias, scale, flipped=True):
    batch_size = query.size(0)
    q_heads = query.size(1)
    if flipped:
        kv_heads = key_cache.size(2)
    else:
        kv_heads = key_cache.size(1)

    query = batch2block(scale * query, block_mapping).unsqueeze(-2)
    key = torch.index_select(key_cache, 0, block_list)
    value = torch.index_select(value_cache, 0, block_list)
    block_bias = block_bias.view(key.size(0), 1, 1, -1)

    if kv_heads != q_heads:
        query = query.unflatten(1, (kv_heads, -1))
        block_bias = block_bias.unsqueeze(1)
        if flipped:
            key = key.unflatten(2, (kv_heads, 1))
            value = value.unflatten(2, (kv_heads, 1))
            key = key.permute(0, 2, 3, 4, 1)
            value = value.permute(0, 2, 3, 1, 4)
        else:
            key = key.unflatten(1, (kv_heads, 1))
            value = value.unflatten(1, (kv_heads, 1))
    else:
        if flipped:
            key = key.permute(0, 2, 3, 1)
            value = value.permute(0, 2, 1, 3)

    attn = (query @ key) + block_bias
    attn = block_softmax(batch_size, attn, block_mapping)
    attn = attn @ value
    attn = block2batch(attn, block_mapping)
    attn = attn.squeeze(-2)
    if kv_heads != q_heads:
        attn = attn.flatten(1, 2)
    return attn


def rms_norm(out, hidden_states, weight, eps):
    htorch.core.mark_step()
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    out.copy_(weight * hidden_states.to(input_dtype))
    htorch.core.mark_step()


def rotate_neox(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rotate_fn = rotate_neox if is_neox_style else rotate_gptj
    q_embed = (q * cos) + (rotate_fn(q) * sin)
    k_embed = (k * cos) + (rotate_fn(k) * sin)
    return q_embed, k_embed


def awq_gemm(*args):
    raise NotImplementedError


def silu_and_mul_wrapper(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    output_shape = (x.shape[:-1] + (d, ))
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    silu_and_mul(out, x)
    return out


@hpu_utils.with_mark_steps
def static_fused_moe(hidden_states, w1, w2, score, topk):
    B, D = hidden_states.shape
    num_experts = w1.shape[0]
    routing_weights = F.softmax(score, dim=1, dtype=torch.float32)
    routing_weights, selected_experts = torch.topk(routing_weights, topk, dim=-1)
    routing_weights = routing_weights.to(hidden_states.dtype)
    final_hidden_states = torch.zeros(
            (1, B, D), dtype=hidden_states.dtype, device=hidden_states.device
    )
    padded_weights = torch.zeros(
            (B, num_experts), dtype=hidden_states.dtype, device=hidden_states.device
    )
    padded_weights.scatter_(-1, selected_experts, routing_weights)
    padded_weights = padded_weights.reshape(-1, B, w1.shape[0])
    padded_weights = padded_weights.permute(2, 0, 1).unsqueeze(-1)

    for expert_idx in range(num_experts):
        padded_weight = padded_weights[expert_idx]
        current_state_static = hidden_states.reshape(-1, D)
        w_output = silu_and_mul_wrapper(torch.matmul(current_state_static, w1[expert_idx].transpose(0, 1)))
        w_output = torch.matmul(w_output, w2[expert_idx].transpose(0, 1))
        current_hidden_states_static = w_output * padded_weight
        final_hidden_states += current_hidden_states_static

    return final_hidden_states.view(-1, D)
