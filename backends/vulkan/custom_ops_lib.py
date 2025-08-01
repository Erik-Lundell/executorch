# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.library

namespace = "et_vk"
lib = torch.library.Library(namespace, "DEF")

#############
## prepack ##
#############


def prepack_impl(x: torch.Tensor):
    return x


name = "prepack"
lib.define(f"{name}(Tensor x) -> Tensor")
lib.impl(name, prepack_impl, "CompositeExplicitAutograd")
prepack_op = getattr(getattr(torch.ops, namespace), name)

#####################
## conv_with_clamp ##
#####################


def conv_with_clamp_impl(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    transposed=False,
    output_padding=0,
    groups=1,
    output_min=-float("inf"),
    output_max=float("inf"),
):
    return torch.clamp(
        torch.convolution(
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        ),
        output_min,
        output_max,
    )


name = "conv_with_clamp"
lib.define(
    f"{name}(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups, Scalar? output_min, Scalar? output_max) -> Tensor"
)
lib.impl(name, conv_with_clamp_impl, "CompositeExplicitAutograd")
conv_with_clamp_op = getattr(getattr(torch.ops, namespace), name)

#########################
## conv_with_clamp.out ##
#########################


def conv_with_clamp_out_impl(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    transposed=False,
    output_padding=0,
    groups=1,
    output_min=-float("inf"),
    output_max=float("inf"),
    out=None,
):
    out = conv_with_clamp_impl(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        output_min,
        output_max,
    )
    return out


name = "conv_with_clamp.out"
lib.define(
    f"{name}(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups, Scalar? output_min, Scalar? output_max, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.impl(name, conv_with_clamp_out_impl, "CompositeExplicitAutograd")

#################
## grid_priors ##
#################


# The dimension of x should be larger than 1
def grid_priors_impl(
    x,
    stride,
    offset,
):
    height, width = x.shape[-2:]
    # Need to specify device of torch.arange to avoid executorch exporting error
    shift_x = (torch.arange(0, width, device=x.device) + offset) * stride
    shift_y = (torch.arange(0, height, device=x.device) + offset) * stride
    # Need to specify indexing parameter ('ij' is the default value) to avoid executorch exporting error
    shift_xx, shift_yy = torch.meshgrid([shift_y, shift_x], indexing="ij")
    shift_xx = shift_xx.reshape(-1)
    shift_yy = shift_yy.reshape(-1)
    shifts = torch.stack((shift_yy, shift_xx), dim=-1)
    return shifts


name = "grid_priors"
lib.define(f"{name}(Tensor self, int stride, float offset) -> Tensor")
lib.impl(name, grid_priors_impl, "CompositeExplicitAutograd")
grid_priors_op = getattr(getattr(torch.ops, namespace), name)


# When lowering to executorch, ops are converted from default to out variant. Hence, custom ops define both variants.
def grid_priors_out_impl(
    x,
    stride,
    offset,
    out,
):
    out = grid_priors_impl(x, stride, offset)
    return out


name = "grid_priors_out"
lib.define(
    f"{name}(Tensor self, int stride, float offset, *, Tensor(a!) out) -> Tensor(a!)"
)
lib.impl(name, grid_priors_out_impl, "CompositeExplicitAutograd")

########################
## linear_weight_int4 ##
########################


def linear_weight_int4_impl(
    x: torch.Tensor,
    weights_4x8: torch.Tensor,
    groupsize: int,
    scales_and_zeros: torch.Tensor,
    inner_k_tiles: int,
):
    original_x_size = x.size()
    out_features = weights_4x8.size(0)
    x = x.reshape(-1, original_x_size[-1])
    weight_int4pack = torch.ops.aten._convert_weight_to_int4pack(
        weights_4x8, inner_k_tiles
    )
    out = torch.ops.aten._weight_int4pack_mm(
        x, weight_int4pack, groupsize, scales_and_zeros
    )
    out_shape = original_x_size[:-1] + (out_features,)
    return out.reshape(out_shape)


name = "linear_weight_int4"
lib.define(
    f"{name}(Tensor self, Tensor mat2, int qGroupSize, Tensor qScaleAndZeros, int inner_k_tiles) -> Tensor"
)
lib.impl(name, linear_weight_int4_impl, "CompositeExplicitAutograd")
linear_weight_int4_op = getattr(getattr(torch.ops, namespace), name)

#################
## linear_qcs4w ##
#################


def linear_qcs4w(
    x: torch.Tensor,
    weights_4x2: torch.Tensor,
    scales: torch.Tensor,
):
    original_x_shape = x.shape
    x = x.reshape(-1, original_x_shape[-1])

    unpacked_weights_shape = weights_4x2.shape
    out_features = unpacked_weights_shape[0]
    in_features = unpacked_weights_shape[1]

    weights_unpacked = torch.empty(
        (out_features, in_features * 2), dtype=torch.int8, device=weights_4x2.device
    )

    weights_unpacked[:, ::2] = weights_4x2 >> 4
    weights_unpacked[:, 1::2] = weights_4x2 & 0x0F

    n_bit = 8
    quant_min = -(2 ** (n_bit - 1))
    quant_max = 2 ** (n_bit - 1) - 1
    dq_weights = torch.ops.quantized_decomposed.dequantize_per_channel(
        weights_unpacked,
        scales,
        None,
        0,
        quant_min,
        quant_max,
        torch.int8,
    )

    out = torch.nn.functional.linear(x, dq_weights)
    out_shape = original_x_shape[:-1] + (out_features,)
    return out.reshape(out_shape)


name = "linear_qcs4w"
lib.define(f"{name}(Tensor self, Tensor weight, Tensor scales) -> Tensor")
lib.impl(name, linear_qcs4w, "CompositeExplicitAutograd")
linear_qc4w_op = getattr(getattr(torch.ops, namespace), name)

########################
## linear_qta8a_qga4w ##
########################


def linear_qta8a_qga4w(
    x_quantized: torch.Tensor,
    input_scale: torch.Tensor,
    input_zero_point: torch.Tensor,
    weights_4bit: torch.Tensor,
    group_size: int,
    weight_scales: torch.Tensor,
    weight_zeros: torch.Tensor,
):
    """
    Dynamic activation + grouped weight quantized linear (QTA8A_QGA4W).

    Args:
        x_quantized: Already quantized input tensor (int8, per-token quantized)
        input_scale: Scale for per-token quantization of input (shape: [batch_size])
        input_zero_point: Zero point for per-token quantization of input (shape: [batch_size])
        weights_4bit: Packed 4-bit quantized weights
        group_size: Group size for weight quantization (int)
        weight_scales: Per-group scales for weights
        weight_zeros: Per-group zero points for weights
    """
    original_x_shape = x_quantized.shape
    feature_dim = original_x_shape[-1]

    # Reshape for processing
    x_quantized_2d = x_quantized.reshape(-1, feature_dim)

    # Unpack 4-bit weights
    unpacked_weights_shape = weights_4bit.shape
    out_features = unpacked_weights_shape[0]
    in_features = unpacked_weights_shape[1]

    weights_unpacked = torch.empty(
        (out_features, in_features * 2), dtype=torch.int8, device=weights_4bit.device
    )

    weights_unpacked[:, ::2] = weights_4bit >> 4
    weights_unpacked[:, 1::2] = weights_4bit & 0x0F

    # Convert to signed 4-bit range [-8, 7]
    weights_unpacked = torch.where(
        weights_unpacked > 7, weights_unpacked - 16, weights_unpacked
    )

    # Dequantize weights using grouped quantization
    actual_in_features = in_features * 2
    num_groups = actual_in_features // group_size

    # Reshape weights for grouped dequantization
    weights_grouped = weights_unpacked.view(out_features, num_groups, group_size)

    # Expand scales and zeros to match grouped weights
    scales_expanded = weight_scales.unsqueeze(-1).expand(-1, -1, group_size)
    zeros_expanded = weight_zeros.unsqueeze(-1).expand(-1, -1, group_size)

    # Dequantize: (quantized - zero_point) * scale
    dq_weights_grouped = (weights_grouped.float() - zeros_expanded) * scales_expanded
    dq_weights = dq_weights_grouped.view(out_features, actual_in_features)

    # Dequantize input (per-token)
    # For per-token quantization, each token (row) has its own scale and zero_point
    x_dequantized = torch.ops.quantized_decomposed.dequantize_per_token(
        x_quantized_2d,
        input_scale,
        input_zero_point,
        -128,
        127,
        torch.int8,
        torch.float32,
    )

    # Perform linear operation
    out = torch.nn.functional.linear(x_dequantized, dq_weights)
    out_shape = original_x_shape[:-1] + (out_features,)
    return out.reshape(out_shape)


name = "linear_qta8a_qga4w"
lib.define(
    f"{name}(Tensor self, Tensor input_scale, Tensor input_zero_point, Tensor weight, int group_size, Tensor weight_scales, Tensor weight_zeros) -> Tensor"
)
lib.impl(name, linear_qta8a_qga4w, "CompositeExplicitAutograd")
linear_qta8a_qga4w_op = getattr(getattr(torch.ops, namespace), name)

######################
## apply_rotary_emb ##
######################


# Note that this implementation is copied from executorch.examples.models.llama.rope
# but it is copied here to avoid introducing a dependency on the llama code.
def apply_rotary_emb_impl(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
):
    def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
        ndim = x.ndim
        freqs_cis_ndim = freqs_cis.ndim
        if freqs_cis_ndim == 3:
            # freqs_cis: (seq_len, n_heads, head_dim // 2)
            assert freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1])
            shape = [
                d if (i == ndim - 3 or i == ndim - 2 or i == ndim - 1) else 1
                for i, d in enumerate(x.shape)
            ]
        else:
            # freqs_cis: (seq_len, head_dim // 2)
            assert freqs_cis.shape == (x.shape[1], x.shape[-1])
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(shape)

    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


name = "apply_rotary_emb"
lib.define(
    f"{name}(Tensor xq, Tensor xk, Tensor freqs_cos, Tensor freqs_sin) -> (Tensor, Tensor)"
)
lib.impl(name, apply_rotary_emb_impl, "CompositeExplicitAutograd")
apply_rotary_emb_op = getattr(getattr(torch.ops, namespace), name)
