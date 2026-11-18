# Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pure torch equivalents of the acext int4 pack ops (pack_int8_tensor_to_packed_int4 and
preprocess_weights_for_mixed_gemm); bit-exact cross-check: tests/utils/test_pack_acext_int4.py."""
import torch

def get_bits_in_quant_type(quant_type: torch.dtype) -> int:
    """
        number of bits one quantized weight element occupies.
    """
    if quant_type == torch.int8:
        return 8
    if quant_type == torch.quint4x2:
        return 4
    raise ValueError(f"invalid quantization type {quant_type}, must be torch.int8 or torch.quint4x2")

def pack_int8_tensor_to_packed_int4_torch(weight: torch.Tensor, validate: bool = True) -> torch.Tensor:
    """
        pack two int8 elements of the last (contiguous) dimension into one byte, low nibble first.
        weight: (..., num_cols) int8 tensor holding values in the int4 range
        returns: (..., num_cols // 2) int8 tensor
    """
    if weight.dtype != torch.int8:
        raise ValueError("Weight must be a int8 tensor")
    if weight.numel() == 0:
        raise ValueError("weight should not be empty tensor")
    if weight.shape[-1] % 2 != 0:
        raise ValueError(f"the last dimension must be even but it is {weight.shape[-1]}")
    if validate and bool(((weight < -8) | (weight > 7)).any()):
        raise ValueError("Value in unpacked tensor not in int4 range")

    unpacked = weight.contiguous().view(torch.uint8)
    low_nibbles = unpacked[..., 0::2] & 0x0F
    high_nibbles = unpacked[..., 1::2] & 0x0F
    return (low_nibbles | (high_nibbles << 4)).view(torch.int8)

def permute_B_rows_for_mixed_gemm_torch(quantized_tensor: torch.Tensor, quant_type: torch.dtype,
                                        is_int8_mma: bool = False) -> torch.Tensor:
    """
        permute the rows of B inside every group of B_ROWS_PER_MMA rows so that ldsm feeds the mma in the right order.
        For int8, each group of 16 rows is permuted using the map below:
         0 1 8 9 2 3 10 11 4 5 12 13 6 7 14 15
        For int4, each group of 32 rows is permuted using the map below:
        0 1 8 9 16 17 24 25 2 3 10 11 18 19 26 27 4 5 12 13 20 21 28 29 6 7 14 15 22 23 30 31
        For int4 with int8 mma, each group of 32 rows is permuted using the map below:
         0 1 2 3 16 17 18 19 4 5 6 7 20 21 22 23 8 9 10 11 24 25 26 27 12 13 14 15 28 29 30 31
        quantized_tensor: (..., num_rows, num_bytes_per_row) int8 tensor, row major
    """
    bits_per_elt = get_bits_in_quant_type(quant_type)
    elts_per_reg = 32 // bits_per_elt
    b_rows_per_mma = 8 * (16 // bits_per_elt)
    num_rows = quantized_tensor.shape[-2]
    num_cols = quantized_tensor.shape[-1] * (8 // bits_per_elt)

    if num_rows % b_rows_per_mma != 0:
        raise ValueError(
            f"Invalid shape for quantized tensor. Number of rows of quantized matrix must be a multiple of {b_rows_per_mma}")
    if num_cols % 8 != 0:
        raise ValueError(
            "Invalid shape for quantized tensor. On turing/Ampere, the number of cols must be a multiple of 8.")
    if is_int8_mma and bits_per_elt != 4:
        raise ValueError("The int8 mma row permutation is only defined for packed int4 weights")

    tile_row = torch.arange(b_rows_per_mma, device=quantized_tensor.device)
    if is_int8_mma:
        tile_read_row = (tile_row % 8) // 4 * 16 + tile_row // 8 * 4 + tile_row % 4
    else:
        tile_read_row = 8 * ((tile_row % elts_per_reg) // 2) + tile_row % 2 + 2 * (tile_row // elts_per_reg)

    ### The map moves whole rows around, so it can be applied to the packed bytes directly.
    shape = quantized_tensor.shape
    tiled = quantized_tensor.contiguous().reshape(*shape[:-2], num_rows // b_rows_per_mma, b_rows_per_mma, shape[-1])
    return tiled.index_select(-2, tile_read_row).reshape(shape)

def subbyte_transpose_torch(quantized_tensor: torch.Tensor, quant_type: torch.dtype) -> torch.Tensor:
    """
        logically transpose a row major [num_rows, num_cols] quantized matrix, handling packed int4 at nibble level.
        quantized_tensor: (..., num_rows, num_bytes_per_row) int8 tensor
        returns: (..., num_cols, num_rows * bits / 8) int8 tensor
    """
    bits_per_elt = get_bits_in_quant_type(quant_type)
    num_rows = quantized_tensor.shape[-2]
    num_cols = quantized_tensor.shape[-1] * (8 // bits_per_elt)

    ### The C++ implementation transposes 64x64 element tiles and copies VECTOR_WIDTH bytes at a time, so it rejects
    ### anything whose byte counts are not a multiple of it. Mirror the constraint, otherwise a shape the C++ op refuses
    ### would silently produce a layout the gemm cannot consume instead of failing loudly.

    vector_width = 32
    col_bytes = num_cols * bits_per_elt // 8
    col_bytes_trans = num_rows * bits_per_elt // 8
    if col_bytes % vector_width or col_bytes_trans % vector_width:
        raise ValueError(
            f"Number of bytes for rows and cols must be a multiple of {vector_width}. However, num_rows_bytes = "
            f"{col_bytes_trans} and num_col_bytes = {col_bytes}.")

    if bits_per_elt == 8:
        return quantized_tensor.transpose(-1, -2).contiguous()
    unpacked = quantized_tensor.contiguous().view(torch.uint8)
    even_rows = unpacked[..., 0::2, :]  ### input rows 2 * j, they end up in the low nibbles of the output
    odd_rows = unpacked[..., 1::2, :]   ### input rows 2 * j + 1, they end up in the high nibbles

    ### Output byte [c, j] packs element (row=2 * j, col=c) and (row=2 * j + 1, col=c). Splitting the output rows by
    ### parity keeps everything packed, an intermediate one byte per element tensor would double the memory footprint.
    ### Careful, the even / odd prefix changes meaning here: even_rows and odd_rows above are indexed by input row
    ### parity, while even_out_rows and odd_out_rows below are indexed by output row parity and each of them reads both
    ### inputs. Output row c holds input column c, which sits in the low nibble of its byte for even c and in the high
    ### nibble for odd c.
    even_out_rows = ((even_rows & 0x0F) | ((odd_rows & 0x0F) << 4)).transpose(-1, -2)
    odd_out_rows = ((even_rows >> 4) | ((odd_rows >> 4) << 4)).transpose(-1, -2)
    transposed = torch.stack((even_out_rows, odd_out_rows), dim=-2)
    return transposed.reshape(*unpacked.shape[:-2], num_cols, num_rows // 2).view(torch.int8)

def add_bias_and_interleave_quantized_tensor_torch(quantized_tensor: torch.Tensor,
                                                   quant_type: torch.dtype) -> torch.Tensor:
    """
        shift the quantized values to unsigned and relayout every 32 bit register so that the gemm main loop needs as
        few instructions as possible to extract the elements.
        For int4 a register [elt_7 elt_6 elt_5 elt_4 elt_3 elt_2 elt_1 elt_0] becomes
        [elt_7 elt_5 elt_3 elt_1 elt_6 elt_4 elt_2 elt_0], for int8 [elt_3 elt_2 elt_1 elt_0] becomes
        [elt_3 elt_1 elt_2 elt_0].
    """
    bits_per_elt = get_bits_in_quant_type(quant_type)
    shape = quantized_tensor.shape
    if quantized_tensor.numel() % 4 != 0:
        raise ValueError(
            "Dimensions of the quantized tensor must be a multiple of 4 bytes for the register relayout")
    registers = quantized_tensor.contiguous().view(torch.uint8).reshape(-1, 4)
    if bits_per_elt == 4:
        ### Adding 8 to a 4 bit two's complement value is a xor with 0x8.
        registers = registers ^ 0x88
        byte_0, byte_1, byte_2, byte_3 = registers[:, 0], registers[:, 1], registers[:, 2], registers[:, 3]
        interleaved = torch.stack((
            (byte_0 & 0x0F) | ((byte_1 & 0x0F) << 4),
            (byte_2 & 0x0F) | ((byte_3 & 0x0F) << 4),
            (byte_0 >> 4) | ((byte_1 >> 4) << 4),
            (byte_2 >> 4) | ((byte_3 >> 4) << 4),
        ), dim=-1)
    else:
        ### Adding 128 to an int8 is a xor with 0x80.
        interleaved = (registers ^ 0x80)[:, [0, 2, 1, 3]]
    return interleaved.reshape(shape).view(torch.int8)

def preprocess_weights_for_mixed_gemm_torch(row_major_quantized_weight: torch.Tensor, quant_type: torch.dtype,
                                            is_int8_mma: bool = False) -> torch.Tensor:
    """
        torch equivalent of torch.ops.acext.preprocess_weights_for_mixed_gemm, usable on cuda tensors.
        row_major_quantized_weight: (num_rows, num_bytes_per_row) or (num_experts, num_rows, num_bytes_per_row) int8
        tensor, row major, already packed when quant_type is torch.quint4x2
        quant_type: torch.int8 or torch.quint4x2
        is_int8_mma: the int4 weights are dequantized right before the int8 mma (a8w4), which needs the int8 mma
            row permutation and no unsigned bias

        returns: an int8 tensor with the same shape as the input, holding the relayouted bytes. The shape is only a
            byte container, exactly like the tensor the C++ op returns: the subbyte transpose swapped rows and columns,
            so the bytes actually form a (num_cols, num_rows * bits / 8) column major matrix. Only slicing along the
            expert dimension is meaningful, and the last two dimensions must not be used to derive gemm sizes.
    """
    if row_major_quantized_weight.dtype != torch.int8:
        raise ValueError("Quantized tensor must be int8 dtype")
    if row_major_quantized_weight.dim() not in (2, 3):
        raise ValueError("Invalid dim. The dim of weight should be 2 or 3")

    processed = row_major_quantized_weight.contiguous()
    ### The permutation works on row major data, so issue it first.
    processed = permute_B_rows_for_mixed_gemm_torch(processed, quant_type, is_int8_mma)
    processed = subbyte_transpose_torch(processed, quant_type)
    if not is_int8_mma:
        processed = add_bias_and_interleave_quantized_tensor_torch(processed, quant_type)
    ### The transpose swapped rows and columns, report the shape the C++ op returns. See the docstring, the shape of
    ### the result carries no meaning beyond the total number of bytes.
    return processed.reshape(row_major_quantized_weight.shape)

def preprocess_weight_for_a8w4_torch(weight: torch.Tensor, validate: bool = True) -> torch.Tensor:
    """
        pack and relayout an int4 weight for the a8w4 gemm, which unpacks and dequantizes the weight in registers
        right before the int8 mma.

        weight: (num_experts, n, k) or (n, k) int8 tensor holding values in the int4 range, k-major
        returns: (num_experts, k, n / 2) or (k, n / 2) int8 tensor on the device of `weight`. As in
            preprocess_weights_for_mixed_gemm_torch the shape is only a byte container, see its docstring.
    """
    ### transpose to k, n and n-major
    packed_weight = pack_int8_tensor_to_packed_int4_torch(weight.transpose(-1, -2).contiguous(), validate=validate)
    return preprocess_weights_for_mixed_gemm_torch(packed_weight, torch.quint4x2, is_int8_mma=True)