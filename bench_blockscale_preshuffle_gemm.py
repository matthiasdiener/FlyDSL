#!/usr/bin/env python3
# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Blockscale Preshuffle GEMM Benchmark for MXFP8 (FP8 A8W8 with per-block scales)."""

import time
import csv
import torch

from flydsl.runtime.device import get_rocm_arch

from kernels.blockscale_preshuffle_gemm import compile_blockscale_preshuffle_gemm
from tests.utils import shuffle_weight
import flydsl.compiler as flyc

ARCH = get_rocm_arch()
DTYPE_FP8 = torch.float8_e4m3fnuz if "gfx942" in ARCH else torch.float8_e4m3fn
SCALE_DTYPE = torch.float8_e8m0fnu  # MXFP8 block scaling format

BLOCK_SHAPE = (128, 128)  # (block_n, block_k)
SCALE_GRANULARITY_K = 32  # MXFP8 block size

# benchmark shapes: (name, M, N, K)
SHAPES = [
    # ("Test-Default", 256, 1536, 7168),  # from test_blockscale_preshuffle_gemm.py
    ("Llama-2-7B MBS=1", 4096, 12288, 4096),
    ("Llama-2-7B MBS=1", 4096, 4096, 4096),
    ("Llama-2-7B MBS=1", 4096, 22016, 4096),
    ("Llama-2-7B MBS=1", 4096, 4096, 11008),
    ("Llama-2-7B MBS=2", 8192, 12288, 4096),
    ("Llama-2-7B MBS=2", 8192, 4096, 4096),
    ("Llama-2-7B MBS=2", 8192, 22016, 4096),
    ("Llama-2-7B MBS=2", 8192, 4096, 11008),
    ("Llama-2-7B MBS=4", 16384, 12288, 4096),
    ("Llama-2-7B MBS=4", 16384, 4096, 4096),
    ("Llama-2-7B MBS=4", 16384, 22016, 4096),
    ("Llama-2-7B MBS=4", 16384, 4096, 11008),
    ("Llama-2-70B MBS=1", 4096, 10240, 8192),
    ("Llama-2-70B MBS=1", 4096, 8192, 8192),
    ("Llama-2-70B MBS=1", 4096, 57344, 8192),
    ("Llama-2-70B MBS=1", 4096, 8192, 28672),
    ("Llama-2-70B MBS=2", 8192, 10240, 8192),
    ("Llama-2-70B MBS=2", 8192, 8192, 8192),
    ("Llama-2-70B MBS=2", 8192, 57344, 8192),
    ("Llama-2-70B MBS=2", 8192, 8192, 28672),
    ("Llama-2-70B MBS=4", 16384, 10240, 8192),
    ("Llama-2-70B MBS=4", 16384, 8192, 8192),
    ("Llama-2-70B MBS=4", 16384, 57344, 8192),
    ("Llama-2-70B MBS=4", 16384, 8192, 28672),
    ("Llama-3.1-8B MBS=1", 8192, 6144, 4096),
    ("Llama-3.1-8B MBS=1", 8192, 4096, 4096),
    ("Llama-3.1-8B MBS=1", 8192, 28672, 4096),
    ("Llama-3.1-8B MBS=1", 8192, 4096, 14336),
    ("Llama-3.1-8B MBS=2", 16384, 6144, 4096),
    ("Llama-3.1-8B MBS=2", 16384, 4096, 4096),
    ("Llama-3.1-8B MBS=2", 16384, 28672, 4096),
    ("Llama-3.1-8B MBS=2", 16384, 4096, 14336),
    ("Llama-3.1-8B MBS=4", 32768, 6144, 4096),
    ("Llama-3.1-8B MBS=4", 32768, 4096, 4096),
    ("Llama-3.1-8B MBS=4", 32768, 28672, 4096),
    ("Llama-3.1-8B MBS=4", 32768, 4096, 14336),
    ("Llama-3.1-405B MBS=1", 8192, 18432, 16384),
    ("Llama-3.1-405B MBS=1", 8192, 16384, 16384),
    ("Llama-3.1-405B MBS=1", 8192, 106496, 16384),
    ("Llama-3.1-405B MBS=1", 8192, 16384, 53248),
    ("Llama-3.1-405B MBS=2", 16384, 18432, 16384),
    ("Llama-3.1-405B MBS=2", 16384, 16384, 16384),
    ("Llama-3.1-405B MBS=2", 16384, 106496, 16384),
    ("Llama-3.1-405B MBS=2", 16384, 16384, 53248),
    ("Llama-3.1-405B MBS=4", 32768, 18432, 16384),
    ("Llama-3.1-405B MBS=4", 32768, 16384, 16384),
    ("Llama-3.1-405B MBS=4", 32768, 106496, 16384),
    ("Llama-3.1-405B MBS=4", 32768, 16384, 53248),
    ("Qwen2.5-7B MBS=1", 8192, 4608, 3584),
    ("Qwen2.5-7B MBS=1", 8192, 3584, 3584),
    ("Qwen2.5-7B MBS=1", 8192, 37888, 3584),
    ("Qwen2.5-7B MBS=1", 8192, 3584, 18944),
    ("Qwen2.5-7B MBS=2", 16384, 4608, 3584),
    ("Qwen2.5-7B MBS=2", 16384, 3584, 3584),
    ("Qwen2.5-7B MBS=2", 16384, 37888, 3584),
    ("Qwen2.5-7B MBS=2", 16384, 3584, 18944),
    ("Qwen2.5-7B MBS=4", 32768, 4608, 3584),
    ("Qwen2.5-7B MBS=4", 32768, 3584, 3584),
    ("Qwen2.5-7B MBS=4", 32768, 37888, 3584),
    ("Qwen2.5-7B MBS=4", 32768, 3584, 18944),
    ("Qwen2.5-72B MBS=1", 8192, 10240, 8192),
    ("Qwen2.5-72B MBS=1", 8192, 8192, 8192),
    ("Qwen2.5-72B MBS=1", 8192, 59136, 8192),
    ("Qwen2.5-72B MBS=1", 8192, 8192, 29568),
    ("Qwen2.5-72B MBS=2", 16384, 10240, 8192),
    ("Qwen2.5-72B MBS=2", 16384, 8192, 8192),
    ("Qwen2.5-72B MBS=2", 16384, 59136, 8192),
    ("Qwen2.5-72B MBS=2", 16384, 8192, 29568),
    ("Qwen2.5-72B MBS=4", 32768, 10240, 8192),
    ("Qwen2.5-72B MBS=4", 32768, 8192, 8192),
    ("Qwen2.5-72B MBS=4", 32768, 59136, 8192),
    ("Qwen2.5-72B MBS=4", 32768, 8192, 29568),
    ("Mistral-7B MBS=1", 4096, 6144, 4096),
    ("Mistral-7B MBS=1", 4096, 4096, 4096),
    ("Mistral-7B MBS=1", 4096, 28672, 4096),
    ("Mistral-7B MBS=1", 4096, 4096, 14336),
    ("Mistral-7B MBS=2", 8192, 6144, 4096),
    ("Mistral-7B MBS=2", 8192, 4096, 4096),
    ("Mistral-7B MBS=2", 8192, 28672, 4096),
    ("Mistral-7B MBS=2", 8192, 4096, 14336),
    ("Mistral-7B MBS=4", 16384, 6144, 4096),
    ("Mistral-7B MBS=4", 16384, 4096, 4096),
    ("Mistral-7B MBS=4", 16384, 28672, 4096),
    ("Mistral-7B MBS=4", 16384, 4096, 14336),
]

# from test_blockscale_preshuffle_gemm.py
def select_tile_config(M: int, N: int, K: int, scale_block_k: int = 128):
    """Auto-select tile config for blockscale GEMM benchmarks."""
    candidates = [
        (16, 64, 256), (16, 128, 256),
        (32, 64, 128), (32, 64, 256), (32, 128, 128), (32, 128, 256),
        (64, 64, 128), (64, 64, 256), (64, 128, 128), (64, 128, 256), (64, 256, 128),
    ]

    def _valid(tm, tn, tk):
        return (N % tn == 0 and K % tk == 0 and tk % scale_block_k == 0
                and tm * tk // 256 >= 16)

    valid = [(tm, tn, tk) for tm, tn, tk in candidates if _valid(tm, tn, tk)]

    if not valid:
        return (64, 128, 128)

    def _score(tm, tn, tk):
        s = 0
        total_blocks = ((M + tm - 1) // tm) * (N // tn)
        s += 15 if total_blocks >= 256 else (10 if total_blocks >= 128 else (5 if total_blocks >= 64 else 0))
        if M <= 48:
            s += 12 if tm == 16 else (8 if tm == 32 else 0)
        elif M <= 128:
            s += 10 if tm == 32 else (6 if tm == 16 else (4 if tm == 64 else 0))
        elif M <= 512:
            s += 12 if tm == 64 else (8 if tm == 32 else 0)
        else:
            s += 12 if tm == 64 else 0
        if M <= 128:
            s += 6 if tn == 64 else (4 if tn == 128 else (2 if tn == 256 else 0))
        else:
            s += 8 if tn == 128 else (4 if tn == 64 else (4 if tn == 256 else 0))
        s += 6 if tk == 128 else 3
        return s

    return max(valid, key=lambda t: _score(*t))


def quantize_to_mxfp8(tensor: torch.Tensor, granule_k: int = SCALE_GRANULARITY_K) -> tuple:
    """Quantize tensor to MXFP8 format with per-block scales (FP32)."""
    *batch_dims, last_dim = tensor.shape
    if last_dim % granule_k != 0:
        raise ValueError(f"Last dimension {last_dim} not divisible by granule size {granule_k}")

    tensor_2d = tensor.view(-1, last_dim)
    m, k = tensor_2d.shape

    blocks = tensor_2d.view(m, k // granule_k, granule_k).float()
    block_max = blocks.abs().amax(dim=2)

    FP8_MAX = 448.0
    safe_max = torch.where(block_max > 0, block_max, torch.ones_like(block_max))
    raw_scale = safe_max / FP8_MAX
    raw_scale = torch.where(raw_scale > 0, raw_scale, torch.ones_like(raw_scale))
    log2_scale = torch.ceil(torch.log2(raw_scale))
    scales_fp32 = torch.pow(2.0, log2_scale)
    scales_fp32 = torch.where(block_max > 0, scales_fp32, torch.ones_like(scales_fp32))

    # Quantize blocks using FP32 scales, then convert result to FP8
    quantized = (blocks / scales_fp32.unsqueeze(-1)).to(DTYPE_FP8)
    quantized = quantized.reshape(m, k)

    quantized = quantized.view(*batch_dims, last_dim)
    scales_fp32 = scales_fp32.view(*batch_dims, -1)

    # Keep scales as FP32 (FlyDSL doesn't support E8M0 dtype directly)
    return quantized, scales_fp32.to(torch.float32)


def benchmark_shape(name, M, N, K, num_iters=20, num_warmup=3, out_dtype="bf16"):
    """Benchmark a single GEMM shape."""
    block_shape_n, block_shape_k = BLOCK_SHAPE
    scale_k = (K + SCALE_GRANULARITY_K - 1) // SCALE_GRANULARITY_K

    torch_out_dtype = torch.bfloat16 if out_dtype == "bf16" else torch.float16
    tile_m, tile_n, tile_k = select_tile_config(M, N, K, scale_block_k=block_shape_k)

    exe = compile_blockscale_preshuffle_gemm(
        M=M, N=N, K=K,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        scale_block_k=block_shape_k,
        out_dtype=out_dtype,
        use_async_copy=False,
    )

    device = torch.device("cuda")

    # Create inputs with MXFP8 quantization
    x_bf16 = (torch.rand((M, K), dtype=torch.bfloat16, device=device) / 10)
    weight_bf16 = (torch.rand((N, K), dtype=torch.bfloat16, device=device) / 10)

    x, x_scale = quantize_to_mxfp8(x_bf16)
    weight, w_scale = quantize_to_mxfp8(weight_bf16)

    b_shuffled = shuffle_weight(weight, layout=(16, 16))
    x_scale_t = x_scale.transpose(0, 1).contiguous().view(-1)
    w_scale_flat = w_scale.contiguous().view(-1)
    c_out = torch.zeros((M, N), dtype=torch_out_dtype, device=device)

    compiled_exe = flyc.compile(exe, c_out, x, b_shuffled, x_scale_t, w_scale_flat,
                               M, N, torch.cuda.current_stream())

    # Warmup
    for _ in range(num_warmup):
        compiled_exe(c_out, x, b_shuffled, x_scale_t, w_scale_flat, M, N, 
                    torch.cuda.current_stream())
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        compiled_exe(c_out, x, b_shuffled, x_scale_t, w_scale_flat, M, N, 
                    torch.cuda.current_stream())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    us_per_iter = (elapsed / num_iters) * 1e6
    flops = 2 * M * N * K
    tflops = flops / (us_per_iter / 1e6) / 1e12
    elem_bytes = 1  # fp8
    bytes_moved = (M * K * elem_bytes) + (N * K * elem_bytes) + (M * N * 2) + (M * scale_k + N // 128 * scale_k) * 4
    tbps = bytes_moved / 1e12 / (us_per_iter / 1e6)

    return {
        "name": name,
        "M": M, "N": N, "K": K,
        "tile_m": tile_m, "tile_n": tile_n, "tile_k": tile_k,
        "us": us_per_iter,
        "tflops": tflops,
        "tbps": tbps,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MXFP8 Blockscale GEMM benchmark")
    parser.add_argument("--num_iters", type=int, default=20, help="Number of benchmark iterations")
    parser.add_argument("--num_warmup", type=int, default=3, help="Number of warmup iterations")
    parser.add_argument("--out_dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--output_csv", type=str, default=None, help="Output CSV file")
    args = parser.parse_args()

    torch.set_default_device("cuda")

    results = []
    total = len(SHAPES)

    print(f"MXFP8 Blockscale GEMM Benchmark ({total} shapes)")

    for idx, (name, M, N, K) in enumerate(SHAPES, 1):
        print(f"\n[{idx}/{total}] {name} M={M}, N={N}, K={K}")
        result = benchmark_shape(name, M, N, K, num_iters=args.num_iters, 
                                num_warmup=args.num_warmup, out_dtype=args.out_dtype)
        print(f"  Time: {result['us']:.2f} us, {result['tflops']:.2f} TFLOPS")
        print(f"  Tile: {result['tile_m']}x{result['tile_n']}x{result['tile_k']}")
        results.append(result)

    # Write results to CSV
    if args.output_csv:
        with open(args.output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults written to {args.output_csv}")

    print(f"Successfully benchmarked {len(results)} shapes.")

    import statistics
    tflops_values = [r['tflops'] for r in results]
    median_tflops = statistics.median(tflops_values)
    mean_tflops = statistics.mean(tflops_values)
    min_tflops = min(tflops_values)
    max_tflops = max(tflops_values)
    
    print("Benchmark summary")
    print(f"Median TFLOPS: {median_tflops:.2f}")
    print(f"Mean TFLOPS:   {mean_tflops:.2f}")
    print(f"Min TFLOPS:    {min_tflops:.2f}")
    print(f"Max TFLOPS:    {max_tflops:.2f}")


if __name__ == "__main__":
    main()
