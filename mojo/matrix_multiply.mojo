# ==============================================================================
# AI Compiler & Kernel Playground
# Experiment 2: High-Performance Matrix Multiplication (GEMM) in Mojo
#
# Highlights the standard high-performance computing (HPC) optimization pipeline:
# 1. Naive 3-loop GEMM (baseline scalar execution)
# 2. Vectorized Inner Loop (SIMD FMA: Fused Multiply-Add)
# 3. 2D Block Tiling (Cache locality optimization for L1/L2 cache lines)
# 4. Multi-Threaded Parallel Tiling (Scales across all CPU threads)
# ==============================================================================

from sys.info import simdwidthof
from algorithm import vectorize, parallelize
from memory import UnsafePointer
from time import perf_counter_ns

alias dtype = DType.float32
alias simd_width = simdwidthof[dtype]() # 8 floats on AVX2, 16 floats on AVX-512


# ------------------------------------------------------------------------------
# 1. Baseline: Naive Triple Loop
# ------------------------------------------------------------------------------
fn matmul_naive(
    c: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    b: UnsafePointer[Scalar[dtype]],
    m: Int, n: Int, k_dim: Int
):
    """Naive O(N^3) matrix multiplication."""
    for i in range(m):
        for j in range(n):
            var acc: Scalar[dtype] = 0.0
            for k in range(k_dim):
                acc += a[i * k_dim + k] * b[k * n + j]
            c[i * n + j] = acc


# ------------------------------------------------------------------------------
# 2. Vectorized Inner Loop with SIMD FMA
# ------------------------------------------------------------------------------
fn matmul_vectorized(
    c: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    b: UnsafePointer[Scalar[dtype]],
    m: Int, n: Int, k_dim: Int
):
    """Loop reordering (i -> k -> j) + SIMD vectorization across row j."""
    for i in range(m):
        for k in range(k_dim):
            var a_ik = a[i * k_dim + k]

            @parameter
            fn vectorize_j[width: Int](j: Int):
                var b_vec = b.load[width=width](k * n + j)
                var c_vec = c.load[width=width](i * n + j)
                # Fused Multiply Add: c = a_ik * b + c
                c.store[width=width](i * n + j, c_vec + a_ik * b_vec)

            vectorize[vectorize_j, simd_width](n)


# ------------------------------------------------------------------------------
# 3. 2D Cache Tiled + SIMD Vectorized GEMM
# ------------------------------------------------------------------------------
fn matmul_tiled(
    c: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    b: UnsafePointer[Scalar[dtype]],
    m: Int, n: Int, k_dim: Int
):
    """2D Loop Tiling ensuring sub-blocks reside in L1/L2 cache."""
    alias tile_m = 64
    alias tile_n = 64
    alias tile_k = 64

    for im in range(0, m, tile_m):
        var m_end = min(im + tile_m, m)
        for km in range(0, k_dim, tile_k):
            var k_end = min(km + tile_k, k_dim)
            for jm in range(0, n, tile_n):
                var n_end = min(jm + tile_n, n)

                # Micro-kernel over the tile
                for i in range(im, m_end):
                    for k in range(km, k_end):
                        var a_ik = a[i * k_dim + k]

                        @parameter
                        fn tile_simd_j[width: Int](j_offset: Int):
                            var j = jm + j_offset
                            var b_vec = b.load[width=width](k * n + j)
                            var c_vec = c.load[width=width](i * n + j)
                            c.store[width=width](i * n + j, c_vec + a_ik * b_vec)

                        vectorize[tile_simd_j, simd_width](n_end - jm)


# ------------------------------------------------------------------------------
# 4. Multi-Threaded Parallel Tiled GEMM
# ------------------------------------------------------------------------------
fn matmul_parallel_tiled(
    c: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    b: UnsafePointer[Scalar[dtype]],
    m: Int, n: Int, k_dim: Int
):
    """Combines 2D Tiling, SIMD Vectorization, and Multi-Core Parallelism."""
    alias tile_m = 64
    alias tile_n = 64
    alias tile_k = 64

    @parameter
    fn process_m_tile(m_block_idx: Int):
        var im = m_block_idx * tile_m
        var m_end = min(im + tile_m, m)

        for km in range(0, k_dim, tile_k):
            var k_end = min(km + tile_k, k_dim)
            for jm in range(0, n, tile_n):
                var n_end = min(jm + tile_n, n)

                for i in range(im, m_end):
                    for k in range(km, k_end):
                        var a_ik = a[i * k_dim + k]

                        @parameter
                        fn par_simd_j[width: Int](j_offset: Int):
                            var j = jm + j_offset
                            var b_vec = b.load[width=width](k * n + j)
                            var c_vec = c.load[width=width](i * n + j)
                            c.store[width=width](i * n + j, c_vec + a_ik * b_vec)

                        vectorize[par_simd_j, simd_width](n_end - jm)

    var num_m_blocks = (m + tile_m - 1) // tile_m
    parallelize[process_m_tile](num_m_blocks)


# ------------------------------------------------------------------------------
# Main Benchmark & GFLOPS Evaluation
# ------------------------------------------------------------------------------
fn main():
    print("==================================================")
    print("  Mojo Matrix Multiplication (GEMM) Optimization  ")
    print("==================================================")
    alias N = 512
    var total_flops = 2.0 * Float64(N) * Float64(N) * Float64(N)

    print("Matrix Dimension:", N, "x", N)
    print("Total Compute Operations:", total_flops / 1e9, "GFLOPs")

    var a = UnsafePointer[Scalar[dtype]].alloc(N * N)
    var b = UnsafePointer[Scalar[dtype]].alloc(N * N)
    var c = UnsafePointer[Scalar[dtype]].alloc(N * N)

    # Initialize matrices
    for i in range(N * N):
        a[i] = 0.5
        b[i] = 1.0
        c[i] = 0.0

    # 1. Benchmark Naive
    var t0 = perf_counter_ns()
    matmul_naive(c, a, b, N, N, N)
    var t1 = perf_counter_ns()
    var naive_ms = (t1 - t0) / 1_000_000.0
    var naive_gflops = (total_flops / (naive_ms / 1000.0)) / 1e9
    print("1. Naive Scalar GEMM:     ", naive_ms, "ms |", naive_gflops, "GFLOPS")

    # Clear C
    for i in range(N * N): c[i] = 0.0

    # 2. Benchmark Vectorized
    t0 = perf_counter_ns()
    matmul_vectorized(c, a, b, N, N, N)
    t1 = perf_counter_ns()
    var vec_ms = (t1 - t0) / 1_000_000.0
    var vec_gflops = (total_flops / (vec_ms / 1000.0)) / 1e9
    print("2. Vectorized SIMD GEMM:  ", vec_ms, "ms |", vec_gflops, "GFLOPS | Speedup:", naive_ms / vec_ms, "x")

    # Clear C
    for i in range(N * N): c[i] = 0.0

    # 3. Benchmark Tiled SIMD
    t0 = perf_counter_ns()
    matmul_tiled(c, a, b, N, N, N)
    t1 = perf_counter_ns()
    var tiled_ms = (t1 - t0) / 1_000_000.0
    var tiled_gflops = (total_flops / (tiled_ms / 1000.0)) / 1e9
    print("3. Tiled + SIMD GEMM:     ", tiled_ms, "ms |", tiled_gflops, "GFLOPS | Speedup:", naive_ms / tiled_ms, "x")

    # Clear C
    for i in range(N * N): c[i] = 0.0

    # 4. Benchmark Parallel Tiled SIMD
    t0 = perf_counter_ns()
    matmul_parallel_tiled(c, a, b, N, N, N)
    t1 = perf_counter_ns()
    var par_ms = (t1 - t0) / 1_000_000.0
    var par_gflops = (total_flops / (par_ms / 1000.0)) / 1e9
    print("4. Parallel Tiled SIMD:   ", par_ms, "ms |", par_gflops, "GFLOPS | Speedup:", naive_ms / par_ms, "x")

    # Verification: c[i] should be N * 0.5 * 1.0 = N * 0.5 = 256.0
    var is_correct = True
    for i in range(min(500, N * N)):
        if abs(c[i] - (Float32(N) * 0.5)) > 0.01:
            is_correct = False
            break
    print("Result Correctness Check:", "PASSED" if is_correct else "FAILED")

    a.free()
    b.free()
    c.free()
