# ==============================================================================
# AI Compiler & Kernel Playground
# Experiment 1: Vector Addition in Mojo
#
# Highlights Mojo's performance hierarchy:
# 1. Naive Scalar Loop (C-like efficiency, no Python interpreter overhead)
# 2. SIMD Vectorized (utilizing hardware vector registers: AVX-256 / AVX-512)
# 3. Parallel SIMD (distributing vector tiles across multiple CPU cores)
# ==============================================================================

from sys.info import simdwidthof
from algorithm import vectorize, parallelize
from memory import UnsafePointer
from time import perf_counter_ns

alias dtype = DType.float32
alias simd_width = simdwidthof[dtype]() # e.g. 8 for AVX2, 16 for AVX-512


# ------------------------------------------------------------------------------
# 1. Baseline: Naive Scalar Loop
# ------------------------------------------------------------------------------
fn vector_add_naive(
    c: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    b: UnsafePointer[Scalar[dtype]],
    size: Int
):
    """Scalar element-by-element addition.
    Direct pointer arithmetic without GC or boxing overhead.
    """
    for i in range(size):
        c[i] = a[i] + b[i]


# ------------------------------------------------------------------------------
# 2. Optimized: SIMD Vectorization
# ------------------------------------------------------------------------------
fn vector_add_simd(
    c: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    b: UnsafePointer[Scalar[dtype]],
    size: Int
):
    """Vectorized kernel using Mojo's vectorized higher-order function.
    Emits SIMD instructions (e.g. vmovups / vaddps) for width chunks.
    """
    @parameter
    fn compute_simd[width: Int](idx: Int):
        var a_vec = a.load[width=width](idx)
        var b_vec = b.load[width=width](idx)
        c.store[width=width](idx, a_vec + b_vec)

    vectorize[compute_simd, simd_width](size)


# ------------------------------------------------------------------------------
# 3. Highly Optimized: Parallel SIMD Kernel
# ------------------------------------------------------------------------------
fn vector_add_parallel(
    c: UnsafePointer[Scalar[dtype]],
    a: UnsafePointer[Scalar[dtype]],
    b: UnsafePointer[Scalar[dtype]],
    size: Int
):
    """Multi-threaded parallelized kernel dividing work across CPU cores."""
    alias tile_size = 64 * 1024 # 64K elements per thread chunk

    @parameter
    fn process_tile(tile_idx: Int):
        var start_idx = tile_idx * tile_size
        var current_chunk = min(tile_size, size - start_idx)
        
        @parameter
        fn chunk_simd[width: Int](offset: Int):
            var idx = start_idx + offset
            var a_vec = a.load[width=width](idx)
            var b_vec = b.load[width=width](idx)
            c.store[width=width](idx, a_vec + b_vec)

        vectorize[chunk_simd, simd_width](current_chunk)

    var num_tiles = (size + tile_size - 1) // tile_size
    parallelize[process_tile](num_tiles)


# ------------------------------------------------------------------------------
# Main Benchmark & Verification Harness
# ------------------------------------------------------------------------------
fn main():
    print("==================================================")
    print("  Mojo Vector Addition Playground (SIMD & Parallel)")
    print("==================================================")
    print("Hardware SIMD Width for Float32:", simd_width)
    
    alias N = 10_000_000
    print("Allocating vectors of size N =", N, "(", (N * 4 * 3) / (1024 * 1024), "MB )")

    var a = UnsafePointer[Scalar[dtype]].alloc(N)
    var b = UnsafePointer[Scalar[dtype]].alloc(N)
    var c = UnsafePointer[Scalar[dtype]].alloc(N)

    # Initialize data
    for i in range(N):
        a[i] = 1.5
        b[i] = 2.5
        c[i] = 0.0

    # 1. Benchmark Naive Scalar
    var t0 = perf_counter_ns()
    vector_add_naive(c, a, b, N)
    var t1 = perf_counter_ns()
    var naive_ms = (t1 - t0) / 1_000_000.0
    print("1. Naive Scalar Mojo:    ", naive_ms, "ms")

    # 2. Benchmark SIMD Vectorized
    t0 = perf_counter_ns()
    vector_add_simd(c, a, b, N)
    t1 = perf_counter_ns()
    var simd_ms = (t1 - t0) / 1_000_000.0
    print("2. SIMD Vectorized Mojo: ", simd_ms, "ms | Speedup:", naive_ms / simd_ms, "x")

    # 3. Benchmark Parallel SIMD
    t0 = perf_counter_ns()
    vector_add_parallel(c, a, b, N)
    t1 = perf_counter_ns()
    var par_ms = (t1 - t0) / 1_000_000.0
    print("3. Parallel SIMD Mojo:   ", par_ms, "ms | Speedup:", naive_ms / par_ms, "x")

    # Verification check
    var is_correct = True
    for i in range(min(1000, N)):
        if c[i] != 4.0:
            is_correct = False
            break
    print("Result Correctness Check:", "PASSED [c[i] == 4.0]" if is_correct else "FAILED")

    # Free memory
    a.free()
    b.free()
    c.free()
