# ==============================================================================
# AI Compiler & Kernel Playground
# Experiment 4: Reduction Operations in Mojo
#
# Demonstrates how compilers & kernels overcome loop-carried dependencies:
# 1. Scalar Reduction (serial accumulation, loop-carried dependency)
# 2. SIMD Vector Reduction (parallel vector accumulation + horizontal add)
# 3. Parallel Block Reduction (multi-core reduction tree)
# ==============================================================================

from sys.info import simdwidthof
from algorithm import vectorize, parallelize
from memory import UnsafePointer
from time import perf_counter_ns

alias dtype = DType.float32
alias simd_width = simdwidthof[dtype]()


# ------------------------------------------------------------------------------
# 1. Baseline: Scalar Reduction (Loop-carried dependency)
# ------------------------------------------------------------------------------
fn sum_scalar(data: UnsafePointer[Scalar[dtype]], size: Int) -> Scalar[dtype]:
    var total: Scalar[dtype] = 0.0
    for i in range(size):
        total += data[i]
    return total


# ------------------------------------------------------------------------------
# 2. SIMD Vector Reduction
# ------------------------------------------------------------------------------
fn sum_simd(data: UnsafePointer[Scalar[dtype]], size: Int) -> Scalar[dtype]:
    """Uses a SIMD vector accumulator to break serial dependency into simd_width parallel paths."""
    var vec_acc = SIMD[dtype, simd_width](0.0)

    @parameter
    fn accum[width: Int](idx: Int):
        var v = data.load[width=width](idx)
        @parameter
        if width == simd_width:
            vec_acc += v
        else:
            # Remainder tail handling
            for i in range(width):
                vec_acc[0] += v[i]

    vectorize[accum, simd_width](size)

    # Horizontal reduction of the vector accumulator
    return vec_acc.reduce_add()


# ------------------------------------------------------------------------------
# 3. Parallel Multi-Core Reduction
# ------------------------------------------------------------------------------
fn sum_parallel(data: UnsafePointer[Scalar[dtype]], size: Int) -> Scalar[dtype]:
    alias block_size = 128 * 1024 # 128K elements per thread block
    var num_blocks = (size + block_size - 1) // block_size
    var block_results = UnsafePointer[Scalar[dtype]].alloc(num_blocks)

    @parameter
    fn process_block(block_idx: Int):
        var start_idx = block_idx * block_size
        var current_len = min(block_size, size - start_idx)
        var local_acc = SIMD[dtype, simd_width](0.0)

        @parameter
        fn accum_chunk[width: Int](offset: Int):
            var v = data.load[width=width](start_idx + offset)
            @parameter
            if width == simd_width:
                local_acc += v
            else:
                for i in range(width):
                    local_acc[0] += v[i]

        vectorize[accum_chunk, simd_width](current_len)
        block_results[block_idx] = local_acc.reduce_add()

    parallelize[process_block](num_blocks)

    # Final reduction of block results
    var final_sum: Scalar[dtype] = 0.0
    for b in range(num_blocks):
        final_sum += block_results[b]

    block_results.free()
    return final_sum


# ------------------------------------------------------------------------------
# Main Benchmark & Verification
# ------------------------------------------------------------------------------
fn main():
    print("==================================================")
    print("  Mojo Reduction Operations (SIMD & Parallel)     ")
    print("==================================================")
    alias N = 10_000_000
    print("Array Size N =", N)

    var data = UnsafePointer[Scalar[dtype]].alloc(N)
    for i in range(N):
        data[i] = 1.0 # Expected sum = 10,000,000.0

    # 1. Scalar Benchmark
    var t0 = perf_counter_ns()
    var res_scalar = sum_scalar(data, N)
    var t1 = perf_counter_ns()
    var scalar_ms = (t1 - t0) / 1_000_000.0
    print("1. Scalar Sum:   ", scalar_ms, "ms | Result =", res_scalar)

    # 2. SIMD Benchmark
    t0 = perf_counter_ns()
    var res_simd = sum_simd(data, N)
    t1 = perf_counter_ns()
    var simd_ms = (t1 - t0) / 1_000_000.0
    print("2. SIMD Sum:     ", simd_ms, "ms | Result =", res_simd, "| Speedup:", scalar_ms / simd_ms, "x")

    # 3. Parallel SIMD Benchmark
    t0 = perf_counter_ns()
    var res_par = sum_parallel(data, N)
    t1 = perf_counter_ns()
    var par_ms = (t1 - t0) / 1_000_000.0
    print("3. Parallel Sum: ", par_ms, "ms | Result =", res_par, "| Speedup:", scalar_ms / par_ms, "x")

    data.free()
