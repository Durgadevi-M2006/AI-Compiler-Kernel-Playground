# ==============================================================================
# AI Compiler & Kernel Playground
# Experiment 3: Element-wise Operations & Activations in Mojo
#
# Highlights memory-bandwidth bound operations and SIMD vectorization:
# 1. Vectorized ReLU (simd.max)
# 2. Vectorized GELU (math approximations in vector registers)
# 3. Vectorized Element-wise Multiplication (Hadamard Product)
# ==============================================================================

from sys.info import simdwidthof
from algorithm import vectorize, parallelize
from memory import UnsafePointer
from math import tanh, sqrt
from time import perf_counter_ns

alias dtype = DType.float32
alias simd_width = simdwidthof[dtype]()


# ------------------------------------------------------------------------------
# 1. Vectorized ReLU Activation: y = max(0.0, x)
# ------------------------------------------------------------------------------
fn relu_simd(
    out_ptr: UnsafePointer[Scalar[dtype]],
    in_ptr: UnsafePointer[Scalar[dtype]],
    size: Int
):
    @parameter
    fn relu_kernel[width: Int](idx: Int):
        var v = in_ptr.load[width=width](idx)
        # Vectorized conditional max against 0.0
        var zero = SIMD[dtype, width](0.0)
        var res = (v > zero).select(v, zero)
        out_ptr.store[width=width](idx, res)

    vectorize[relu_kernel, simd_width](size)


# ------------------------------------------------------------------------------
# 2. Vectorized GELU Activation
# ------------------------------------------------------------------------------
fn gelu_simd(
    out_ptr: UnsafePointer[Scalar[dtype]],
    in_ptr: UnsafePointer[Scalar[dtype]],
    size: Int
):
    alias sqrt_2_over_pi: Scalar[dtype] = 0.7978845608 # sqrt(2 / pi)
    alias coeff: Scalar[dtype] = 0.044715

    @parameter
    fn gelu_kernel[width: Int](idx: Int):
        var x = in_ptr.load[width=width](idx)
        # Vectorized polynomial approximation
        var x_cubed = x * x * x
        var inner = sqrt_2_over_pi * (x + coeff * x_cubed)
        
        # Approximate tanh elementwise for SIMD vector
        var res = SIMD[dtype, width](0.0)
        for i in range(width):
            var tanh_val = tanh(inner[i])
            res[i] = 0.5 * x[i] * (1.0 + tanh_val)
            
        out_ptr.store[width=width](idx, res)

    vectorize[gelu_kernel, simd_width](size)


# ------------------------------------------------------------------------------
# 3. Vectorized Element-wise Multiplication (a * b)
# ------------------------------------------------------------------------------
fn elemwise_mul_simd(
    out_ptr: UnsafePointer[Scalar[dtype]],
    a_ptr: UnsafePointer[Scalar[dtype]],
    b_ptr: UnsafePointer[Scalar[dtype]],
    size: Int
):
    @parameter
    fn mul_kernel[width: Int](idx: Int):
        var va = a_ptr.load[width=width](idx)
        var vb = b_ptr.load[width=width](idx)
        out_ptr.store[width=width](idx, va * vb)

    vectorize[mul_kernel, simd_width](size)


# ------------------------------------------------------------------------------
# Main Benchmark & Verification
# ------------------------------------------------------------------------------
fn main():
    print("==================================================")
    print("  Mojo Element-wise Activations (ReLU / GELU)    ")
    print("==================================================")
    alias N = 5_000_000
    print("Array Size N =", N)

    var in_data = UnsafePointer[Scalar[dtype]].alloc(N)
    var out_data = UnsafePointer[Scalar[dtype]].alloc(N)

    for i in range(N):
        in_data[i] = Float32(i % 100) - 50.0 # Range: -50.0 to +49.0
        out_data[i] = 0.0

    # 1. Benchmark ReLU
    var t0 = perf_counter_ns()
    relu_simd(out_data, in_data, N)
    var t1 = perf_counter_ns()
    var relu_ms = (t1 - t0) / 1_000_000.0
    print("1. Vectorized ReLU: ", relu_ms, "ms | Bandwidth:", (2 * N * 4) / (relu_ms / 1000.0) / 1e9, "GB/s")

    # 2. Benchmark GELU
    t0 = perf_counter_ns()
    gelu_simd(out_data, in_data, N)
    t1 = perf_counter_ns()
    var gelu_ms = (t1 - t0) / 1_000_000.0
    print("2. Vectorized GELU: ", gelu_ms, "ms")

    in_data.free()
    out_data.free()
