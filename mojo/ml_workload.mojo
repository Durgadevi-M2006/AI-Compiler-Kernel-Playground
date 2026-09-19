# ==============================================================================
# AI Compiler & Kernel Playground
# Experiment 5: Fused Neural Network Layer in Mojo
#
# Key Compiler Concept: Operator Fusion
# Unfused: Input -> GEMM -> Store Z to DRAM -> Load Z -> ReLU -> Store A to DRAM
# Fused:   Input -> [GEMM + Bias + ReLU inside registers/L1] -> Store A to DRAM
# Saves 50% memory bandwidth and increases arithmetic intensity.
# ==============================================================================

from sys.info import simdwidthof
from algorithm import vectorize, parallelize
from memory import UnsafePointer
from time import perf_counter_ns

alias dtype = DType.float32
alias simd_width = simdwidthof[dtype]()


# ------------------------------------------------------------------------------
# Fused Linear + Bias + ReLU Kernel
# ------------------------------------------------------------------------------
fn fused_dense_relu_layer(
    out_ptr: UnsafePointer[Scalar[dtype]],
    in_ptr: UnsafePointer[Scalar[dtype]],
    weight_ptr: UnsafePointer[Scalar[dtype]],
    bias_ptr: UnsafePointer[Scalar[dtype]],
    batch_size: Int,
    in_features: Int,
    out_features: Int
):
    """Computes: Out = ReLU(X @ W + Bias) with zero intermediate DRAM writes."""
    alias tile_b = 16
    alias tile_o = 64

    @parameter
    fn process_batch_tile(b_tile_idx: Int):
        var b_start = b_tile_idx * tile_b
        var b_end = min(b_start + tile_b, batch_size)

        for b in range(b_start, b_end):
            for o_start in range(0, out_features, tile_o):
                var o_end = min(o_start + tile_o, out_features)

                @parameter
                fn simd_out[width: Int](o_offset: Int):
                    var o = o_start + o_offset
                    # Load bias into accumulator register
                    var acc = bias_ptr.load[width=width](o)

                    # Accumulate matrix multiply: x_row @ W_cols
                    for k in range(in_features):
                        var x_val = in_ptr[b * in_features + k]
                        var w_vec = weight_ptr.load[width=width](k * out_features + o)
                        acc += x_val * w_vec

                    # Fused ReLU activation right in registers before writing to memory
                    var zero = SIMD[dtype, width](0.0)
                    var activated = (acc > zero).select(acc, zero)

                    # Write directly to final output tensor
                    out_ptr.store[width=width](b * out_features + o, activated)

                vectorize[simd_out, simd_width](o_end - o_start)

    var num_batch_tiles = (batch_size + tile_b - 1) // tile_b
    parallelize[process_batch_tile](num_batch_tiles)


# ------------------------------------------------------------------------------
# Main Benchmark & Demonstration
# ------------------------------------------------------------------------------
fn main():
    print("==================================================")
    print("  Mojo Fused Neural Network Layer (GEMM + ReLU)   ")
    print("==================================================")
    alias B = 128
    alias IN_DIM = 256
    alias OUT_DIM = 512

    print("Batch Size:", B, "| In:", IN_DIM, "| Out:", OUT_DIM)
    var total_flops = 2.0 * Float64(B) * Float64(IN_DIM) * Float64(OUT_DIM)

    var x = UnsafePointer[Scalar[dtype]].alloc(B * IN_DIM)
    var w = UnsafePointer[Scalar[dtype]].alloc(IN_DIM * OUT_DIM)
    var b = UnsafePointer[Scalar[dtype]].alloc(OUT_DIM)
    var out = UnsafePointer[Scalar[dtype]].alloc(B * OUT_DIM)

    # Initialize
    for i in range(B * IN_DIM): x[i] = 0.1
    for i in range(IN_DIM * OUT_DIM): w[i] = 0.05
    for i in range(OUT_DIM): b[i] = -0.5
    for i in range(B * OUT_DIM): out[i] = 0.0

    var t0 = perf_counter_ns()
    fused_dense_relu_layer(out, x, w, b, B, IN_DIM, OUT_DIM)
    var t1 = perf_counter_ns()
    var duration_ms = (t1 - t0) / 1_000_000.0
    var gflops = (total_flops / (duration_ms / 1000.0)) / 1e9

    print("Fused Layer Execution Time:", duration_ms, "ms")
    print("Compute Throughput:        ", gflops, "GFLOPS")

    # Check output
    var pass_check = True
    for i in range(min(100, B * OUT_DIM)):
        if out[i] < 0.0:
            pass_check = False
            break
    print("ReLU Non-negativity Verification:", "PASSED" if pass_check else "FAILED")

    x.free()
    w.free()
    b.free()
    out.free()
