// ==============================================================================
// MLIR Lowering Pipeline: Reduction (Sum / Max)
// Dialect progression: linalg.generic reduction -> scf.for -> vector.reduction
// ==============================================================================

module @reduction_pipeline {
  // Stage 1: linalg reduction
  func.func @sum_linalg(%input: tensor<1024xf32>, %init: tensor<f32>) -> tensor<f32> {
    %res = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
    } ins(%input : tensor<1024xf32>) outs(%init : tensor<f32>) {
    ^bb0(%in: f32, %acc: f32):
      %sum = arith.addf %in, %acc : f32
      linalg.yield %sum : f32
    } -> tensor<f32>
    return %res : tensor<f32>
  }

  // Stage 2: Hardware Vector Reduction using vector dialect
  func.func @sum_vector(%input: memref<1024xf32>) -> f32 {
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c8 = arith.constant 8 : index
    %c0_f32 = arith.constant 0.0 : f32
    %vec_zero = vector.broadcast %c0_f32 : f32 to vector<8xf32>

    // Accumulate in 8-lane SIMD vector register
    %final_vec = scf.for %i = %c0 to %c1024 step %c8 iter_args(%acc = %vec_zero) -> (vector<8xf32>) {
      %val = vector.transfer_read %input[%i], %c0_f32 : memref<1024xf32>, vector<8xf32>
      %next_acc = arith.addf %val, %acc : vector<8xf32>
      scf.yield %next_acc : vector<8xf32>
    }

    // Horizontal tree reduction to single scalar
    %scalar_sum = vector.reduction <add>, %final_vec : vector<8xf32> into f32
    return %scalar_sum : f32
  }
}
