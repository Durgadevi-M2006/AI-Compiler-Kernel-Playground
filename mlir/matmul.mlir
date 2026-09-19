// ==============================================================================
// MLIR Lowering Pipeline: Matrix Multiplication (GEMM)
// Dialect progression: linalg.matmul -> tiled loops -> vector.contract / fma -> llvm
// ==============================================================================

// ------------------------------------------------------------------------------
// STAGE 1: High-Level Named Op (linalg.matmul)
// ------------------------------------------------------------------------------
module @stage1_linalg_matmul {
  func.func @matmul_linalg(%A: tensor<512x512xf32>, %B: tensor<512x512xf32>, %C: tensor<512x512xf32>) -> tensor<512x512xf32> {
    %res = linalg.matmul
      ins(%A, %B : tensor<512x512xf32>, tensor<512x512xf32>)
      outs(%C : tensor<512x512xf32>) -> tensor<512x512xf32>
    return %res : tensor<512x512xf32>
  }
}

// ------------------------------------------------------------------------------
// STAGE 2: Cache Tiled Loop Nests
// Lowering pass: --linalg-tile="tile-sizes=64,64,64"
// ------------------------------------------------------------------------------
module @stage2_tiled_loops {
  func.func @matmul_tiled(%A: memref<512x512xf32>, %B: memref<512x512xf32>, %C: memref<512x512xf32>) {
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c64 = arith.constant 64 : index

    // 2D Outer Tile Loops (L1/L2 Cache Slicing)
    scf.for %i_tile = %c0 to %c512 step %c64 {
      scf.for %j_tile = %c0 to %c512 step %c64 {
        scf.for %k_tile = %c0 to %c512 step %c64 {
          // Subview creation and micro-kernel execution
          %sub_a = memref.subview %A[%i_tile, %k_tile] [64, 64] [1, 1] : memref<512x512xf32> to memref<64x64xf32>
          %sub_b = memref.subview %B[%k_tile, %j_tile] [64, 64] [1, 1] : memref<512x512xf32> to memref<64x64xf32>
          %sub_c = memref.subview %C[%i_tile, %j_tile] [64, 64] [1, 1] : memref<512x512xf32> to memref<64x64xf32>
          linalg.matmul ins(%sub_a, %sub_b : memref<64x64xf32>, memref<64x64xf32>) outs(%sub_c : memref<64x64xf32>)
        }
      }
    }
    return
  }
}

// ------------------------------------------------------------------------------
// STAGE 3: Vectorized Micro-Kernel with FMA (vector Dialect)
// Lowering pass: --convert-linalg-to-vector
// ------------------------------------------------------------------------------
module @stage3_vector_fma {
  func.func @matmul_vector_micro_kernel(%A_sub: memref<64x64xf32>, %B_sub: memref<64x64xf32>, %C_sub: memref<64x64xf32>) {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c8 = arith.constant 8 : index
    %c0_f32 = arith.constant 0.0 : f32

    scf.for %i = %c0 to %c64 step %c8 {
      scf.for %j = %c0 to %c64 step %c8 {
        // Load accumulator vector tile (8x8 floats)
        %c_tile = vector.transfer_read %C_sub[%i, %j], %c0_f32 : memref<64x64xf32>, vector<8x8xf32>
        
        %res_tile = scf.for %k = %c0 to %c64 step %c8 iter_args(%acc = %c_tile) -> (vector<8x8xf32>) {
          %a_tile = vector.transfer_read %A_sub[%i, %k], %c0_f32 : memref<64x64xf32>, vector<8x8xf32>
          %b_tile = vector.transfer_read %B_sub[%k, %j], %c0_f32 : memref<64x64xf32>, vector<8x8xf32>
          // Hardware Fused Multiply Add (vector.contract / fma)
          %fma_res = vector.contract {
            indexing_maps = [
              affine_map<(d0, d1, d2) -> (d0, d2)>,
              affine_map<(d0, d1, d2) -> (d2, d1)>,
              affine_map<(d0, d1, d2) -> (d0, d1)>
            ],
            iterator_types = ["parallel", "parallel", "reduction"],
            kind = #vector.kind<add>
          } %a_tile, %b_tile, %acc : vector<8x8xf32>, vector<8x8xf32> into vector<8x8xf32>
          scf.yield %fma_res : vector<8x8xf32>
        }
        
        vector.transfer_write %res_tile, %C_sub[%i, %j] : vector<8x8xf32>, memref<64x64xf32>
      }
    }
    return
  }
}
