// ==============================================================================
// MLIR Optimization Pipeline: Operator Fusion (MatMul + ReLU)
// Demonstrates how compiler fusion eliminates DRAM memory round trips.
// ==============================================================================

// ------------------------------------------------------------------------------
// BEFORE FUSION: Unfused Separate Passes (2 Memory Writes, 1 Intermediate Buffer)
// ------------------------------------------------------------------------------
module @unfused_matmul_relu {
  func.func @unfused(%A: memref<128x256xf32>, %B: memref<256x512xf32>, %Out: memref<128x512xf32>) {
    // 1. Allocate expensive intermediate buffer in DRAM
    %Z = memref.alloc() : memref<128x512xf32>
    
    // 2. Compute MatMul and write %Z to memory
    linalg.matmul ins(%A, %B : memref<128x256xf32>, memref<256x512xf32>)
                  outs(%Z : memref<128x512xf32>)

    // 3. Read %Z from memory, apply ReLU, and write %Out to memory
    %c0_f32 = arith.constant 0.0 : f32
    linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%Z : memref<128x512xf32>) outs(%Out : memref<128x512xf32>) {
    ^bb0(%z_val: f32, %out_val: f32):
      %cmp = arith.cmpf ogt, %z_val, %c0_f32 : f32
      %relu = arith.select %cmp, %z_val, %c0_f32 : f32
      linalg.yield %relu : f32
    }
    
    memref.dealloc %Z : memref<128x512xf32>
    return
  }
}

// ------------------------------------------------------------------------------
// AFTER FUSION: Fused Kernel (Zero Intermediate DRAM Buffer, Pure Register Lifetime)
// Compiler pass: --linalg-fuse-elementwise-ops
// ------------------------------------------------------------------------------
module @fused_matmul_relu {
  func.func @fused(%A: memref<128x256xf32>, %B: memref<256x512xf32>, %Out: memref<128x512xf32>) {
    %c0_f32 = arith.constant 0.0 : f32
    
    // MatMul + ReLU computed directly inside register tile without %Z allocation
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>, // Matrix A
        affine_map<(d0, d1, d2) -> (d2, d1)>, // Matrix B
        affine_map<(d0, d1, d2) -> (d0, d1)>  // Output
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%A, %B : memref<128x256xf32>, memref<256x512xf32>)
      outs(%Out : memref<128x512xf32>) {
    ^bb0(%a_val: f32, %b_val: f32, %out_val: f32):
      %prod = arith.mulf %a_val, %b_val : f32
      %sum = arith.addf %prod, %out_val : f32
      // Fused ReLU inline
      %cmp = arith.cmpf ogt, %sum, %c0_f32 : f32
      %relu = arith.select %cmp, %sum, %c0_f32 : f32
      linalg.yield %relu : f32
    }
    return
  }
}
