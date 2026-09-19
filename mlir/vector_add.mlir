// ==============================================================================
// MLIR Lowering Pipeline: Vector Addition (Float32)
// Dialect progression: linalg -> scf -> vector -> llvm
// ==============================================================================

// ------------------------------------------------------------------------------
// STAGE 1: High-Level Representation (linalg Dialect)
// Expresses compute intent without tying to specific loop structures or hardware.
// ------------------------------------------------------------------------------
module @stage1_linalg {
  func.func @vec_add_linalg(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>) {
    linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%arg0, %arg1 : memref<1024xf32>, memref<1024xf32>)
      outs(%arg2 : memref<1024xf32>) {
    ^bb0(%in1: f32, %in2: f32, %out: f32):
      %sum = arith.addf %in1, %in2 : f32
      linalg.yield %sum : f32
    }
    return
  }
}

// ------------------------------------------------------------------------------
// STAGE 2: Loop Lowering (scf Dialect - Structured Control Flow)
// Lowering pass: --convert-linalg-to-loops
// ------------------------------------------------------------------------------
module @stage2_scf {
  func.func @vec_add_scf(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>) {
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index

    scf.for %i = %c0 to %c1024 step %c1 {
      %val_a = memref.load %arg0[%i] : memref<1024xf32>
      %val_b = memref.load %arg1[%i] : memref<1024xf32>
      %sum = arith.addf %val_a, %val_b : f32
      memref.store %sum, %arg2[%i] : memref<1024xf32>
    }
    return
  }
}

// ------------------------------------------------------------------------------
// STAGE 3: Hardware Vectorization (vector Dialect)
// Lowering pass: --affine-vectorize="virtual-vector-size=8"
// Maps 8 floats to a 256-bit AVX vector register.
// ------------------------------------------------------------------------------
module @stage3_vector {
  func.func @vec_add_vector(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>) {
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c8 = arith.constant 8 : index
    %c0_f32 = arith.constant 0.0 : f32

    scf.for %i = %c0 to %c1024 step %c8 {
      %vec_a = vector.transfer_read %arg0[%i], %c0_f32 : memref<1024xf32>, vector<8xf32>
      %vec_b = vector.transfer_read %arg1[%i], %c0_f32 : memref<1024xf32>, vector<8xf32>
      %vec_sum = arith.addf %vec_a, %vec_b : vector<8xf32>
      vector.transfer_write %vec_sum, %arg2[%i] : vector<8xf32>, memref<1024xf32>
    }
    return
  }
}

// ------------------------------------------------------------------------------
// STAGE 4: Low-Level Machine IR (llvm Dialect)
// Lowering pass: --convert-vector-to-llvm --convert-func-to-llvm
// Direct 1:1 mapping to LLVM instructions for final machine code generation.
// ------------------------------------------------------------------------------
module @stage4_llvm {
  llvm.func @vec_add_llvm(%ptr_a: !llvm.ptr, %ptr_b: !llvm.ptr, %ptr_c: !llvm.ptr, %n: i64) {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c8 = llvm.mlir.constant(8 : i64) : i64

    // Vector load, add, store instruction sequence
    %vec_a = llvm.load %ptr_a : !llvm.ptr -> vector<8xf32>
    %vec_b = llvm.load %ptr_b : !llvm.ptr -> vector<8xf32>
    %sum = llvm.fadd %vec_a, %vec_b : vector<8xf32>
    llvm.store %sum, %ptr_c : vector<8xf32>, !llvm.ptr
    llvm.return
  }
}
