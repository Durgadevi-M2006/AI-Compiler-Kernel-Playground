# AI Compiler & Kernel Playground - Interview Preparation & Defense Guide

This document contains **20 comprehensive interview questions and model answers** designed to help you confidently present and defend this project during campus placements, internship evaluations, and technical interviews.

---

## Section 1: Project Overview & Motivation

### Q1: Can you give a 60-second elevator pitch of your project?
**Answer**:  
> "My project is the **AI Compiler & Kernel Playground**. AI workloads like Matrix Multiplication and Neural Network forward passes are computationally heavy and memory-bound. While Python is great for rapid prototyping, its dynamic interpreter and memory boxing introduce massive overheads.  
> In this project, I built an interactive playground that compares baseline Python against hardware-optimized **Mojo** kernels, explores **MLIR intermediate representation dialects** (`linalg`, `scf`, `vector`, `llvm`), and demonstrates **MAX runtime graph optimizations** like operator fusion. Across 5 real workloads, we achieve up to **$5,900\times$ speedups** over naive Python and exceed $70\text{ GFLOPS}$ through SIMD vectorization, cache tiling, and register-level fusion."

---

### Q2: Why did you choose Python and Mojo instead of just C++ and CUDA?
**Answer**:  
> "Python provides the essential high-level baseline and dynamic prototyping interface that AI researchers use daily. However, C++ and CUDA have steep learning curves, fragmented syntax, and lack unified multi-level compiler integration.  
> **Mojo** was designed by the creators of LLVM and Swift specifically for AI systems engineering. It provides Python syntax ergonomics with full systems-level control: compile-time types, explicit `SIMD` hardware vector registers, zero-cost memory pointers via `UnsafePointer`, and built-in `parallelize` abstractions that run across diverse CPU/GPU targets. It directly bridges the gap between high-level AI code and low-level compiler optimization."

---

## Section 2: Hardware, CPU Architecture & Mojo Optimization

### Q3: Why is Pure Python so slow for matrix multiplication and vector addition?
**Answer**:  
> "There are four fundamental reasons:
> 1. **Dynamic Boxing**: Every single floating-point number in Python is a heap-allocated `PyObject` structure (taking $\sim 24\text{ bytes}$), requiring pointer dereferencing rather than direct register values.
> 2. **Interpreter Overhead**: In a loop of $N$ iterations, Python checks types, increments reference counts, and dispatches bytecode on every single iteration.
> 3. **Lack of Vectorization**: Python's interpreter executes scalar operations one by one; it cannot automatically pack 8 floats into a 256-bit AVX register.
> 4. **Memory Locality / Cache Thrashing**: Python list pointers are scattered in memory, leading to continuous L1/L2 cache misses."

---

### Q4: How does SIMD work in Mojo and what performance gains does it produce?
**Answer**:  
> "SIMD stands for **Single Instruction, Multiple Data**. Modern CPUs have 256-bit (AVX2) or 512-bit (AVX-512) vector registers.  
> In Mojo, we use `SIMD[DType.float32, 8]`. When doing addition, the compiler emits a single `vaddps` assembly instruction that computes 8 float additions simultaneously in 1 clock cycle. In our benchmarks, adding explicit SIMD vectorization alone provided a **$50\times - 75\times$ speedup** over naive Python."

---

### Q5: What is Loop Tiling (Cache Blocking) and why is it critical for GEMM?
**Answer**:  
> "Matrix Multiplication is an $O(N^3)$ operation. For large matrices (e.g. $512 \times 512 = 1\text{ MB}$), data does not fit into the fast CPU L1 Data Cache ($32\text{ KB} - 64\text{ KB}$). When iterating through columns of Matrix B in naive loops, the CPU repeatedly evicts cache lines, stalling on DRAM bandwidth.  
> **Loop Tiling** partitions matrices into small 2D sub-blocks (e.g. $64 \times 64 = 16\text{ KB}$) that stay resident in L1 cache during the entire inner computation. This increases data reuse by $8\times - 15\times$ and allows arithmetic units to operate at peak throughput."

---

### Q6: How does Mojo's `parallelize` differ from Python's `threading`?
**Answer**:  
> "Python's `threading` module is constrained by the **Global Interpreter Lock (GIL)**, meaning only one thread can execute Python bytecode at a time on multi-core CPUs.  
> Mojo has no GIL. Its `algorithm.parallelize` function schedules tasks across a high-performance, work-stealing native thread pool, directly saturating all physical and logical CPU cores without runtime lock contention."

---

## Section 3: MLIR & AI Compilers

### Q7: What is MLIR and why is it better than LLVM IR for AI?
**Answer**:  
> "MLIR stands for **Multi-Level Intermediate Representation**. Traditional LLVM IR is low-level and flat (pointers and scalar registers); once high-level tensor operations like `conv2d` or `matmul` are translated to LLVM IR, the multidimensional loop nest structure is lost, making loop tiling and operator fusion difficult.  
> MLIR introduces **Dialects**. It allows the compiler to represent operations at multiple levels of abstraction (`linalg` for tensors $\to$ `scf`/`affine` for tiled loops $\to$ `vector` for SIMD $\to$ `llvm` for pointers), applying targeted transformations at the ideal abstraction layer before lowering to machine code."

---

### Q8: What are the main MLIR dialects you explored in this project?
**Answer**:  
> "We explored five core dialects:
> 1. **`linalg` (Linear Algebra)**: High-level structured operations (`linalg.matmul`, `linalg.generic`) with declarative affine indexing maps and iterator types (`parallel` vs `reduction`).
> 2. **`scf` (Structured Control Flow)**: Structured loops (`scf.for`) and conditionals (`scf.if`).
> 3. **`affine`**: Polyhedral loop transformations and cache tiling.
> 4. **`vector`**: Hardware SIMD register operations (`vector.transfer_read`, `vector.contract`, `vector.fma`, `vector.reduction`).
> 5. **`llvm`**: Direct low-level machine types (`llvm.ptr`, `llvm.fadd`, `llvm.store`) ready for binary generation."

---

### Q9: What happens during the `--convert-linalg-to-vector` pass?
**Answer**:  
> "This pass translates high-level tensor/memref operations into hardware vector register operations. For example, a tiled matrix multiplication loop is lowered into `vector.transfer_read` (loading vector register tiles) and `vector.contract` or `vector.fma` (hardware Fused-Multiply-Add), converting nested scalar operations into vectorized micro-kernels."

---

## Section 4: MAX Engine & Operator Fusion

### Q10: What is Operator Fusion and why is it crucial for Large Language Models (LLMs)?
**Answer**:  
> "In deep learning frameworks, operations like $\text{Linear}(X, W) \to \text{BiasAdd} \to \text{ReLU}$ are executed as separate kernel launches. In an unfused pipeline:
> 1. Kernel 1 computes GEMM and writes matrix $Z_1$ to DRAM.
> 2. Kernel 2 reads $Z_1$ from DRAM, adds bias, and writes $Z_2$ back to DRAM.
> 3. Kernel 3 reads $Z_2$ from DRAM, applies ReLU, and writes the output to DRAM.
> 
> Because DRAM is orders of magnitude slower than CPU registers, the pipeline stalls on memory bus bandwidth.  
> **Operator Fusion** merges all three operations into a single kernel. Accumulation, bias addition, and activation happen directly in CPU vector registers, cutting DRAM memory traffic by $\ge 50\%$ and drastically increasing arithmetic intensity."

---

### Q11: What is Arithmetic Intensity and how does it relate to the Roofline Model?
**Answer**:  
> "Arithmetic Intensity ($I$) is defined as:
> $$I = \frac{\text{Total Floating Point Operations (FLOPs)}}{\text{Total Memory Traffic (Bytes Transferred to/from DRAM)}}$$
> According to the **Roofline Model**, if a kernel has low arithmetic intensity, its performance is **memory-bandwidth bound**. By applying operator fusion and cache tiling, we increase arithmetic intensity, shifting the kernel from memory-bound to **compute-bound**, allowing it to reach peak hardware GFLOPS."

---

## Section 5: Verification, Engineering & System Design

### Q12: How did you ensure numerical correctness across implementations?
**Answer**:  
> "We wrote an automated test suite using `pytest` that strictly verifies mathematical equivalence:
> - For vector and matrix operations: `np.allclose(result, reference, atol=1e-5)`
> - For reductions: `math.isclose(res, ref, rel_tol=1e-4)`
> - For neural network layers: Comparing full softmax probability distributions against PyTorch reference tensors."

---

### Q13: If you had 3 more months on this project, what would you implement next?
**Answer**:  
> "I would expand the playground in three directions:
> 1. **GPU Kernel Support**: Implement equivalent kernels in Mojo targeting NVIDIA GPUs via PTX and WebGPU.
> 2. **FlashAttention Kernel**: Implement the fused FlashAttention-2 algorithm (online softmax tiling) in Mojo.
> 3. **Custom MLIR Pass Plugin**: Write a C++/LLVM custom MLIR compiler pass that automatically detects and fuses attention QKV projections."
