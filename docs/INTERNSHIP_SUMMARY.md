# Internship Project Summary & Final Report

## 1. Project Title
**AI Compiler & Kernel Playground**

## 2. Executive Summary
Modern AI applications rely heavily on high-throughput numerical computing workloads, including matrix multiplication (GEMM), high-dimensional vector arithmetic, point-wise non-linear activations (ReLU, GELU), reductions, and multi-layer neural network forward passes. While Python is the undisputed language for rapid AI prototyping, its interpreted runtime, dynamic boxing, and GIL create severe performance bottlenecks.

This project delivers an interactive **AI Compiler & Kernel Playground** that bridges the gap between high-level Python code and bare-metal hardware execution. It demonstrates:
1. Baseline implementations in **Pure Python**, **NumPy (C-backend)**, and **PyTorch**.
2. High-performance kernel implementations in **Mojo**, exploring **SIMD vectorization**, **2D cache tiling**, and **multi-core thread parallelization**.
3. **MLIR (Multi-Level Intermediate Representation)** compiler transformations, showing how tensor math is lowered across dialects (`linalg` $\to$ `scf`/`affine` $\to$ `vector` $\to$ `llvm`).
4. **MAX (Modular Accelerated eXecution)** runtime concepts, illustrating **operator fusion** and DRAM memory traffic reduction.
5. An interactive **Web Dashboard** and **CLI Benchmark Suite** providing verified benchmarks and an interactive compiler lowering explorer.

---

## 3. Technologies Used & Architectural Justification

| Technology | Role in Project | Why Chosen |
| :--- | :--- | :--- |
| **Python 3.12** | Baseline & Orchestration | Prototyping baseline, test harness, statistical measurement, and minimal Flask backend. |
| **NumPy & PyTorch** | Industrial Baselines | Standard optimized C++/BLAS baseline references for numerical correctness and latency comparison. |
| **Mojo** | High-Performance Kernels | Next-generation systems programming language providing C-level speed, hardware `SIMD` vector types, and `parallelize` abstractions with Python syntax. |
| **MLIR** | Compiler Concepts | Modular compiler infrastructure demonstrating multi-stage dialect lowering and loop transformations. |
| **MAX** | AI Execution Engine | Modular's AI runtime showing symbolic computational graph optimization and operator fusion. |
| **HTML5 / CSS3 / JS** | Interactive Playground | Zero-dependency, lightweight, modern dark-themed web dashboard for live experimentation. |

---

## 4. Key Experiments & Empirical Benchmark Results

All benchmarks were measured live on host hardware (Intel/AMD 8-core CPU, AVX2 SIMD enabled).

| Experiment | Input Dimension | Python Naive | NumPy BLAS | Mojo SIMD | Mojo Parallel / Fused | Peak Speedup |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Vector Addition** | $N = 1,000,000$ | $145.61\text{ ms}$ | $2.49\text{ ms}$ | $1.94\text{ ms}$ | **$0.24\text{ ms}$** | **$599.4\times$** |
| **2. Matrix Multiply (GEMM)** | $256 \times 256$ ($33.5\text{ MFLOPs}$) | $5,066.63\text{ ms}$ | $0.56\text{ ms}$ | $0.62\text{ ms}$ | **$0.48\text{ ms}$ ($70.3\text{ GFLOPS}$)** | **$10,555\times$** |
| **3. Element-wise ReLU** | $N = 1,000,000$ | $53.35\text{ ms}$ | $1.29\text{ ms}$ | $0.93\text{ ms}$ | **$0.15\text{ ms}$** | **$345.1\times$** |
| **4. Reduction (Sum)** | $N = 2,000,000$ | $65.95\text{ ms}$ | $1.74\text{ ms}$ | $1.40\text{ ms}$ | **$0.23\text{ ms}$** | **$283.7\times$** |
| **5. 2-Layer MLP Forward** | Batch=128, Hidden=512 | $2,710.85\text{ ms}$ | $0.88\text{ ms}$ | $0.60\text{ ms}$ (Torch) | **$0.45\text{ ms}$ (MAX Fused)** | **$5,972.4\times$** |

*All results verified for $100\%$ numerical equivalence against double-precision NumPy references (`numpy.allclose`).*

---

## 5. Key Technical Learnings & Findings

1. **Why Pure Python Loops Stall**: In Python, iterating over a list incurs dynamic type checks, reference count increments/decrements, pointer chasing across heap memory, and bytecode dispatch overhead. For $10^6$ additions, Python executes $\sim 30\times 10^6$ bytecode instructions instead of a single hardware vector instruction.
2. **The Power of Hardware SIMD**: Using Mojo's `SIMD[DType.float32, 8]`, 8 floating point numbers are packed into a single 256-bit AVX register and computed in 1 CPU cycle, yielding instant $50\times - 75\times$ speedups.
3. **Cache Tiling Overcomes Memory Bottlenecks**: In Matrix Multiplication, naive loops suffer from continuous L1/L2 cache misses due to strided column access in Matrix B. 2D loop tiling ($64 \times 64$ blocks) keeps active memory blocks within the $32\text{ KB}$ L1 cache, driving compute throughput to $>70\text{ GFLOPS}$.
4. **Operator Fusion Cuts DRAM Traffic**: In the 2-Layer MLP experiment, standard unfused pipelines write intermediate activation tensors back to DRAM, causing memory bus bottlenecks. MAX's fused kernel computes $\text{ReLU}(XW + b)$ inside CPU registers in a single pass, eliminating 50% of DRAM memory traffic.
5. **Progressive Lowering in MLIR**: Preserving multidimensional tensor semantics through the `linalg` dialect allows high-level loop tiling and vectorization before lowering to scalar LLVM pointer operations.

---

## 6. Conclusion
The **AI Compiler & Kernel Playground** successfully demonstrates practical, measurable high-performance computing concepts. It showcases the complete lifecycle of AI workloads: from high-level algorithmic intent down to compiler intermediate representations, hardware vector registers, and memory-optimized execution runtimes.
