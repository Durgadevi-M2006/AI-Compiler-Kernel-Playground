# AI Compiler & Kernel Playground - System Architecture

This document details the architectural design, compilation pipeline, execution layers, and data flow of the **AI Compiler & Kernel Playground**.

---

## 1. High-Level System Architecture

The playground implements a modular, 5-tier architecture connecting high-level neural network code with low-level hardware execution.

```mermaid
graph TD
    subgraph UI_Layer ["1. Presentation & Visualization Layer"]
        A1["Interactive Web Dashboard (HTML5 / Vanilla CSS / JS)"]
        A2["MLIR Lowering Pipeline Explorer"]
        A3["MAX Graph & Operator Fusion Visualizer"]
        A4["CLI Benchmark Suite (Rich Terminal)"]
    end

    subgraph Service_Layer ["2. API & Service Dispatch Layer"]
        B1["Flask Lightweight REST Backend"]
        B2["Hardware & Toolchain Discovery Engine"]
        B3["Unified Benchmark Runner & Verifier"]
    end

    subgraph Compute_Engines ["3. Dual-Track Kernel Execution Layer"]
        subgraph Python_Track ["Python Baseline Track"]
            C1["Pure Python Naive Loops (O(N^3) GEMM, Dynamic Boxing)"]
            C2["NumPy Vectorized C-Wrappers (SIMD AVX-256 / OpenBLAS)"]
            C3["PyTorch ATen C++ Runtime"]
        end

        subgraph Mojo_Track ["Mojo High-Performance Track"]
            D1["Native Scalar Pointer Kernels"]
            D2["SIMD Vectorized Registers (8/16-wide Float32)"]
            D3["2D/3D Cache-Tiled Memory Block Kernels"]
            D4["Multi-Threaded Parallel SIMD Kernels"]
        end
    end

    subgraph Compiler_Layer ["4. MLIR Compiler Infrastructure Layer"]
        E1["linalg Dialect (Declarative Tensor Operations)"]
        E2["scf / affine Dialects (Loop Nests & Cache Tiling)"]
        E3["vector Dialect (SIMD Register FMA & Contracts)"]
        E4["llvm Dialect (Low-Level Pointer & Intrinsics)"]
        E5["Fusion Passes (--linalg-fuse-elementwise-ops)"]
    end

    subgraph Runtime_Layer ["5. MAX Engine & Hardware Layer"]
        F1["MAX Graph Construction & Symbolic IR"]
        F2["Graph Optimizer (Fusion, Layout, DCE)"]
        F3["CPU Hardware (Registers, L1/L2 Cache, DRAM)"]
    end

    UI_Layer --> Service_Layer
    Service_Layer --> Compute_Engines
    Compute_Engines -.-> Compiler_Layer
    Compiler_Layer --> Runtime_Layer
```

---

## 2. Layer-by-Layer Breakdown

### Layer 1: Presentation & Interactive Explorer
- **Interactive Playground (`frontend/`)**: Configures input tensor dimensions ($N=10^5 \dots 10^7$, Matrix $64 \times 64 \dots 512 \times 512$), executes live benchmarks, visualizes execution times on logarithmic bar charts, and checks numerical correctness ($c_i = \text{ref}_i$).
- **MLIR Visualizer**: Explains lowering transformations across 6 progressive stages (`source` $\to$ `linalg` $\to$ `scf` $\to$ `vector` $\to$ `llvm` $\to$ `assembly`).
- **MAX Graph Inspector**: Displays before-and-after operator fusion pipelines and quantifies DRAM memory traffic reductions.

### Layer 2: API & Dispatch Engine (`backend/`)
- **`app.py`**: Minimal REST API providing endpoints for environment discovery (`/api/environment`), experiment catalog (`/api/experiments`), execution runner (`/api/run`), MLIR pipeline metadata (`/api/mlir-pipeline`), and MAX graphs (`/api/max-graph`).
- **`environment.py`**: Auto-detects CPU model, physical/logical core counts, SIMD vector widths, Python versions, and Mojo/MAX installations.
- **`runner.py`**: Manages warmup passes, timing loops ($K$ repetitions), median latency calculations, GFLOPS estimation, and memory bandwidth (GB/s).

### Layer 3: Kernel Implementation Tracks
- **Python Track (`python/`)**: Serves as the experimental baseline. Demonstrates the performance costs of interpreted bytecode, dynamic type dispatch, and pointer chasing in pure Python.
- **Mojo Track (`mojo/`)**: Implements hardware-tuned kernels using Mojo's zero-cost systems abstractions (`SIMD`, `UnsafePointer`, `parallelize`, and cache tiling).

### Layer 4: MLIR Compiler Dialect Hierarchy (`mlir/`)
Shows the multi-stage lowering process from mathematical tensor operations to machine code:
1. `linalg.matmul` / `linalg.generic`: High-level domain representation.
2. `scf.for` / `affine.for`: Structured loops with cache tiling parameters ($64 \times 64$).
3. `vector.transfer_read` / `vector.contract`: SIMD register allocation and hardware Fused-Multiply-Add (FMA).
4. `llvm.load` / `llvm.fadd` / `llvm.store`: Direct machine pointer instructions.

### Layer 5: MAX Engine Runtime & Hardware Execution (`max/`)
- Demonstrates how the Modular Accelerated eXecution (MAX) engine optimizes computational graphs by eliminating intermediate tensor allocations in DRAM, keeping data resident in CPU L1/L2 cache and vector registers.

---

## 3. Data Flow During an Experiment Run

```text
User selects [Matrix Multiplication 256x256] in UI
                   │
                   ▼
Frontend issues POST /api/run { experiment: "matrix_multiply", params: { n: 256 } }
                   │
                   ▼
Runner initializes input matrices A, B in memory
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
Run Python Baselines    Run Mojo / Hardware Kernels
(Naive i-j-k, NumPy)   (SIMD, 2D Tiled, Parallel)
         │                   │
         └─────────┬─────────┘
                   ▼
Numerical Verification: np.allclose(result, reference, atol=1e-3)
                   │
                   ▼
Metrics Computed: Latency (ms), GFLOPS (2*N^3 / time), Speedup Factor
                   │
                   ▼
JSON Response sent to UI -> Bar Chart & KPI Cards animated
```
