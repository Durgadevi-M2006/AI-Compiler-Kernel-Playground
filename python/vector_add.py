"""
Experiment 1: Vector Addition (Python Implementations)

Demonstrates the performance characteristics of:
1. Naive Pure Python (loop-by-loop) - Bottlenecks: Dynamic type checking, interpreter overhead, boxing/unboxing
2. Vectorized NumPy - C-extension, SIMD vector instructions, contiguous memory layout
3. PyTorch Baseline - Hardware-optimized tensor execution
"""

import time
from typing import List, Tuple
import numpy as np


def vector_add_python_naive(a: List[float], b: List[float]) -> List[float]:
    """Pure Python element-by-element addition using a loop.
    
    Time Complexity: O(N)
    Memory: Allocates a new list of Python float objects.
    Compiler/Runtime Cost: Dynamic type checks on every iteration, GIL overhead, pointer chasing.
    """
    n = len(a)
    result = [0.0] * n
    for i in range(n):
        result[i] = a[i] + b[i]
    return result


def vector_add_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """NumPy vectorized vector addition.
    
    Time Complexity: O(N)
    Memory: Contiguous C-array buffer.
    Compiler/Runtime Cost: Executes compiled C/Fortran loop with AVX/SSE vector instructions.
    """
    return a + b


def run_vector_add_experiment(size: int = 1_000_000, runs: int = 5) -> dict:
    """Runs and benchmarks Python vector addition implementations."""
    # Generate random test data
    np_a = np.random.randn(size).astype(np.float32)
    np_b = np.random.randn(size).astype(np.float32)
    
    # NumPy benchmark (warmup + timed)
    _ = vector_add_numpy(np_a, np_b)
    numpy_times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        np_res = vector_add_numpy(np_a, np_b)
        t1 = time.perf_counter()
        numpy_times.append((t1 - t0) * 1000.0) # in ms
    numpy_median = float(np.median(numpy_times))

    # Python Naive benchmark (for large N, adjust runs if necessary to avoid hanging)
    py_runs = min(runs, 3) if size > 1_000_000 else runs
    py_a = np_a.tolist()
    py_b = np_b.tolist()
    
    # Warmup
    _ = vector_add_python_naive(py_a[:min(100, size)], py_b[:min(100, size)])
    
    py_times = []
    for _ in range(py_runs):
        t0 = time.perf_counter()
        py_res = vector_add_python_naive(py_a, py_b)
        t1 = time.perf_counter()
        py_times.append((t1 - t0) * 1000.0)
    py_median = float(np.median(py_times))

    # Verification: assert correctness
    is_correct = bool(np.allclose(np.array(py_res, dtype=np.float32), np_res, atol=1e-5))
    speedup = py_median / numpy_median if numpy_median > 0 else 1.0

    return {
        "experiment": "Vector Addition",
        "size": size,
        "python_naive_ms": round(py_median, 3),
        "numpy_vectorized_ms": round(numpy_median, 3),
        "speedup_numpy_vs_naive": round(speedup, 2),
        "correctness": is_correct,
        "bandwidth_gb_per_sec_numpy": round((3 * size * 4) / (numpy_median / 1000.0) / 1e9, 2)
    }


if __name__ == "__main__":
    result = run_vector_add_experiment(size=500_000)
    print("Vector Add Benchmark Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
