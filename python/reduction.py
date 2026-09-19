"""
Experiment 4: Reduction Operations (Python Implementations)

Demonstrates data reduction workloads (reducing N inputs to 1 scalar):
1. Sum Reduction: sum(X) = x_0 + x_1 + ... + x_{N-1}
2. Max Reduction: max(X)
3. L2 Norm (Euclidean): sqrt(sum(X^2))

Key Compiler Concept:
- Loop Carried Dependency: Serial reduction forces sequential data dependency.
- Tree / Parallel Reduction: Associative reordering enables tree-based SIMD vector reduction.
"""

import math
import time
from typing import List
import numpy as np


def sum_python_naive(x: List[float]) -> float:
    """Sequential scalar sum with loop-carried dependency."""
    total = 0.0
    for val in x:
        total += val
    return total


def sum_numpy(x: np.ndarray) -> float:
    """NumPy SIMD-accelerated reduction."""
    return float(np.sum(x))


def max_python_naive(x: List[float]) -> float:
    """Sequential scalar max."""
    if not x:
        return 0.0
    m = x[0]
    for val in x[1:]:
        if val > m:
            m = val
    return m


def max_numpy(x: np.ndarray) -> float:
    """NumPy SIMD max."""
    return float(np.max(x))


def l2_norm_python_naive(x: List[float]) -> float:
    """Sequential L2 Euclidean norm."""
    total_sq = 0.0
    for val in x:
        total_sq += val * val
    return math.sqrt(total_sq)


def l2_norm_numpy(x: np.ndarray) -> float:
    """NumPy L2 norm."""
    return float(np.linalg.norm(x))


def run_reduction_experiment(op_type: str = "sum", size: int = 1_000_000, runs: int = 5) -> dict:
    """Runs and benchmarks reduction operations."""
    np_x = np.random.randn(size).astype(np.float32)
    
    if op_type == "sum":
        _ = sum_numpy(np_x)
        t0 = time.perf_counter()
        for _ in range(runs):
            np_res = sum_numpy(np_x)
        t1 = time.perf_counter()
        numpy_time = ((t1 - t0) / runs) * 1000.0
        
        py_x = np_x.tolist()
        t0 = time.perf_counter()
        py_res = sum_python_naive(py_x)
        t1 = time.perf_counter()
        python_time = (t1 - t0) * 1000.0
        
    elif op_type == "max":
        _ = max_numpy(np_x)
        t0 = time.perf_counter()
        for _ in range(runs):
            np_res = max_numpy(np_x)
        t1 = time.perf_counter()
        numpy_time = ((t1 - t0) / runs) * 1000.0
        
        py_x = np_x.tolist()
        t0 = time.perf_counter()
        py_res = max_python_naive(py_x)
        t1 = time.perf_counter()
        python_time = (t1 - t0) * 1000.0
        
    elif op_type == "norm":
        _ = l2_norm_numpy(np_x)
        t0 = time.perf_counter()
        for _ in range(runs):
            np_res = l2_norm_numpy(np_x)
        t1 = time.perf_counter()
        numpy_time = ((t1 - t0) / runs) * 1000.0
        
        py_x = np_x.tolist()
        t0 = time.perf_counter()
        py_res = l2_norm_python_naive(py_x)
        t1 = time.perf_counter()
        python_time = (t1 - t0) * 1000.0
        
    else:
        raise ValueError(f"Unknown reduction op: {op_type}")
        
    is_correct = math.isclose(py_res, np_res, rel_tol=1e-3, abs_tol=1e-3)
    speedup = python_time / numpy_time if numpy_time > 0 else 1.0

    return {
        "experiment": f"Reduction ({op_type.upper()})",
        "size": size,
        "python_naive_ms": round(python_time, 3),
        "numpy_vectorized_ms": round(numpy_time, 3),
        "speedup_numpy_vs_naive": round(speedup, 2),
        "correctness": is_correct
    }


if __name__ == "__main__":
    for op in ["sum", "max", "norm"]:
        res = run_reduction_experiment(op, size=500_000)
        print(f"Reduction {op}: {res}")
