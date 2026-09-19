"""
Experiment 3: Element-wise Operations (Python Implementations)

Demonstrates point-wise neural network activation functions & operations:
1. ReLU (Rectified Linear Unit): y = max(0, x)
2. GELU (Gaussian Error Linear Unit): Modern Transformer activation (BERT, GPT, LLaMA)
3. Sigmoid: y = 1 / (1 + exp(-x))
4. Element-wise Multiplication (Hadamard product): y = a * b

Key Compiler Concept:
- Element-wise operations are memory-bandwidth bound.
- Fusion prevents intermediate memory round-trips to DRAM.
"""

import math
import time
from typing import List
import numpy as np


def relu_python_naive(x: List[float]) -> List[float]:
    """Pure Python scalar ReLU."""
    return [val if val > 0.0 else 0.0 for val in x]


def relu_numpy(x: np.ndarray) -> np.ndarray:
    """NumPy vectorized ReLU."""
    return np.maximum(0.0, x)


def gelu_python_naive(x: List[float]) -> List[float]:
    """Pure Python scalar GELU (tanh approximation)."""
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    result = [0.0] * len(x)
    for i, val in enumerate(x):
        inner = sqrt_2_over_pi * (val + 0.044715 * (val ** 3))
        result[i] = 0.5 * val * (1.0 + math.tanh(inner))
    return result


def gelu_numpy(x: np.ndarray) -> np.ndarray:
    """NumPy vectorized GELU."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x ** 3))))


def sigmoid_numpy(x: np.ndarray) -> np.ndarray:
    """NumPy vectorized Sigmoid."""
    return 1.0 / (1.0 + np.exp(-x))


def run_elementwise_experiment(op_type: str = "relu", size: int = 500_000, runs: int = 5) -> dict:
    """Runs and benchmarks element-wise activation operations."""
    np_x = np.random.randn(size).astype(np.float32)
    
    if op_type == "relu":
        # NumPy
        _ = relu_numpy(np_x)
        t0 = time.perf_counter()
        for _ in range(runs):
            np_out = relu_numpy(np_x)
        t1 = time.perf_counter()
        numpy_time = ((t1 - t0) / runs) * 1000.0
        
        # Python
        py_x = np_x.tolist()
        t0 = time.perf_counter()
        py_out = relu_python_naive(py_x)
        t1 = time.perf_counter()
        python_time = (t1 - t0) * 1000.0
        
    elif op_type == "gelu":
        # NumPy
        _ = gelu_numpy(np_x)
        t0 = time.perf_counter()
        for _ in range(runs):
            np_out = gelu_numpy(np_x)
        t1 = time.perf_counter()
        numpy_time = ((t1 - t0) / runs) * 1000.0
        
        # Python
        py_x = np_x.tolist()
        t0 = time.perf_counter()
        py_out = gelu_python_naive(py_x)
        t1 = time.perf_counter()
        python_time = (t1 - t0) * 1000.0
        
    else:
        raise ValueError(f"Unknown op_type: {op_type}")
        
    is_correct = bool(np.allclose(np.array(py_out, dtype=np.float32), np_out, atol=1e-4))
    speedup = python_time / numpy_time if numpy_time > 0 else 1.0

    return {
        "experiment": f"Element-wise {op_type.upper()}",
        "size": size,
        "python_naive_ms": round(python_time, 3),
        "numpy_vectorized_ms": round(numpy_time, 3),
        "speedup_numpy_vs_naive": round(speedup, 2),
        "correctness": is_correct
    }


if __name__ == "__main__":
    for op in ["relu", "gelu"]:
        res = run_elementwise_experiment(op, size=300_000)
        print(f"Elementwise {op.upper()}: {res}")
