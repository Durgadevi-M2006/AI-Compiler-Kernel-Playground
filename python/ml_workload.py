"""
Experiment 5: AI / ML Numerical Workload (Multi-Layer Perceptron Forward Pass)

Demonstrates end-to-end neural network inference:
Input (Batch, In_Features) 
  -> Linear_1 (Weight W1 + Bias b1)
  -> ReLU Activation
  -> Linear_2 (Weight W2 + Bias b2)
  -> Softmax Classifier

Key Compiler Concept:
- Operator Fusion (Horizontal & Vertical Fusion):
  Merging GEMM + Add_Bias + ReLU into a single fused compute kernel.
  Eliminates writing intermediate tensors Z1 and A1 to main memory.
"""

import math
import time
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn


def mlp_forward_python_naive(
    x: List[List[float]],
    w1: List[List[float]],
    b1: List[float],
    w2: List[List[float]],
    b2: List[float]
) -> List[List[float]]:
    """Pure Python scalar implementation of a 2-layer MLP."""
    batch_size = len(x)
    in_dim = len(x[0])
    hidden_dim = len(w1[0])
    out_dim = len(w2[0])
    
    # Layer 1: GEMM + Bias + ReLU
    z1 = [[0.0 for _ in range(hidden_dim)] for _ in range(batch_size)]
    for b in range(batch_size):
        for h in range(hidden_dim):
            acc = b1[h]
            for i in range(in_dim):
                acc += x[b][i] * w1[i][h]
            # Inline fused ReLU
            z1[b][h] = acc if acc > 0.0 else 0.0
            
    # Layer 2: GEMM + Bias
    z2 = [[0.0 for _ in range(out_dim)] for _ in range(batch_size)]
    for b in range(batch_size):
        for o in range(out_dim):
            acc = b2[o]
            for h in range(hidden_dim):
                acc += z1[b][h] * w2[h][o]
            z2[b][o] = acc

    # Softmax
    probs = [[0.0 for _ in range(out_dim)] for _ in range(batch_size)]
    for b in range(batch_size):
        max_val = max(z2[b])
        exp_sum = sum(math.exp(val - max_val) for val in z2[b])
        for o in range(out_dim):
            probs[b][o] = math.exp(z2[b][o] - max_val) / exp_sum
            
    return probs


def mlp_forward_numpy(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray
) -> np.ndarray:
    """NumPy vectorized MLP forward pass."""
    # Layer 1: X @ W1 + b1 -> ReLU
    z1 = np.maximum(0.0, np.matmul(x, w1) + b1)
    # Layer 2: Z1 @ W2 + b2
    z2 = np.matmul(z1, w2) + b2
    # Softmax with numerical stability
    exp_shifted = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
    probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
    return probs


def mlp_forward_torch(x_t: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """PyTorch optimized forward pass."""
    with torch.no_grad():
        return model(x_t)


def run_ml_workload_experiment(
    batch_size: int = 64,
    in_dim: int = 128,
    hidden_dim: int = 256,
    out_dim: int = 10,
    runs: int = 5
) -> dict:
    """Runs and benchmarks the ML forward pass workload across implementations."""
    # Initialize weights
    np.random.seed(42)
    x = np.random.randn(batch_size, in_dim).astype(np.float32)
    w1 = np.random.randn(in_dim, hidden_dim).astype(np.float32) * 0.01
    b1 = np.zeros(hidden_dim, dtype=np.float32)
    w2 = np.random.randn(hidden_dim, out_dim).astype(np.float32) * 0.01
    b2 = np.zeros(out_dim, dtype=np.float32)

    # PyTorch Setup
    torch_model = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
        nn.Softmax(dim=1)
    )
    with torch.no_grad():
        torch_model[0].weight.copy_(torch.from_numpy(w1.T))
        torch_model[0].bias.copy_(torch.from_numpy(b1))
        torch_model[2].weight.copy_(torch.from_numpy(w2.T))
        torch_model[2].bias.copy_(torch.from_numpy(b2))
    torch_x = torch.from_numpy(x)

    # NumPy benchmark
    _ = mlp_forward_numpy(x, w1, b1, w2, b2)
    numpy_times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        np_out = mlp_forward_numpy(x, w1, b1, w2, b2)
        t1 = time.perf_counter()
        numpy_times.append((t1 - t0) * 1000.0)
    numpy_median = float(np.median(numpy_times))

    # PyTorch benchmark
    _ = mlp_forward_torch(torch_x, torch_model)
    torch_times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        torch_out = mlp_forward_torch(torch_x, torch_model)
        t1 = time.perf_counter()
        torch_times.append((t1 - t0) * 1000.0)
    torch_median = float(np.median(torch_times))

    # Python Naive benchmark
    py_x = x.tolist()
    py_w1 = w1.tolist()
    py_b1 = b1.tolist()
    py_w2 = w2.tolist()
    py_b2 = b2.tolist()
    
    t0 = time.perf_counter()
    py_out = mlp_forward_python_naive(py_x, py_w1, py_b1, py_w2, py_b2)
    t1 = time.perf_counter()
    python_time = (t1 - t0) * 1000.0

    is_correct = bool(np.allclose(np.array(py_out, dtype=np.float32), np_out, atol=1e-4))
    speedup = python_time / numpy_median if numpy_median > 0 else 1.0

    return {
        "experiment": "2-Layer MLP Forward Inference",
        "dimensions": f"Batch={batch_size}, In={in_dim}, Hidden={hidden_dim}, Out={out_dim}",
        "python_naive_ms": round(python_time, 2),
        "numpy_vectorized_ms": round(numpy_median, 3),
        "torch_inference_ms": round(torch_median, 3),
        "speedup_numpy_vs_naive": round(speedup, 1),
        "correctness": is_correct
    }


if __name__ == "__main__":
    res = run_ml_workload_experiment()
    print("ML Workload Benchmark:")
    for k, v in res.items():
        print(f"  {k}: {v}")
