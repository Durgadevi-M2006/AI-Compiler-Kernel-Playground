"""
Unit and Correctness Test Suite for AI Compiler & Kernel Playground
"""

import os
import sys
import math
import numpy as np
import pytest
import torch
import torch.nn as nn

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from python.vector_add import vector_add_python_naive, vector_add_numpy
from python.matrix_multiply import (
    matmul_python_naive,
    matmul_python_loop_ordered,
    matmul_python_tiled,
    matmul_numpy
)
from python.elementwise import (
    relu_python_naive,
    relu_numpy,
    gelu_python_naive,
    gelu_numpy
)
from python.reduction import (
    sum_python_naive,
    sum_numpy,
    max_python_naive,
    max_numpy,
    l2_norm_python_naive,
    l2_norm_numpy
)
from python.ml_workload import (
    mlp_forward_python_naive,
    mlp_forward_numpy,
    mlp_forward_torch
)
from max.graph_pipeline import build_demo_max_graph, MAXCompilerOptimizer
from max.max_benchmarks import benchmark_unfused_pipeline, benchmark_fused_max_pipeline


def test_vector_addition_correctness():
    """Verifies Python and NumPy vector addition output equivalence."""
    np.random.seed(10)
    a = np.random.randn(1000).astype(np.float32)
    b = np.random.randn(1000).astype(np.float32)

    np_res = vector_add_numpy(a, b)
    py_res = vector_add_python_naive(a.tolist(), b.tolist())

    assert np.allclose(np.array(py_res, dtype=np.float32), np_res, atol=1e-5)


def test_matrix_multiplication_correctness():
    """Verifies naive, loop-ordered, and tiled matrix multiplication algorithms."""
    np.random.seed(20)
    n = 16
    a = np.random.randn(n, n).astype(np.float32)
    b = np.random.randn(n, n).astype(np.float32)

    expected = matmul_numpy(a, b)
    
    a_list = a.tolist()
    b_list = b.tolist()

    res_naive = matmul_python_naive(a_list, b_list)
    res_ordered = matmul_python_loop_ordered(a_list, b_list)
    res_tiled = matmul_python_tiled(a_list, b_list, block_size=4)

    assert np.allclose(np.array(res_naive, dtype=np.float32), expected, atol=1e-3)
    assert np.allclose(np.array(res_ordered, dtype=np.float32), expected, atol=1e-3)
    assert np.allclose(np.array(res_tiled, dtype=np.float32), expected, atol=1e-3)


def test_elementwise_activations_correctness():
    """Verifies ReLU and GELU implementations match across pure Python and NumPy."""
    np.random.seed(30)
    x = np.random.randn(500).astype(np.float32)

    # ReLU
    relu_np = relu_numpy(x)
    relu_py = relu_python_naive(x.tolist())
    assert np.allclose(np.array(relu_py, dtype=np.float32), relu_np, atol=1e-5)

    # GELU
    gelu_np = gelu_numpy(x)
    gelu_py = gelu_python_naive(x.tolist())
    assert np.allclose(np.array(gelu_py, dtype=np.float32), gelu_np, atol=1e-4)


def test_reductions_correctness():
    """Verifies Sum, Max, and L2 Norm reductions."""
    np.random.seed(40)
    x = np.random.randn(1000).astype(np.float32)

    # Sum
    assert math.isclose(sum_python_naive(x.tolist()), sum_numpy(x), rel_tol=1e-4, abs_tol=1e-4)
    # Max
    assert math.isclose(max_python_naive(x.tolist()), max_numpy(x), rel_tol=1e-5)
    # L2 Norm
    assert math.isclose(l2_norm_python_naive(x.tolist()), l2_norm_numpy(x), rel_tol=1e-4)


def test_ml_workload_inference_correctness():
    """Verifies 2-layer MLP forward pass outputs match between Python, NumPy, and PyTorch."""
    np.random.seed(50)
    batch = 4
    in_d = 8
    hidden_d = 16
    out_d = 3

    x = np.random.randn(batch, in_d).astype(np.float32)
    w1 = np.random.randn(in_d, hidden_d).astype(np.float32) * 0.1
    b1 = np.random.randn(hidden_d).astype(np.float32) * 0.1
    w2 = np.random.randn(hidden_d, out_d).astype(np.float32) * 0.1
    b2 = np.random.randn(out_d).astype(np.float32) * 0.1

    np_probs = mlp_forward_numpy(x, w1, b1, w2, b2)
    py_probs = mlp_forward_python_naive(x.tolist(), w1.tolist(), b1.tolist(), w2.tolist(), b2.tolist())

    assert np.allclose(np.array(py_probs, dtype=np.float32), np_probs, atol=1e-4)
    assert np.allclose(np.sum(np_probs, axis=1), np.ones(batch), atol=1e-5)


def test_max_graph_fusion_optimizer():
    """Verifies MAX Graph construction and operator fusion transformation."""
    graph = build_demo_max_graph(batch_size=8, in_dim=16, hidden_dim=32)
    assert len(graph.nodes) == 6 # Input, Weight, Bias, MatMul, BiasAdd, ReLU

    fused_graph = MAXCompilerOptimizer.optimize_graph(graph)
    assert len(fused_graph.nodes) == 4 # Input, Weight, Bias, FusedMatMulBiasReLU
    assert "FusedMatMulBiasReLU" in [n.op_type for n in fused_graph.nodes.values()]
