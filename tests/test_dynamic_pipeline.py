import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.services.analyzer import (
    analyze_code_optimizations,
    generate_computation_graph,
    generate_mlir_for_program,
    PythonCodeInspector
)

client = TestClient(app)


def test_python_code_inspector_matrix_mult():
    code = """
def matmul(N=32):
    a = [[1.0]*N for _ in range(N)]
    b = [[2.0]*N for _ in range(N)]
    c = [[0.0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                c[i][j] += a[i][k] * b[k][j]
"""
    inspector = PythonCodeInspector()
    info = inspector.inspect(code)
    assert info["loop_depth"] == 3
    assert info["is_matrix_op"] is True
    assert "matmul" in info["function_names"]


def test_python_code_inspector_reduction():
    code = """
def sum_all(arr):
    total = 0.0
    for x in arr:
        total += x
    return total
"""
    inspector = PythonCodeInspector()
    info = inspector.inspect(code)
    assert info["is_reduction"] is True
    assert info["loop_depth"] == 1


def test_python_code_inspector_custom_code():
    code = """
def calculate(data):
    out = []
    for item in data:
        out.append(item * 2.5 + 1.0)
    return out
"""
    inspector = PythonCodeInspector()
    info = inspector.inspect(code)
    assert info["loop_depth"] == 1
    assert "*" in info["binary_ops"]
    assert "+" in info["binary_ops"]


def test_computation_graph_matmul_not_vector_add():
    code = """
for i in range(N):
    for j in range(N):
        for k in range(N):
            c[i][j] += a[i][k] * b[k][j]
"""
    graph = generate_computation_graph(code, "python")
    assert "before" in graph and "after" in graph
    
    # Verify Before nodes contain Matrix Multiplication, not vector addition
    before_labels = [n["label"] for n in graph["before"]["nodes"]]
    assert any("Matrix" in lbl or "MatMul" in lbl or "GEMM" in lbl for lbl in before_labels)
    assert not any("Vector Addition" in lbl for lbl in before_labels)

    # Verify After nodes contain Tiled / SIMD GEMM
    after_labels = [n["label"] for n in graph["after"]["nodes"]]
    assert any("Tiled" in lbl or "GEMM" in lbl or "Matrix" in lbl for lbl in after_labels)


def test_computation_graph_reduction_not_vector_add():
    code = """
total = 0.0
for x in data:
    total += x
"""
    graph = generate_computation_graph(code, "python")
    before_labels = [n["label"] for n in graph["before"]["nodes"]]
    assert any("Reduction" in lbl or "Accumulator" in lbl for lbl in before_labels)


def test_mlir_generation_matmul_not_vector_add():
    code = """
for i in range(N):
    for j in range(N):
        for k in range(N):
            c[i][j] += a[i][k] * b[k][j]
"""
    mlir_data = generate_mlir_for_program(code, "python")
    assert "original_mlir" in mlir_data
    assert "stages" in mlir_data
    assert len(mlir_data["stages"]) == 5
    
    # Must contain linalg.matmul
    assert "linalg.matmul" in mlir_data["original_mlir"]


def test_api_analyze_code_dynamic():
    # 1. Test Matrix Multiplication
    matmul_code = """
def matmul(N=64):
    for i in range(N):
        for j in range(N):
            for k in range(N):
                c[i][j] += a[i][k] * b[k][j]
"""
    res = client.post("/api/analyze-code", json={"language": "python", "code": matmul_code, "program_id": "matmul"})
    assert res.status_code == 200
    data = res.json()
    assert "Matrix" in data["workload"] or "GEMM" in data["workload"]
    assert data["loop_count"] == 3
    assert len(data["detected_operations"]) > 0

    # 2. Test Custom Code
    custom_code = """
def my_func(a, b):
    return [a[i] + b[i] * 3 for i in range(len(a))]
"""
    res = client.post("/api/analyze-code", json={"language": "python", "code": custom_code, "program_id": "custom"})
    assert res.status_code == 200
    data = res.json()
    assert data["loop_count"] >= 1
    assert "graph" in data
    assert "mlir" in data


def test_api_compare_performance_dynamic():
    orig_code = """
a = [1.0] * 5000
b = [2.0] * 5000
c = []
for i in range(len(a)):
    c.append(a[i] + b[i])
"""
    opt_code = """
import numpy as np
a = np.ones(5000, dtype=np.float32)
b = np.full(5000, 2.0, dtype=np.float32)
c = a + b
"""
    res = client.post("/api/compare-performance", json={
        "language": "python",
        "original_code": orig_code,
        "optimized_code": opt_code
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert "original_time_ms" in data
    assert "optimized_time_ms" in data
    assert "speedup_factor" in data
    assert "improvement_percent" in data
