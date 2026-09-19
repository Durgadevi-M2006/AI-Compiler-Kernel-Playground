"""
FastAPI integration test suite for all REST endpoints
"""

import sys
import os
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.main import app

client = TestClient(app)


def test_health_check():
    """Verifies GET /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_get_environment():
    """Verifies GET /api/environment returns hardware info."""
    response = client.get("/api/environment")
    assert response.status_code == 200
    data = response.json()
    assert "cpu_processor" in data
    assert "logical_cores" in data


def test_get_experiments():
    """Verifies GET /api/experiments returns all 5 workloads."""
    response = client.get("/api/experiments")
    assert response.status_code == 200
    data = response.json()
    assert len(data["experiments"]) == 5


def test_post_run_experiment():
    """Verifies POST /api/run-experiment executes and returns benchmarks."""
    response = client.post(
        "/api/run-experiment",
        json={"experiment": "vector_add", "params": {"size": 20000}}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "metrics" in data["data"]
    assert "optimization_analysis" in data["data"]


def test_post_run_code():
    """Verifies POST /api/run-code executes custom Python code."""
    response = client.post(
        "/api/run-code",
        json={"language": "python", "code": "print('FASTAPI_RUNNER_OK')"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert "FASTAPI_RUNNER_OK" in data["output"]
    assert "optimization_analysis" in data


def test_history_and_export():
    """Verifies session history retrieval and CSV/JSON export."""
    # 1. Get history
    res = client.get("/api/history")
    assert res.status_code == 200
    assert "history" in res.json()

    # 2. Export CSV
    res_csv = client.get("/api/export/csv")
    assert res_csv.status_code == 200
    assert "text/csv" in res_csv.headers["content-type"]

    # 3. Export JSON
    res_json = client.get("/api/export/json")
    assert res_json.status_code == 200
    assert "application/json" in res_json.headers["content-type"]

    # 4. Clear history
    res_del = client.delete("/api/history")
    assert res_del.status_code == 200
    assert res_del.json()["status"] == "success"


def test_mlir_and_max_endpoints():
    """Verifies MLIR and MAX data routes."""
    res_mlir = client.get("/api/mlir-pipeline")
    assert res_mlir.status_code == 200
    assert "pipelines" in res_mlir.json()

    res_max = client.get("/api/max-graph")
    assert res_max.status_code == 200
    assert "benchmark" in res_max.json()


def test_post_analyze_code_vector_ops():
    """Verifies POST /api/analyze-code classifies workload and returns Why/How/Impact."""
    code = "c = [a[i] + b[i] for i in range(N)]"
    res = client.post("/api/analyze-code", json={"language": "python", "code": code})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["workload"] == "Vector Operations"
    assert len(data["opportunities"]) > 0
    opp = data["opportunities"][0]
    assert "why" in opp
    assert "how" in opp
    assert "impact" in opp


def test_post_optimize_code_vector_add():
    """Verifies POST /api/optimize-code generates transformed code and changes list."""
    code = "c = [a[i] + b[i] for i in range(1000)]"
    res = client.post("/api/optimize-code", json={"language": "python", "code": code})
    assert res.status_code == 200
    data = res.json()
    assert data["can_optimize"] is True
    assert "np.add" in data["optimized_code"] or "numpy" in data["optimized_code"]
    assert len(data["changes_applied"]) > 0


def test_post_compare_performance():
    """Verifies POST /api/compare-performance executes both and returns genuine speedup."""
    orig_code = "total = sum([i * 2 for i in range(10000)])\nprint(total)"
    opt_code = "import numpy as np\ntotal = np.sum(np.arange(10000, dtype=np.int64) * 2)\nprint(total)"
    res = client.post(
        "/api/compare-performance",
        json={"language": "python", "original_code": orig_code, "optimized_code": opt_code}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert "original_time_ms" in data
    assert "optimized_time_ms" in data
    assert "speedup_factor" in data
    assert "speedup_str" in data
    assert "improvement_percent" in data
    assert "benchmark_note" in data


def test_post_unsupported_optimization_fallback():
    """Verifies fallback behavior when no automatic transformation pattern is safely identified."""
    code = "print('Hello world non-numerical script')"
    res = client.post("/api/optimize-code", json={"language": "python", "code": code})
    assert res.status_code == 200
    data = res.json()
    assert data["can_optimize"] is False
    assert "No automatic optimization was safely identified" in data["message"]
