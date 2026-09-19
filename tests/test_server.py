"""
Integration tests for Flask Backend Endpoints and Web Playground Services
"""

import json
import pytest
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    """Verifies that the index page is served correctly."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"AI Compiler & Kernel Playground" in response.data


def test_environment_endpoint(client):
    """Verifies the /api/environment endpoint returns expected hardware fields."""
    response = client.get("/api/environment")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "cpu_processor" in data
    assert "logical_cores" in data
    assert "python_version" in data
    assert "numpy_version" in data


def test_experiments_list_endpoint(client):
    """Verifies the /api/experiments endpoint lists all 5 experiments."""
    response = client.get("/api/experiments")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "experiments" in data
    exp_ids = [e["id"] for e in data["experiments"]]
    assert "vector_add" in exp_ids
    assert "matrix_multiply" in exp_ids
    assert "elementwise_relu" in exp_ids
    assert "reduction_sum" in exp_ids
    assert "ml_workload" in exp_ids


def test_run_experiment_vector_add(client):
    """Verifies live benchmark execution via POST /api/run for vector_add."""
    response = client.post(
        "/api/run",
        data=json.dumps({"experiment": "vector_add", "params": {"size": 50000}}),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "success"
    assert "metrics" in data["data"]
    assert data["data"]["metrics"]["correctness"] is True


def test_run_experiment_matrix_multiply(client):
    """Verifies live benchmark execution via POST /api/run for matrix_multiply."""
    response = client.post(
        "/api/run",
        data=json.dumps({"experiment": "matrix_multiply", "params": {"n": 64}}),
        content_type="application/json"
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "success"
    assert "numpy_gflops" in data["data"]["metrics"]
    assert data["data"]["metrics"]["correctness"] is True


def test_mlir_pipeline_endpoint(client):
    """Verifies /api/mlir-pipeline returns the structured dialect passes."""
    response = client.get("/api/mlir-pipeline")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "pipelines" in data
    assert "vector_add" in data["pipelines"]
    assert "matmul" in data["pipelines"]
    assert "operator_fusion" in data["pipelines"]


def test_max_graph_endpoint(client):
    """Verifies /api/max-graph returns graph optimization passes and memory traffic evaluation."""
    response = client.get("/api/max-graph")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "raw_graph" in data
    assert "optimized_graph" in data
    assert "benchmark" in data
    assert data["benchmark"]["dram_bandwidth_savings_pct"] > 0
