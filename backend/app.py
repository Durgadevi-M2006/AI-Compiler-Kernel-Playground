"""
AI Compiler & Kernel Playground - Web Backend Server (Flask)
"""

import os
import sys
import json
from flask import Flask, jsonify, request, send_from_directory

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.environment import detect_environment
from backend.runner import run_experiment
from max.graph_pipeline import build_demo_max_graph, MAXCompilerOptimizer
from max.max_benchmarks import run_max_comparison

app = Flask(__name__, static_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend")))


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory(app.static_folder, path)


@app.route("/api/environment", methods=["GET"])
def get_environment():
    """Returns detected host hardware, CPU SIMD capabilities, and compiler toolchains."""
    env = detect_environment()
    return jsonify(env)


@app.route("/api/experiments", methods=["GET"])
def list_experiments():
    """Returns metadata for all available playground experiments."""
    experiments = [
        {
            "id": "vector_add",
            "name": "Experiment 1: Vector Addition",
            "category": "Vector & Memory Workloads",
            "description": "Compares Python scalar loops against NumPy SIMD and Mojo vectorized + multi-core parallel kernels.",
            "default_params": {"size": 1_000_000, "runs": 5},
            "param_controls": [
                {"name": "size", "label": "Vector Size (N)", "type": "select", "options": [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000], "default": 1_000_000}
            ]
        },
        {
            "id": "matrix_multiply",
            "name": "Experiment 2: Matrix Multiplication (GEMM)",
            "category": "Compute-Bound Workloads",
            "description": "Compares naive O(N^3) Python against loop reordering (i-k-j), OpenBLAS, Mojo 2D Cache Tiling, and Mojo Parallel SIMD.",
            "default_params": {"n": 256, "runs": 3},
            "param_controls": [
                {"name": "n", "label": "Matrix Dimension (NxN)", "type": "select", "options": [64, 128, 256, 512], "default": 256}
            ]
        },
        {
            "id": "elementwise_relu",
            "name": "Experiment 3: Element-wise Activations (ReLU & GELU)",
            "category": "Memory-Bandwidth Bound",
            "description": "Benchmarks pointwise neural network activation kernels (ReLU and GELU approximation).",
            "default_params": {"size": 1_000_000, "op": "relu"},
            "param_controls": [
                {"name": "op", "label": "Activation Function", "type": "select", "options": ["relu", "gelu"], "default": "relu"},
                {"name": "size", "label": "Tensor Size", "type": "select", "options": [200_000, 500_000, 1_000_000, 3_000_000], "default": 1_000_000}
            ]
        },
        {
            "id": "reduction_sum",
            "name": "Experiment 4: Reduction Operations (Sum / Max)",
            "category": "Loop-Carried Dependency",
            "description": "Demonstrates breaking serial reduction dependencies using SIMD vector accumulators and multi-core reduction trees.",
            "default_params": {"size": 2_000_000, "op": "sum"},
            "param_controls": [
                {"name": "op", "label": "Reduction Operation", "type": "select", "options": ["sum", "max"], "default": "sum"},
                {"name": "size", "label": "Array Size", "type": "select", "options": [500_000, 1_000_000, 2_000_000, 5_000_000], "default": 2_000_000}
            ]
        },
        {
            "id": "ml_workload",
            "name": "Experiment 5: AI / ML Forward Inference Layer",
            "category": "Neural Network Pipeline",
            "description": "End-to-end 2-layer MLP forward pass comparing naive Python, NumPy, PyTorch ATen, and Mojo/MAX Fused Kernel.",
            "default_params": {"batch_size": 128, "in_dim": 256, "hidden_dim": 512, "out_dim": 10},
            "param_controls": [
                {"name": "batch_size", "label": "Batch Size", "type": "select", "options": [32, 64, 128, 256], "default": 128},
                {"name": "hidden_dim", "label": "Hidden Dimension", "type": "select", "options": [128, 256, 512, 1024], "default": 512}
            ]
        }
    ]
    return jsonify({"experiments": experiments})


@app.route("/api/run", methods=["POST"])
def run_benchmark_endpoint():
    """Executes live benchmark for requested experiment with given parameters."""
    data = request.get_json() or {}
    exp_id = data.get("experiment", "vector_add")
    params = data.get("params", {})
    
    try:
        result = run_experiment(exp_id, params)
        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/mlir-pipeline", methods=["GET"])
def get_mlir_pipeline():
    """Returns MLIR dialect lowering pipeline metadata and dialect code snippets."""
    pipeline_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mlir", "pipeline.json"))
    if os.path.exists(pipeline_file):
        with open(pipeline_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({"error": "pipeline.json not found"}), 404


@app.route("/api/max-graph", methods=["GET"])
def get_max_graph():
    """Returns MAX Graph optimization demonstration and runtime benchmark."""
    raw_graph = build_demo_max_graph()
    opt_graph = MAXCompilerOptimizer.optimize_graph(raw_graph)
    benchmark_res = run_max_comparison()

    return jsonify({
        "raw_graph": raw_graph.get_summary(),
        "optimized_graph": opt_graph.get_summary(),
        "benchmark": benchmark_res
    })


def start_server(port: int = 5000, host: str = "127.0.0.1"):
    print(f"Starting AI Compiler & Kernel Playground server on http://{host}:{port}")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    start_server()
