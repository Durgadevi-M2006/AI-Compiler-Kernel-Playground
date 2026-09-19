"""
FastAPI router for predefined experiments and benchmarks
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

from backend.runner import run_experiment
from backend.services.history_store import session_history
from backend.services.analyzer import analyze_code_optimizations

router = APIRouter(prefix="/api", tags=["Experiments"])


class ExperimentRequest(BaseModel):
    experiment: str
    params: Optional[Dict[str, Any]] = None


@router.get("/experiments")
async def get_experiments_catalog():
    """Returns catalog of all 5 supported AI compiler experiments."""
    return {
        "experiments": [
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
    }


@router.post("/run-experiment")
@router.post("/run")
async def run_experiment_endpoint(req: ExperimentRequest):
    """Executes live benchmark for requested experiment and records history."""
    try:
        result = run_experiment(req.experiment, req.params or {})
        
        # Attach optimization analysis
        analysis = analyze_code_optimizations("", experiment_type=req.experiment)
        result["optimization_analysis"] = analysis

        # Record into session history
        summary = result.get("summary_table", [])
        last_row = summary[-1] if summary else {}
        first_row = summary[0] if summary else {}
        
        session_history.add_entry(
            experiment_name=result.get("experiment", req.experiment),
            language="Python vs Mojo",
            parameters=req.params or {},
            execution_time_ms=last_row.get("Time (ms)", 0.0),
            speedup=last_row.get("Speedup") or f"{last_row.get('GFLOPS', '')} GFLOPS",
            correctness=result.get("metrics", {}).get("correctness", True),
            notes=last_row.get("Optimization", last_row.get("Notes", "Hardware-tuned"))
        )

        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
