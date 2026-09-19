"""
FastAPI router for MAX Graph and Operator Fusion benchmarks
"""

from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter
from backend.services.analyzer import generate_max_graph_for_program
from max.graph_pipeline import build_demo_max_graph, MAXCompilerOptimizer
from max.max_benchmarks import run_max_comparison

router = APIRouter(prefix="/api", tags=["MAX Engine"])


class MAXGraphRequest(BaseModel):
    code: Optional[str] = None
    language: Optional[str] = "python"
    program_id: Optional[str] = None


@router.get("/max-graph")
async def get_max_graph_data():
    """Returns symbolic computation graphs before and after operator fusion alongside live benchmarks."""
    raw_graph = build_demo_max_graph()
    opt_graph = MAXCompilerOptimizer.optimize_graph(raw_graph)
    benchmark_res = run_max_comparison()

    return {
        "raw_graph": raw_graph.get_summary(),
        "optimized_graph": opt_graph.get_summary(),
        "benchmark": benchmark_res
    }


@router.post("/max-graph")
async def post_max_graph_data(req: MAXGraphRequest):
    """Dynamically generates MAX computation graphs and memory traffic comparison for the active program."""
    if req.code and req.code.strip():
        return generate_max_graph_for_program(req.code, req.language or "python")
    return await get_max_graph_data()

