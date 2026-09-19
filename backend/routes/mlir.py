"""
FastAPI router for MLIR lowering pipeline
"""

import os
import json
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api", tags=["MLIR Compiler"])


@router.get("/mlir-pipeline")
async def get_mlir_pipeline():
    """Returns MLIR dialect lowering pipeline stages and dialect code snippets."""
    pipeline_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "mlir", "pipeline.json"))
    if os.path.exists(pipeline_file):
        with open(pipeline_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    raise HTTPException(status_code=404, detail="pipeline.json not found")
