"""
FastAPI router for user-submitted custom code execution,
interactive optimization analysis, safe code optimization generation,
dynamic computation graph generation, MLIR generation, and live performance comparison.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from backend.services.executor import execute_user_code
from backend.services.analyzer import (
    classify_workload,
    detect_optimization_opportunities,
    generate_optimized_code,
    generate_computation_graph,
    generate_mlir_for_program,
    analyze_code_optimizations
)
from backend.services.history_store import session_history

router = APIRouter(tags=["Custom Code & Optimization"])


# ------------------------------------------------------------------------------
# Request / Response Schemas
# ------------------------------------------------------------------------------
class CodeExecutionRequest(BaseModel):
    language: str = "python"
    code: str
    timeout_seconds: Optional[float] = 5.0


class CodeAnalysisRequest(BaseModel):
    language: str = "python"
    code: str
    program_id: Optional[str] = None


class CodeOptimizationRequest(BaseModel):
    language: str = "python"
    code: str


class PerformanceComparisonRequest(BaseModel):
    language: str = "python"
    original_code: str
    optimized_code: Optional[str] = None
    program_id: Optional[str] = None
    timeout_seconds: Optional[float] = 5.0


class GraphGenerationRequest(BaseModel):
    language: str = "python"
    code: str


class MLIRGenerationRequest(BaseModel):
    language: str = "python"
    code: str


# ------------------------------------------------------------------------------
# 1. Analyze Code Endpoint (Single Source of Truth)
# ------------------------------------------------------------------------------
@router.post("/api/analyze-code")
@router.post("/analyze-code")
@router.post("/api/analyze")
async def analyze_code_endpoint(req: CodeAnalysisRequest):
    """
    Analyzes submitted Python or Mojo code, extracts AST operations, builds dynamic Before/After
    computation graphs, generates MLIR intermediate representations, and identifies optimization opportunities.
    """
    if not req.code.strip():
        raise HTTPException(status_code=400, detail="Code cannot be empty for analysis.")

    analysis = analyze_code_optimizations(code=req.code, language=req.language, program_id=req.program_id)
    return analysis


# ------------------------------------------------------------------------------
# 2. Dynamic Graph Generation Endpoint
# ------------------------------------------------------------------------------
@router.post("/api/graph")
async def generate_graph_endpoint(req: GraphGenerationRequest):
    """Dynamically generates Before and After computation graphs based on AST operations."""
    if not req.code.strip():
        raise HTTPException(status_code=400, detail="Code cannot be empty for graph generation.")

    graph_data = generate_computation_graph(code=req.code, language=req.language)
    return graph_data


# ------------------------------------------------------------------------------
# 3. Dynamic MLIR Generation Endpoint
# ------------------------------------------------------------------------------
@router.post("/api/mlir")
async def generate_mlir_endpoint(req: MLIRGenerationRequest):
    """Dynamically generates Original/Optimized MLIR and a 5-stage lowering pipeline for active code."""
    if not req.code.strip():
        raise HTTPException(status_code=400, detail="Code cannot be empty for MLIR generation.")

    mlir_data = generate_mlir_for_program(code=req.code, language=req.language)
    return mlir_data


# ------------------------------------------------------------------------------
# 4. Optimize Code Endpoint
# ------------------------------------------------------------------------------
@router.post("/api/optimize-code")
@router.post("/optimize-code")
@router.post("/api/optimize")
async def optimize_code_endpoint(req: CodeOptimizationRequest):
    """
    Generates a safely transformed optimized version of the code for supported patterns,
    along with a list of changes applied.
    """
    if not req.code.strip():
        raise HTTPException(status_code=400, detail="Code cannot be empty for optimization.")

    opt_result = generate_optimized_code(code=req.code, language=req.language)
    return {
        "status": "success",
        "language": req.language,
        "original_code": req.code,
        "workload": opt_result["workload"],
        "can_optimize": opt_result["can_optimize"],
        "optimized_code": opt_result["optimized_code"],
        "changes_applied": opt_result["changes_applied"],
        "message": opt_result.get("message")
    }


# ------------------------------------------------------------------------------
# 5. Run Code Endpoint
# ------------------------------------------------------------------------------
@router.post("/api/run-code")
@router.post("/run-code")
async def run_code_endpoint(req: CodeExecutionRequest):
    """Safely executes user-submitted code in sandboxed subprocess and provides optimization advice."""
    if not req.code.strip():
        raise HTTPException(status_code=400, detail="Code cannot be empty.")

    exec_res = execute_user_code(
        language=req.language,
        code=req.code,
        timeout_seconds=req.timeout_seconds or 5.0
    )

    analysis = analyze_code_optimizations(code=req.code, language=req.language)
    exec_res["optimization_analysis"] = analysis

    session_history.add_entry(
        experiment_name=f"Custom {req.language.upper()} ({analysis['workload']})",
        language=req.language.capitalize(),
        parameters={"code_length_lines": len(req.code.splitlines())},
        execution_time_ms=round(exec_res["execution_time_seconds"] * 1000.0, 3),
        speedup="User Code",
        correctness=(exec_res["status"] == "SUCCESS"),
        notes=f"Status: {exec_res['status']}" + (f" ({exec_res['error_type']})" if exec_res.get('error_type') else "")
    )

    return exec_res


# ------------------------------------------------------------------------------
# 6. Compare Performance & Verified Benchmark Endpoint
# ------------------------------------------------------------------------------
@router.post("/api/compare-performance")
@router.post("/compare-performance")
@router.post("/api/benchmark")
@router.post("/benchmark")
async def compare_performance_endpoint(req: PerformanceComparisonRequest):
    """
    Executes BOTH original and transformed code in the sandbox, checks output correctness,
    measures execution times over warm-up and timed runs, and logs comparison into session history.
    """
    if not req.original_code.strip():
        raise HTTPException(status_code=400, detail="Original code cannot be empty.")

    optimized_code = req.optimized_code
    changes_applied = []
    workload = classify_workload(req.original_code, req.language)

    if not optimized_code or not optimized_code.strip():
        gen = generate_optimized_code(req.original_code, req.language)
        if gen["can_optimize"] and gen["optimized_code"]:
            optimized_code = gen["optimized_code"]
            changes_applied = gen["changes_applied"]
        else:
            optimized_code = req.original_code
            changes_applied = ["No automatic code transformation available; executed original code."]

    # 1. Warm-up and Timed Runs for Original Code
    _ = execute_user_code(language=req.language, code=req.original_code, timeout_seconds=req.timeout_seconds or 5.0)
    orig_times = []
    orig_res = None
    for _ in range(3):
        res = execute_user_code(language=req.language, code=req.original_code, timeout_seconds=req.timeout_seconds or 5.0)
        orig_res = res
        orig_times.append(res["execution_time_seconds"] * 1000.0)

    # 2. Warm-up and Timed Runs for Transformed Code
    _ = execute_user_code(language=req.language, code=req.optimized_code or req.original_code, timeout_seconds=req.timeout_seconds or 5.0)
    opt_times = []
    opt_res = None
    for _ in range(3):
        res = execute_user_code(language=req.language, code=req.optimized_code or req.original_code, timeout_seconds=req.timeout_seconds or 5.0)
        opt_res = res
        opt_times.append(res["execution_time_seconds"] * 1000.0)

    orig_ms = round(sorted(orig_times)[len(orig_times)//2], 3)
    opt_ms = round(sorted(opt_times)[len(opt_times)//2], 3)

    # Correctness Check
    is_success = (orig_res["status"] == "SUCCESS" and opt_res["status"] == "SUCCESS")
    output_match = is_success
    correctness_message = "✓ Correctness check passed"
    
    if is_success:
        orig_out = orig_res["output"].strip()
        opt_out = opt_res["output"].strip()
        if "error" in orig_out.lower() or "error" in opt_out.lower():
            output_match = False
            correctness_message = "❌ Execution contained error messages"
        else:
            # Check numerical outputs for discrepancy
            import re
            import math
            orig_nums = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", orig_out)]
            opt_nums = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", opt_out)]
            if orig_nums and opt_nums:
                # If last output numbers distinctly differ
                if not math.isclose(orig_nums[-1], opt_nums[-1], rel_tol=1e-2, abs_tol=1e-2):
                    output_match = False
                    correctness_message = f"❌ Output result mismatch: baseline produced {orig_nums[-1]}, transformed produced {opt_nums[-1]}"

    correctness_verified = is_success and output_match

    # Difference and Percentage Change Formula:
    # difference = transformed_time - baseline_time
    # percentage_change = ((transformed_time - baseline_time) / baseline_time) * 100
    time_diff_ms = round(opt_ms - orig_ms, 3)
    if orig_ms > 0:
        perf_change_pct = round(((opt_ms - orig_ms) / orig_ms) * 100.0, 1)
        improvement_pct = round(((orig_ms - opt_ms) / orig_ms) * 100.0, 1)
        speedup_factor = round(orig_ms / max(0.0001, opt_ms), 2)
    else:
        perf_change_pct = 0.0
        improvement_pct = 0.0
        speedup_factor = 1.0

    # Classify Performance with 3% tolerance
    if not correctness_verified:
        perf_verdict = "CORRECTNESS_FAILED"
        perf_message = "❌ Correctness check failed. Benchmark comparison skipped."
        is_faster = False
    elif perf_change_pct <= -3.0:
        perf_verdict = "FASTER"
        perf_message = "✓ Transformed version is faster"
        is_faster = True
    elif perf_change_pct >= 3.0:
        perf_verdict = "SLOWER"
        perf_message = "⚠️ Transformed version is slower"
        is_faster = False
    else:
        perf_verdict = "EQUAL"
        perf_message = "≈ No significant performance difference"
        is_faster = False

    speedup_str = f"{speedup_factor}x"
    has_real_transformation = (
        changes_applied 
        and len(changes_applied) > 0 
        and "No automatic code transformation" not in changes_applied[0]
    )
    optimization_status = "Optimization Applied" if has_real_transformation else "Opportunity Detected"
    optimization_title = "✓ Optimization Applied" if has_real_transformation else "Optimization Opportunity Detected"

    benchmark_type = "Subprocess Execution" if req.language == "python" else "Simulation Benchmark"
    runtime_mode = "Native Python Subprocess" if req.language == "python" else "Simulation Mode"

    session_history.add_optimization_entry(
        experiment=f"Optimization Comparison ({workload})",
        language=req.language.capitalize(),
        original_code=req.original_code,
        detected_opportunities=[opp["title"] for opp in detect_optimization_opportunities(req.original_code, req.language)],
        optimizations_applied=changes_applied,
        original_execution_time_ms=orig_ms,
        optimized_execution_time_ms=opt_ms,
        speedup=speedup_str,
        notes=f"Baseline: {orig_ms} ms | Transformed: {opt_ms} ms | Performance Change: {perf_change_pct:+.1f}% | Verdict: {perf_message}"
    )

    return {
        "status": "SUCCESS" if correctness_verified else "ERROR_OR_MISMATCH",
        "workload": workload,
        "language": req.language,
        "program_id": req.program_id,
        "original_result": orig_res,
        "optimized_result": opt_res,
        "original_code": req.original_code,
        "optimized_code": optimized_code,
        "original_time_ms": orig_ms,
        "optimized_time_ms": opt_ms,
        "baseline_time_ms": orig_ms,
        "transformed_time_ms": opt_ms,
        "time_difference_ms": time_diff_ms,
        "difference_ms": time_diff_ms,
        "speedup_factor": speedup_factor,
        "speedup_str": speedup_str,
        "percentage_change": perf_change_pct,
        "performance_change_percent": perf_change_pct,
        "improvement_percent": improvement_pct,
        "is_faster": is_faster,
        "performance_verdict": perf_verdict,
        "performance_message": perf_message,
        "optimization_status": optimization_status,
        "optimization_title": optimization_title,
        "has_real_transformation": has_real_transformation,
        "correctness_verified": correctness_verified,
        "correctness_message": correctness_message,
        "output_match": output_match,
        "changes_applied": changes_applied,
        "benchmark_type": benchmark_type,
        "runtime_mode": runtime_mode,
        "benchmark_note": f"Measured across warm-up and timed iterations ({runtime_mode})."
    }
