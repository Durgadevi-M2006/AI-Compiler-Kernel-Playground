"""
Unit tests for the sandboxed code execution engine
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.services.executor import execute_user_code
from backend.services.analyzer import analyze_code_optimizations


def test_execute_valid_python_code():
    """Verifies that valid Python code executes and returns output with timing."""
    code = "result = [i * 2 for i in range(5)]\nprint(result)"
    res = execute_user_code("python", code, timeout_seconds=3.0)

    assert res["status"] == "SUCCESS"
    assert "[0, 2, 4, 6, 8]" in res["output"]
    assert res["execution_time_seconds"] > 0.0
    assert res["error_type"] is None


def test_execute_python_syntax_error():
    """Verifies SyntaxError is identified with line number."""
    code = "def invalid_func(\n    print('missing closing paren')"
    res = execute_user_code("python", code, timeout_seconds=3.0)

    assert res["status"] == "ERROR"
    assert res["error_type"] == "SyntaxError"
    assert res["error_line"] is not None


def test_execute_python_runtime_error():
    """Verifies RuntimeError and line extraction."""
    code = "a = 10\nb = 0\nprint(a / b)"
    res = execute_user_code("python", code, timeout_seconds=3.0)

    assert res["status"] == "ERROR"
    assert "ZeroDivisionError" in res["error_message"]
    assert res["error_line"] == 3


def test_execute_timeout_handling():
    """Verifies that infinite loops are killed by the timeout."""
    code = "while True:\n    pass"
    res = execute_user_code("python", code, timeout_seconds=0.5)

    assert res["status"] == "TIMEOUT"
    assert res["error_type"] == "TimeoutError"


def test_execute_security_restriction():
    """Verifies that restricted tokens are blocked."""
    code = "import os\nos.system('dir')"
    res = execute_user_code("python", code, timeout_seconds=3.0)

    assert res["status"] == "ERROR"
    assert res["error_type"] == "SecurityRestriction"


def test_execute_mojo_simulation_engine():
    """Verifies that Mojo code is parsed, validated, and executed in simulation mode on Windows."""
    mojo_code = """
fn main():
    var a = [1.0, 2.0, 3.0]
    var b = [4.0, 5.0, 6.0]
    var result = [a[i] + b[i] for i in range(len(a))]
    print("A:", a)
    print("B:", b)
    print("Result:", result)
"""
    res = execute_user_code("mojo", mojo_code, timeout_seconds=3.0)
    assert res["status"] == "SUCCESS"
    assert "Result: [5.0, 7.0, 9.0]" in res["output"]


def test_execute_mojo_vector_pointer_template():
    """Verifies that Mojo pointers and SIMD vector widths calculate correctly."""
    mojo_code = """
alias dtype = DType.float32
alias simd_width = simdwidthof[dtype]()

fn main():
    var a = UnsafePointer[Scalar[dtype]].alloc(10)
    var b = UnsafePointer[Scalar[dtype]].alloc(10)
    var c = UnsafePointer[Scalar[dtype]].alloc(10)
    for i in range(10):
        a[i] = 1.5
        b[i] = 2.5
        c[i] = a[i] + b[i]
    print("Output c[0]:", c[0])
    a.free()
    b.free()
    c.free()
"""
    res = execute_user_code("mojo", mojo_code, timeout_seconds=3.0)
    assert res["status"] == "SUCCESS"
    assert "Output c[0]: 4.0" in res["output"]


def test_analyzer_gemm_detection():
    """Verifies that the optimizer identifies Matrix Computation optimization opportunities."""
    code = "for i in range(N):\n    for j in range(N):\n        for k in range(N):\n            c[i][j] += a[i][k] * b[k][j]"
    analysis = analyze_code_optimizations(code, "python")

    assert "Matrix" in analysis["workload_category"] or "Matrix Computation" in analysis["workload"]
    assert analysis["total_suggestions"] >= 3
    titles = [o["title"] for o in analysis["opportunities"]]
    assert any("Loop" in t or "Memory" in t for t in titles)
    assert any("SIMD" in t or "Vector" in t for t in titles)
