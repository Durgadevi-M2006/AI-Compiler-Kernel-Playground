import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.services.mojo_simulator import (
    simulate_mojo_execution,
    extract_simd_width,
    transpile_mojo_to_python,
    SIMD,
    UnsafePointer
)


def test_mojo_simd_width_extraction():
    code1 = "alias simd_width = 4\nfn main(): pass"
    assert extract_simd_width(code1) == 4

    code2 = "var zeros = SIMD[DType.float32, 16](0.0)"
    assert extract_simd_width(code2) == 16

    code3 = "alias dtype = DType.float32\nalias simd_width = simdwidthof[dtype]()"
    assert extract_simd_width(code3) == 8


def test_mojo_elementwise_relu_no_width_name_error():
    code = """
from sys.info import simdwidthof
from algorithm import vectorize
from memory import UnsafePointer

alias dtype = DType.float32
alias simd_width = simdwidthof[dtype]()

fn main():
    print("Mojo Vectorized ReLU Kernel")
    alias N = 100
    var x = UnsafePointer[Scalar[dtype]].alloc(N)
    var out = UnsafePointer[Scalar[dtype]].alloc(N)
    
    for i in range(N):
        x[i] = Float32(i % 100) - 50.0
        
    @parameter
    fn relu_simd[width: Int](idx: Int):
        var vx = x.load[width=width](idx)
        var zeros = SIMD[dtype, width](0.0)
        var res = (vx > zeros).select(vx, zeros)
        out.store[width=width](idx, res)
        
    vectorize[relu_simd, simd_width](N)
    print("ReLU Completed. out[0] =", out[0])
    print("out[60] =", out[60])
"""
    res = simulate_mojo_execution(code)
    assert res["status"] == "SUCCESS", f"Execution failed: {res.get('error_message')}"
    assert "ReLU Completed. out[0] = 0.0" in res["output"]
    assert "out[60] = 10.0" in res["output"]
    assert res["runtime"] == "Simulation Mode"


def test_mojo_reduction_simulation():
    code = """
from sys.info import simdwidthof
from algorithm import vectorize
from memory import UnsafePointer

alias dtype = DType.float32
alias simd_width = simdwidthof[dtype]()

fn main():
    alias N = 100
    var data = UnsafePointer[Scalar[dtype]].alloc(N)
    for i in range(N):
        data[i] = 1.5
        
    var accum = SIMD[dtype, simd_width](0.0)
    
    @parameter
    fn accum_simd[width: Int](idx: Int):
        var v = data.load[width=width](idx)
        if width == simd_width:
            accum += v
        else:
            for i in range(width):
                accum[0] += v[i]
                
    vectorize[accum_simd, simd_width](N)
    var total = accum.reduce_add()
    print("Reduction Total =", total)
"""
    res = simulate_mojo_execution(code)
    assert res["status"] == "SUCCESS", f"Execution failed: {res.get('error_message')}"
    assert "Reduction Total = 150.0" in res["output"]


def test_python_element_add_correctness():
    def element_add(a, b):
        result = []
        for i in range(len(a)):
            result.append(a[i] + b[i])
        return result

    a = [1, 2, 3, 4]
    b = [10, 20, 30, 40]
    res = element_add(a, b)
    assert res == [11, 22, 33, 44]


def test_python_relu_correctness():
    def relu(numbers):
        result = []
        for i in range(len(numbers)):
            if numbers[i] > 0:
                result.append(numbers[i])
            else:
                result.append(0)
        return result

    numbers = [-5, 3, -2, 8, -1, 6]
    assert relu(numbers) == [0, 3, 0, 8, 0, 6]
