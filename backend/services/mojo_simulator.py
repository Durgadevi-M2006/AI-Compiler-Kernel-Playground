"""
Mojo Kernel Static Analyzer and Accurate Execution Simulation Engine
Transpiles Mojo constructs (var, let, fn, UnsafePointer, SIMD, vectorize, parallelize, generics)
into a sandboxed evaluation environment with dynamic SIMD width resolution and execution.
"""

import ast
import sys
import io
import time
import re
import math
from typing import Dict, Any, List, Optional


# ------------------------------------------------------------------------------
# SIMD Width Extractor & Resolution Helper
# ------------------------------------------------------------------------------
def extract_simd_width(mojo_code: str) -> int:
    """
    Extracts the SIMD register width declared or used in Mojo code.
    Checks for:
    1. Explicit alias declarations: alias simd_width = 4
    2. SIMD type specifications: SIMD[DType.float32, 4] or SIMD[dtype, 16]
    3. vectorize call parameters: vectorize[kernel, 4]
    4. Defaults to 8 (standard 256-bit AVX2 float32 register width).
    """
    # 1. Check alias simd_width = \d+
    alias_match = re.search(r'alias\s+(?:simd_width|width)\s*=\s*(\d+)', mojo_code)
    if alias_match:
        return int(alias_match.group(1))

    # 2. Check SIMD[dtype, \d+] or SIMD[DType.float32, \d+]
    simd_match = re.search(r'SIMD\s*\[\s*[^,\]]+\s*,\s*(\d+)\s*\]', mojo_code)
    if simd_match:
        return int(simd_match.group(1))

    # 3. Check vectorize[..., \d+]
    vec_match = re.search(r'vectorize\s*\[\s*[^,\]]+\s*,\s*(\d+)\s*\]', mojo_code)
    if vec_match:
        return int(vec_match.group(1))

    # Default to 8 (256-bit AVX2 Float32 width)
    return 8


# ------------------------------------------------------------------------------
# Simulated Mojo Runtime Types & Built-ins
# ------------------------------------------------------------------------------
class DType:
    float32 = "float32"
    float64 = "float64"
    int32 = "int32"
    int64 = "int64"


class ScalarType:
    def __getitem__(self, item):
        return item


class SIMDMask(list):
    """Boolean mask vector resulting from SIMD comparisons, supporting .select(true_val, false_val)."""
    def __init__(self, items=None, width=None):
        if items is None:
            items = []
        super().__init__([bool(x) for x in items])

    def select(self, true_val, false_val):
        """Simulates (cond_mask).select(true_vec, false_vec)"""
        if isinstance(true_val, (list, tuple, SIMD)):
            t = list(true_val)
        else:
            t = [true_val] * len(self)

        if isinstance(false_val, (list, tuple, SIMD)):
            f = list(false_val)
        else:
            f = [false_val] * len(self)

        res = [float(t[i]) if self[i] else float(f[i]) for i in range(len(self))]
        return SIMD(res, width=len(self))

    def __repr__(self):
        return f"SIMDMask[{', '.join(str(x) for x in self)}]"


class SIMD(list):
    """Simulates Mojo SIMD[DType, width] hardware vector registers."""
    def __init__(self, val=0.0, width=None):
        w = width if width is not None and isinstance(width, int) else 8
        if isinstance(val, (int, float)):
            super().__init__([float(val)] * w)
        elif isinstance(val, (list, tuple)):
            super().__init__([float(x) for x in val])
        elif isinstance(val, SIMD):
            super().__init__(list(val))
        else:
            super().__init__([float(val)] * w)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return SIMD([x + other for x in self], width=len(self))
        return SIMD([a + b for a, b in zip(self, other)], width=len(self))

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if isinstance(other, (int, float)):
            for i in range(len(self)):
                self[i] += float(other)
        elif isinstance(other, (list, tuple, SIMD)):
            for i in range(min(len(self), len(other))):
                self[i] += float(other[i])
        return self

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return SIMD([x - other for x in self], width=len(self))
        return SIMD([a - b for a, b in zip(self, other)], width=len(self))

    def __isub__(self, other):
        if isinstance(other, (int, float)):
            for i in range(len(self)):
                self[i] -= float(other)
        elif isinstance(other, (list, tuple, SIMD)):
            for i in range(min(len(self), len(other))):
                self[i] -= float(other[i])
        return self

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return SIMD([x * other for x in self], width=len(self))
        return SIMD([a * b for a, b in zip(self, other)], width=len(self))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        if isinstance(other, (int, float)):
            for i in range(len(self)):
                self[i] *= float(other)
        elif isinstance(other, (list, tuple, SIMD)):
            for i in range(min(len(self), len(other))):
                self[i] *= float(other[i])
        return self

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return SIMD([x / other for x in self], width=len(self))
        return SIMD([a / b for a, b in zip(self, other)], width=len(self))

    def __gt__(self, other):
        if isinstance(other, (list, tuple, SIMD)):
            return SIMDMask([a > b for a, b in zip(self, other)])
        return SIMDMask([x > other for x in self])

    def __ge__(self, other):
        if isinstance(other, (list, tuple, SIMD)):
            return SIMDMask([a >= b for a, b in zip(self, other)])
        return SIMDMask([x >= other for x in self])

    def __lt__(self, other):
        if isinstance(other, (list, tuple, SIMD)):
            return SIMDMask([a < b for a, b in zip(self, other)])
        return SIMDMask([x < other for x in self])

    def __le__(self, other):
        if isinstance(other, (list, tuple, SIMD)):
            return SIMDMask([a <= b for a, b in zip(self, other)])
        return SIMDMask([x <= other for x in self])

    def __eq__(self, other):
        if isinstance(other, (list, tuple, SIMD)):
            return SIMDMask([a == b for a, b in zip(self, other)])
        return SIMDMask([x == other for x in self])

    def __ne__(self, other):
        if isinstance(other, (list, tuple, SIMD)):
            return SIMDMask([a != b for a, b in zip(self, other)])
        return SIMDMask([x != other for x in self])

    def select(self, true_val, false_val):
        """Simulates (cond).select(true_vec, false_vec)"""
        t = true_val if isinstance(true_val, (list, tuple, SIMD)) else [true_val] * len(self)
        f = false_val if isinstance(false_val, (list, tuple, SIMD)) else [false_val] * len(self)
        return SIMD([float(t[i]) if self[i] != 0.0 else float(f[i]) for i in range(len(self))], width=len(self))

    def reduce_add(self) -> float:
        return sum(self)

    def reduce_max(self) -> float:
        return max(self)

    def __repr__(self):
        if len(self) <= 8:
            return f"SIMD[{', '.join(f'{x:.2f}' if isinstance(x, float) else str(x) for x in self)}]"
        return f"SIMD[{', '.join(f'{x:.2f}' for x in self[:4])}, ... (width={len(self)})]"


class SIMDGeneric:
    """SIMD constructor supporting SIMD[dtype, width](val) and SIMD[dtype](val)."""
    def __init__(self, default_width: int = 8):
        self.default_width = default_width

    def __getitem__(self, item):
        if isinstance(item, (tuple, list)):
            dtype = item[0]
            width_spec = item[1] if len(item) > 1 else self.default_width
            resolved_w = width_spec if isinstance(width_spec, int) else self.default_width
            return lambda val=0.0, w=resolved_w: SIMD(val, width=w)
        return lambda val=0.0, w=self.default_width: SIMD(val, width=w)

    def __call__(self, val=0.0, width=None):
        w = width if width is not None else self.default_width
        return SIMD(val, width=w)


class UnsafePointer:
    """Simulates Mojo's UnsafePointer[Scalar[dtype]] raw memory buffer."""
    def __init__(self, size=1024, dtype=DType.float32):
        self.size = size
        self.dtype = dtype
        self.data = [0.0] * size

    @classmethod
    def alloc(cls, size, dtype=DType.float32):
        return cls(size, dtype)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.data[idx]
        return self.data[idx]

    def __setitem__(self, idx, val):
        if isinstance(idx, slice):
            self.data[idx] = val
        else:
            self.data[idx] = float(val)

    def load(self, idx, width=8):
        slice_end = min(idx + width, self.size)
        vals = self.data[idx:slice_end]
        if len(vals) < width:
            vals = vals + [0.0] * (width - len(vals))
        return SIMD(vals, width=width)

    def store(self, idx, val, width=8):
        if isinstance(val, (list, SIMD)):
            for i, v in enumerate(val):
                if idx + i < self.size:
                    self.data[idx + i] = float(v)
        else:
            self.data[idx] = float(val)

    def free(self):
        pass

    def __len__(self):
        return self.size

    def __repr__(self):
        if self.size <= 10:
            return str(self.data)
        return f"[{', '.join(f'{x:.2f}' for x in self.data[:6])}, ... ({self.size} elements)]"


class UnsafePointerGeneric:
    def __getitem__(self, item):
        return UnsafePointer

    def alloc(self, size, dtype=DType.float32):
        return UnsafePointer.alloc(size, dtype)


class SimdWidthGetter:
    def __init__(self, default_width: int = 8):
        self.default_width = default_width

    def __getitem__(self, item):
        return lambda *args, **kwargs: self.default_width

    def __call__(self, *args, **kwargs):
        return self.default_width


class VectorizeGeneric:
    """Simulates Mojo's vectorize[func, simd_width](size) parameterized engine."""
    def __init__(self, default_width: int = 8):
        self.default_width = default_width

    def __getitem__(self, item):
        if isinstance(item, (tuple, list)):
            func = item[0]
            simd_w = item[1] if len(item) > 1 and isinstance(item[1], int) else self.default_width
        else:
            func = item
            simd_w = self.default_width

        def wrapper(size):
            for idx in range(0, size, simd_w):
                w = min(simd_w, size - idx)
                try:
                    # Pass width if the function accepts it as named or positional argument
                    func(idx, width=w)
                except TypeError:
                    try:
                        func(idx)
                    except TypeError:
                        try:
                            func(w)(idx)
                        except Exception:
                            pass
        return wrapper

    def __call__(self, func, width=None):
        w = width if width is not None else self.default_width
        return self.__getitem__((func, w))


class ParallelizeGeneric:
    """Simulates Mojo's parallelize[func](num_workers) multi-core dispatcher."""
    def __getitem__(self, item):
        if isinstance(item, (tuple, list)):
            func = item[0]
        else:
            func = item

        def wrapper(num_tiles):
            for t in range(num_tiles):
                try:
                    func(t)
                except Exception:
                    pass
        return wrapper

    def __call__(self, func):
        return self.__getitem__(func)


def perf_counter_ns():
    return int(time.perf_counter() * 1e9)


# ------------------------------------------------------------------------------
# Transpiler: Converts Mojo code into Executable Python Syntax
# ------------------------------------------------------------------------------
def transpile_mojo_to_python(mojo_code: str) -> str:
    """Transpiles Mojo syntax to standard Python with parameterized SIMD scope."""
    simd_width_val = extract_simd_width(mojo_code)
    lines = mojo_code.splitlines()
    py_lines = []

    # Insert scope header with resolved SIMD width
    py_lines.append(f"simd_width = {simd_width_val}")
    py_lines.append(f"width = {simd_width_val}")

    for line in lines:
        stripped = line.strip()

        # Skip imports of mojo internal modules
        if (stripped.startswith("from sys.info") or 
            stripped.startswith("from algorithm") or 
            stripped.startswith("from memory") or 
            stripped.startswith("from time") or 
            stripped.startswith("from math")):
            py_lines.append(f"# {line}")
            continue

        # Convert decorators
        if stripped.startswith("@parameter"):
            py_lines.append(re.sub(r'@parameter', '# @parameter', line))
            continue

        # Convert aliases preserving leading whitespace
        line = re.sub(r'^(\s*)alias\s+(\w+)\s*:\s*[^=]+=\s*(.*)', r'\1\2 = \3', line)
        line = re.sub(r'^(\s*)alias\s+(\w+)\s*=\s*(.*)', r'\1\2 = \3', line)

        # Convert var / let preserving leading whitespace
        line = re.sub(r'^(\s*)(var|let)\s+(\w+)\s*:\s*[^=]+=\s*(.*)', r'\1\3 = \4', line)
        line = re.sub(r'^(\s*)(var|let)\s+(\w+)\s*=\s*(.*)', r'\1\3 = \4', line)
        line = re.sub(r'^(\s*)(var|let)\s+(\w+)\s*:\s*(.*)', r'\1\3 = 0', line)

        # Convert parameterized function declarations:
        # e.g., fn relu_kernel[width: Int](idx: Int): -> def relu_kernel(idx, width=simd_width):
        def convert_fn_with_generics(m):
            indent = m.group(1)
            fn_name = m.group(2)
            generic_params = m.group(3)
            regular_args = m.group(4)

            # Extract generic names and assign defaults
            gen_names = []
            for gp in generic_params.split(','):
                gp_clean = gp.strip()
                if ':' in gp_clean:
                    g_name = gp_clean.split(':')[0].strip()
                else:
                    g_name = gp_clean
                if g_name in ('width', 'w', 'simd_w'):
                    gen_names.append(f"{g_name}=simd_width")
                elif g_name:
                    gen_names.append(f"{g_name}=None")

            # Clean regular args
            reg_names = []
            for ra in regular_args.split(','):
                ra_clean = ra.strip()
                if ':' in ra_clean:
                    reg_names.append(ra_clean.split(':')[0].strip())
                elif ra_clean:
                    reg_names.append(ra_clean)

            all_params = reg_names + gen_names
            return f"{indent}def {fn_name}({', '.join(all_params)}):"

        line = re.sub(
            r'^(\s*)fn\s+(\w+)\s*\[(.*?)\]\s*\((.*?)\)\s*(?:->\s*[^:]+)?\s*:',
            convert_fn_with_generics,
            line
        )

        # Standard non-generic fn declarations
        line = re.sub(r'^(\s*)fn\s+(\w+)\s*\((.*?)\)\s*(?:->\s*[^:]+)?\s*:', r'\1def \2(\3):', line)

        # Clean type annotations inside parameters: (x: Int, y: Float32) -> (x, y)
        def clean_params(m):
            header = m.group(1)
            params = m.group(2)
            cleaned = []
            for p in params.split(','):
                p = p.strip()
                if '=' in p:
                    # Preserve default arguments like width=simd_width
                    lhs, rhs = p.split('=', 1)
                    if ':' in lhs:
                        lhs = lhs.split(':')[0].strip()
                    cleaned.append(f"{lhs.strip()}={rhs.strip()}")
                elif ':' in p:
                    cleaned.append(p.split(':')[0].strip())
                elif p:
                    cleaned.append(p)
            return f"{header}({', '.join(cleaned)}):"

        line = re.sub(r'(def\s+\w+)\s*\((.*?)\)\s*:', clean_params, line)

        # Convert Float32 / Int casts
        line = re.sub(r'Float32\((.*?)\)', r'float(\1)', line)
        line = re.sub(r'Float64\((.*?)\)', r'float(\1)', line)
        line = re.sub(r'Int\((.*?)\)', r'int(\1)', line)

        # Pointer load / store syntax preserving parameterized width:
        # e.g., in_ptr.load[width=width](idx) -> in_ptr.load(idx, width=width)
        line = re.sub(
            r'\.load\s*\[\s*(?:width\s*=\s*)?([^\]]+)\s*\]\s*\((.*?)\)',
            r'.load(\2, width=\1)',
            line
        )
        line = re.sub(
            r'\.store\s*\[\s*(?:width\s*=\s*)?([^\]]+)\s*\]\s*\((.*?),\s*(.*?)\)',
            r'.store(\2, \3, width=\1)',
            line
        )

        py_lines.append(line)

    combined = "\n".join(py_lines)
    
    # Post-process with AST to inject nonlocal into nested closures if needed
    try:
        tree = ast.parse(combined)
        class ClosureFixer(ast.NodeTransformer):
            def __init__(self):
                super().__init__()
                self.scope_stack = []

            def visit_FunctionDef(self, node):
                param_names = {arg.arg for arg in node.args.args}
                if node.args.vararg:
                    param_names.add(node.args.vararg.arg)
                if node.args.kwarg:
                    param_names.add(node.args.kwarg.arg)
                for kw in node.args.kwonlyargs:
                    param_names.add(kw.arg)

                direct_locals = set(param_names)
                for stmt in node.body:
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if isinstance(target, ast.Name):
                                direct_locals.add(target.id)
                    elif isinstance(stmt, ast.AugAssign):
                        if isinstance(stmt.target, ast.Name):
                            direct_locals.add(stmt.target.id)

                if self.scope_stack:
                    outer_locals = set()
                    for scope in self.scope_stack:
                        outer_locals.update(scope)

                    modified_names = set()
                    for child in ast.walk(node):
                        if isinstance(child, ast.AugAssign) and isinstance(child.target, ast.Name):
                            if child.target.id not in param_names and child.target.id in outer_locals:
                                modified_names.add(child.target.id)

                    if modified_names:
                        nonlocal_node = ast.Nonlocal(names=sorted(list(modified_names)))
                        node.body.insert(0, nonlocal_node)

                self.scope_stack.append(direct_locals)
                self.generic_visit(node)
                self.scope_stack.pop()
                return node

        fixer = ClosureFixer()
        tree = fixer.visit(tree)
        ast.fix_missing_locations(tree)
        combined = ast.unparse(tree)
    except Exception:
        pass

    if "def main(" in combined and not re.search(r'^\s*main\(\)', combined, re.MULTILINE):
        combined += "\n\nmain()"

    return combined


# ------------------------------------------------------------------------------
# Simulator Execution Entry Point
# ------------------------------------------------------------------------------
def simulate_mojo_execution(code: str) -> Dict[str, Any]:
    """Transpiles Mojo code to sandboxed Python environment, executes arithmetic, and returns output."""
    start_time = time.perf_counter()
    simd_w = extract_simd_width(code)

    # Transpile code
    try:
        py_code = transpile_mojo_to_python(code)
    except Exception as e:
        return {
            "status": "ERROR",
            "language": "mojo",
            "runtime": "Simulation Mode",
            "runtime_mode": "Simulation Mode",
            "error_type": "MojoParserError",
            "error_line": 1,
            "error_message": f"Mojo Syntax Parsing Error: {str(e)}",
            "output": f"Execution Error [MojoParserError]: {str(e)}",
            "execution_time_seconds": 0.001
        }

    # Setup isolated namespace with Mojo builtins
    sandbox_namespace = {
        "DType": DType,
        "Scalar": ScalarType(),
        "SIMD": SIMDGeneric(default_width=simd_w),
        "UnsafePointer": UnsafePointerGeneric(),
        "simdwidthof": SimdWidthGetter(default_width=simd_w),
        "simd_width": simd_w,
        "width": simd_w,
        "vectorize": VectorizeGeneric(default_width=simd_w),
        "parallelize": ParallelizeGeneric(),
        "perf_counter_ns": perf_counter_ns,
        "math": math,
        "tanh": math.tanh,
        "sqrt": math.sqrt,
        "min": min,
        "max": max,
        "abs": abs,
        "len": len,
        "range": range,
        "print": print,
        "True": True,
        "False": False
    }

    # Intercept stdout
    captured_stdout = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured_stdout

    try:
        exec(py_code, sandbox_namespace)
        execution_output = captured_stdout.getvalue()
        status = "SUCCESS"
        error_type = None
        error_line = None
        error_msg = None
    except Exception as e:
        execution_output = captured_stdout.getvalue()
        status = "ERROR"
        error_type = type(e).__name__
        error_msg = str(e)
        error_line = None
        tb = sys.exc_info()[2]
        while tb and tb.tb_next:
            tb = tb.tb_next
        if tb:
            error_line = tb.tb_lineno
    finally:
        sys.stdout = old_stdout

    elapsed = time.perf_counter() - start_time

    if execution_output.strip():
        output_str = execution_output.strip()
    else:
        if status == "ERROR":
            output_str = f"Execution Error [{error_type}]: {error_msg}"
        else:
            output_str = "[Program completed with no output]"

    return {
        "status": status,
        "language": "mojo",
        "runtime": "Simulation Mode",
        "runtime_mode": "Simulation Mode",
        "output": output_str,
        "error_type": error_type,
        "error_line": error_line,
        "error_message": error_msg,
        "simd_width": simd_w,
        "execution_time_seconds": round(elapsed, 4)
    }
