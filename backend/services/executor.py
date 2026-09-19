"""
Sandboxed and Controlled Code Execution Engine for User-Submitted Code
"""

import os
import sys
import time
import tempfile
import subprocess
from typing import Dict, Any

from backend.services.mojo_simulator import simulate_mojo_execution


def execute_user_code(language: str, code: str, timeout_seconds: float = 5.0) -> Dict[str, Any]:
    """Safely executes user-submitted Python or Mojo code with timeout and error handling."""
    language = language.lower().strip()
    if language not in ("python", "mojo"):
        return {
            "status": "ERROR",
            "language": language,
            "output": "",
            "error_type": "UnsupportedLanguage",
            "error_line": None,
            "error_message": f"Language '{language}' is not supported. Supported: python, mojo.",
            "execution_time_seconds": 0.0
        }

    # Security check: Block destructive system/filesystem operations
    forbidden_tokens = [
        "os.system", "shutil.rmtree", "subprocess.Popen", "subprocess.call",
        "__import__('os')", "eval(", "exec(", "open('/etc", "C:\\Windows\\System32"
    ]
    for token in forbidden_tokens:
        if token in code:
            return {
                "status": "ERROR",
                "language": language,
                "output": "",
                "error_type": "SecurityRestriction",
                "error_line": None,
                "error_message": f"Execution rejected: Code contains restricted token '{token}' for system safety.",
                "execution_time_seconds": 0.0
            }

    # --------------------------------------------------------------------------
    # Python Code Execution
    # --------------------------------------------------------------------------
    if language == "python":
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as temp_f:
            temp_path = temp_f.name
            temp_f.write(code)

        start_time = time.perf_counter()
        try:
            res = subprocess.run(
                [sys.executable, temp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_seconds
            )
            elapsed = time.perf_counter() - start_time

            if res.returncode == 0:
                return {
                    "status": "SUCCESS",
                    "language": "python",
                    "output": res.stdout,
                    "error_type": None,
                    "error_line": None,
                    "error_message": None,
                    "execution_time_seconds": round(elapsed, 4)
                }
            else:
                stderr = res.stderr
                # Parse error type and line
                error_type = "RuntimeError"
                error_line = None

                # Search last non-empty line of traceback for exception type
                lines = [l.strip() for l in stderr.strip().splitlines() if l.strip()]
                if lines:
                    last_line = lines[-1]
                    if ":" in last_line:
                        possible_err = last_line.split(":")[0].strip()
                        if " " not in possible_err and (possible_err.endswith("Error") or possible_err.endswith("Exception")):
                            error_type = possible_err
                    elif last_line.endswith("Error") or last_line.endswith("Exception"):
                        error_type = last_line

                if "SyntaxError" in stderr:
                    error_type = "SyntaxError"

                # Extract line number from traceback
                for line in stderr.splitlines():
                    if "File " in line and "line " in line:
                        parts = line.split("line ")
                        if len(parts) > 1:
                            num_str = parts[1].split(",")[0].split()[0]
                            if num_str.isdigit():
                                error_line = int(num_str)

                return {
                    "status": "ERROR",
                    "language": "python",
                    "output": res.stdout,
                    "error_type": error_type,
                    "error_line": error_line,
                    "error_message": stderr.strip(),
                    "execution_time_seconds": round(elapsed, 4)
                }

        except subprocess.TimeoutExpired:
            return {
                "status": "TIMEOUT",
                "language": "python",
                "output": "",
                "error_type": "TimeoutError",
                "error_line": None,
                "error_message": f"Execution exceeded maximum limit of {timeout_seconds}s.",
                "execution_time_seconds": timeout_seconds
            }
        except Exception as e:
            return {
                "status": "ERROR",
                "language": "python",
                "output": "",
                "error_type": type(e).__name__,
                "error_line": None,
                "error_message": str(e),
                "execution_time_seconds": 0.0
            }
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    # --------------------------------------------------------------------------
    # Mojo Code Execution (Native or Windows Simulator)
    # --------------------------------------------------------------------------
    elif language == "mojo":
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mojo", delete=False, encoding="utf-8") as temp_f:
            temp_path = temp_f.name
            temp_f.write(code)

        start_time = time.perf_counter()
        try:
            # Check if mojo CLI is available on PATH
            res = subprocess.run(
                ["mojo", "run", temp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_seconds
            )
            elapsed = time.perf_counter() - start_time

            if res.returncode == 0:
                return {
                    "status": "SUCCESS",
                    "language": "mojo",
                    "output": res.stdout,
                    "error_type": None,
                    "error_line": None,
                    "error_message": None,
                    "execution_time_seconds": round(elapsed, 4)
                }
            else:
                return {
                    "status": "ERROR",
                    "language": "mojo",
                    "output": res.stdout,
                    "error_type": "MojoCompilationOrRuntimeError",
                    "error_line": None,
                    "error_message": res.stderr.strip(),
                    "execution_time_seconds": round(elapsed, 4)
                }

        except FileNotFoundError:
            # Mojo CLI not installed on host PATH (Windows native)
            # Run intelligent Mojo static compiler analyzer & simulator
            sim_res = simulate_mojo_execution(code)
            return sim_res
            
        except subprocess.TimeoutExpired:
            return {
                "status": "TIMEOUT",
                "language": "mojo",
                "output": "",
                "error_type": "TimeoutError",
                "error_line": None,
                "error_message": f"Execution exceeded maximum limit of {timeout_seconds}s.",
                "execution_time_seconds": timeout_seconds
            }
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
