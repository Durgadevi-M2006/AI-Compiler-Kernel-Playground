"""
Hardware, Compiler, and Runtime Environment Discovery Module
"""

import os
import platform
import subprocess
import sys
import numpy as np


def detect_environment() -> dict:
    """Discovers host hardware capabilities, compiler toolchains, and runtime support."""
    # CPU & OS Details
    cpu_model = platform.processor() or "Unknown CPU"
    os_name = f"{platform.system()} {platform.release()}"
    num_logical_cores = os.cpu_count() or 1
    python_ver = sys.version.split()[0]

    # Check Mojo CLI
    mojo_available = False
    mojo_version = "Not Installed"
    try:
        res = subprocess.run(["mojo", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3)
        if res.returncode == 0:
            mojo_available = True
            mojo_version = res.stdout.strip()
    except Exception:
        pass

    # Check Modular / MAX CLI
    max_available = False
    max_version = "Not Installed"
    try:
        res = subprocess.run(["modular", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3)
        if res.returncode == 0:
            max_available = True
            max_version = res.stdout.strip()
    except Exception:
        pass

    # Detect PyTorch
    torch_available = False
    torch_ver = "Not Installed"
    try:
        import torch
        torch_available = True
        torch_ver = torch.__version__
    except ImportError:
        pass

    # SIMD Width & Vector Capabilities estimation
    # On x86_64, default AVX2 is 8 floats (256-bit), AVX-512 is 16 floats (512-bit)
    simd_width_floats = 8
    is_64bit = sys.maxsize > 2**32

    return {
        "os": os_name,
        "machine_arch": platform.machine(),
        "cpu_processor": cpu_model,
        "logical_cores": num_logical_cores,
        "is_64bit": is_64bit,
        "python_version": python_ver,
        "numpy_version": np.__version__,
        "torch_version": torch_ver,
        "torch_available": torch_available,
        "mojo_installed": mojo_available,
        "mojo_version": mojo_version,
        "max_installed": max_available,
        "max_version": max_version,
        "estimated_simd_width": simd_width_floats,
        "supported_experiments": [
            "vector_add",
            "matrix_multiply",
            "elementwise_relu",
            "reduction_sum",
            "ml_workload"
        ]
    }


if __name__ == "__main__":
    env = detect_environment()
    print("Detected Environment:")
    for k, v in env.items():
        print(f"  {k}: {v}")
