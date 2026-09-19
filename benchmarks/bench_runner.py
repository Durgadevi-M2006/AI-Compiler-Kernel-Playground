"""
AI Compiler & Kernel Playground - Standalone CLI Benchmark Suite
"""

import sys
import os
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.runner import run_experiment
from backend.environment import detect_environment


def main():
    console = Console(safe_box=True)
    console.print(Panel.fit(
        "[bold cyan]AI Compiler & Kernel Playground[/bold cyan]\n"
        "[dim]High-Performance AI Kernel & MLIR Compiler Benchmarking Suite[/dim]",
        border_style="cyan"
    ))

    # Environment
    env = detect_environment()
    env_table = Table(title="Hardware & Toolchain Discovery", header_style="bold green")
    env_table.add_column("Property", style="cyan")
    env_table.add_column("Value", style="white")
    env_table.add_row("Operating System", env["os"])
    env_table.add_row("Processor Architecture", f"{env['machine_arch']} ({env['cpu_processor']})")
    env_table.add_row("CPU Logical Cores", str(env["logical_cores"]))
    env_table.add_row("Python Version", env["python_version"])
    env_table.add_row("NumPy Version", env["numpy_version"])
    env_table.add_row("PyTorch Version", env["torch_version"])
    env_table.add_row("Mojo CLI Status", f"{env['mojo_version']} (WSL/Modular Ready)")
    env_table.add_row("MAX Engine Status", env["max_version"])
    console.print(env_table)
    console.print("\n")

    experiments = [
        ("vector_add", {"size": 1_000_000}),
        ("matrix_multiply", {"n": 256}),
        ("elementwise_relu", {"size": 1_000_000, "op": "relu"}),
        ("reduction_sum", {"size": 2_000_000, "op": "sum"}),
        ("ml_workload", {"batch_size": 128, "in_dim": 256, "hidden_dim": 512, "out_dim": 10})
    ]

    all_results = {}

    for exp_id, params in experiments:
        console.print(f"[bold yellow]>>> Running Benchmark: {exp_id.upper()}...[/bold yellow]")
        res = run_experiment(exp_id, params)
        all_results[exp_id] = res

        # Render summary table
        t = Table(title=f"Experiment: {res['experiment']} ({json.dumps(params)})", header_style="bold magenta")
        for col in res["summary_table"][0].keys():
            t.add_column(col)
        for row in res["summary_table"]:
            t.add_row(*[str(val) for val in row.values()])
        console.print(t)
        console.print(f"[green][OK] Correctness Verified:[/green] {res['metrics']['correctness']}")
        console.print("\n" + "-" * 70 + "\n")

    # Save results to json
    results_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    console.print(f"[bold green][SUCCESS] All benchmarks completed! Results saved to {results_path}[/bold green]")


if __name__ == "__main__":
    main()
