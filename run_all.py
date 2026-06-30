"""
Run All Solvers
================

Generates the GPU allocation instance and runs all four solvers,
printing a comparison table matching the results in the paper.

Usage:
    uv run run_all.py               # full run (greedy + MCF + Gurobi + HiGHS)
    uv run run_all.py --skip-highs  # skip HiGHS (saves >6 minutes)
    uv run run_all.py --skip-gurobi # skip Gurobi (requires license)
"""

from __future__ import annotations

import argparse
import sys
import time

from instance import create_instance


def format_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}min"


def main():
    parser = argparse.ArgumentParser(description="GPU allocation benchmark")
    parser.add_argument("--skip-highs", action="store_true",
                        help="Skip HiGHS MILP (saves >6 minutes)")
    parser.add_argument("--skip-gurobi", action="store_true",
                        help="Skip Gurobi MILP (requires license)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("GPU Allocation Benchmark")
    print("DeQL Studio: Declarative Decision-Making over Relational Data")
    print("VLDB 2026 Demo Track\n")

    print("Generating instance...", flush=True)
    inst = create_instance(seed=args.seed)
    print(f"  {inst.n_pools:,} pools x {inst.n_workloads:,} workloads")
    print(f"  {inst.n_assignments:,} candidate assignments")
    print(f"  Total supply/demand: {int(inst.capacities.sum()):,}\n")

    results = []

    # Greedy
    print("Running greedy...", end=" ", flush=True)
    from greedy import solve_greedy
    r = solve_greedy(inst)
    print(f"{format_time(r['time_s'])}, ${r['objective']:,}, "
          f"{r['status']} ({r['n_violations']}/{r['n_pools']} violations)")
    results.append(r)

    # MCF
    print("Running MCF (OR-Tools cost-scaling)...", end=" ", flush=True)
    from mcf import solve_mcf
    r = solve_mcf(inst)
    print(f"{format_time(r['time_s'])}, ${r['objective']:,}, {r['status']}")
    results.append(r)
    mcf_time = r["time_s"]

    # Gurobi
    if args.skip_gurobi:
        print("Skipping Gurobi (--skip-gurobi)")
    else:
        try:
            import gurobipy
        except ImportError:
            print("Skipping Gurobi (gurobipy not installed)")
        else:
            print("Running Gurobi MILP...", end=" ", flush=True)
            from gurobi import solve_milp_gurobi
            r = solve_milp_gurobi(inst)
            print(f"{format_time(r['time_s'])}, ${r['objective']:,}, {r['status']}")
            if r["incumbents"]:
                first = r["incumbents"][0]
                print(f"  first heuristic: {format_time(first['time_s'])}, "
                      f"${first['objective']:,}")
            results.append(r)

    # HiGHS
    if args.skip_highs:
        print("Skipping HiGHS (--skip-highs)")
    else:
        print("Running HiGHS MILP (this may take >6 minutes)...", end=" ", flush=True)
        from highs import solve_milp_highs
        r = solve_milp_highs(inst)
        print(f"{format_time(r['time_s'])}, ${r.get('objective', '?'):,}, {r['status']}")
        results.append(r)

    # Summary table
    print(f"\n{'Solver':<12} {'Time':>8} {'Cost':>12} {'Status':<12} {'vs MCF':>8}")
    print("-" * 56)
    for r in results:
        speedup = ""
        if r["solver"] != "mcf" and r["status"] == "optimal" and mcf_time > 0:
            speedup = f"{r['time_s'] / mcf_time:.0f}x"
        obj_str = f"${r['objective']:,}" if r["objective"] < float("inf") else "---"
        print(f"{r['solver']:<12} {format_time(r['time_s']):>8} {obj_str:>12} "
              f"{r['status']:<12} {speedup:>8}")


if __name__ == "__main__":
    main()
