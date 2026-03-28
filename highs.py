"""
MILP Solver (HiGHS)
=====================

Builds the full MILP formulation: binary active + continuous quantity + big-M
linking, then solves with HiGHS. HiGHS presolve eliminates the binary variables
through generic reductions, reaching LP relaxation -- but this takes >6 minutes
at the demo scale (1000 pools x 4000 workloads).

Open-source solver, no license required.
"""

from __future__ import annotations

import time

import highspy
import numpy as np

from instance import AllocationInstance


def solve_milp_highs(inst: AllocationInstance, time_limit: float = 600.0) -> dict:
    """MILP formulation solved with HiGHS."""
    t0 = time.perf_counter()

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", time_limit)

    n = inst.n_assignments
    big_m = float(inst.capacities.max())
    inf = highspy.kHighsInf

    # Variables: active[0..n-1] binary, quantity[n..2n-1] continuous
    col_lower = np.zeros(2 * n)
    col_upper = np.concatenate([np.ones(n), np.full(n, big_m)])
    h.addVars(2 * n, col_lower, col_upper)

    col_costs = np.concatenate([np.zeros(n), inst.costs.astype(np.float64)])
    h.changeColsCost(2 * n, np.arange(2 * n, dtype=np.int32), col_costs)
    h.changeObjectiveSense(highspy.ObjSense.kMinimize)

    # Mark active[0..n-1] as integer (binary)
    h.changeColsIntegrality(
        n, np.arange(n, dtype=np.int32),
        np.full(n, highspy.HighsVarType.kInteger),
    )

    # Big-M linking: quantity[r] - big_m * active[r] <= 0
    bigm_starts = np.arange(n, dtype=np.int32) * 2
    bigm_indices = np.empty(2 * n, dtype=np.int32)
    bigm_values = np.empty(2 * n)
    for r in range(n):
        bigm_indices[2 * r] = r
        bigm_values[2 * r] = -big_m
        bigm_indices[2 * r + 1] = n + r
        bigm_values[2 * r + 1] = 1.0
    h.addRows(n, np.full(n, -inf), np.zeros(n),
              2 * n, bigm_starts, bigm_indices, bigm_values)

    # Supply + demand constraints
    _add_supply_demand(h, inst, qty_offset=n)

    t_build = time.perf_counter() - t0
    t_solve_start = time.perf_counter()
    h.run()
    solve_time = time.perf_counter() - t_solve_start
    total_time = time.perf_counter() - t0

    status = h.getModelStatus()
    if status == highspy.HighsModelStatus.kOptimal:
        obj = h.getInfoValue("objective_function_value")[1]
        return {
            "solver": "highs",
            "time_s": total_time,
            "build_s": t_build,
            "solve_s": solve_time,
            "objective": int(round(obj)),
            "status": "optimal",
        }
    elif status == highspy.HighsModelStatus.kTimeLimit:
        return {
            "solver": "highs",
            "time_s": total_time,
            "objective": float("inf"),
            "status": "timeout",
        }
    else:
        return {
            "solver": "highs",
            "time_s": total_time,
            "objective": float("inf"),
            "status": str(status),
        }


def _add_supply_demand(h, inst: AllocationInstance, qty_offset: int):
    """Add supply (pool capacity) and demand (workload) constraints."""
    n = inst.n_assignments
    inf = highspy.kHighsInf

    assigns_by_pool = [[] for _ in range(inst.n_pools)]
    assigns_by_wl = [[] for _ in range(inst.n_workloads)]
    for r in range(n):
        assigns_by_pool[inst.assign_pool[r]].append(r)
        assigns_by_wl[inst.assign_workload[r]].append(r)

    row_lower, row_upper = [], []
    csr_indices, csr_values, csr_starts = [], [], []

    for p in range(inst.n_pools):
        assigns = assigns_by_pool[p]
        if not assigns:
            continue
        csr_starts.append(len(csr_indices))
        for r in assigns:
            csr_indices.append(qty_offset + r)
            csr_values.append(1.0)
        row_lower.append(-inf)
        row_upper.append(float(inst.capacities[p]))

    for w in range(inst.n_workloads):
        assigns = assigns_by_wl[w]
        if not assigns:
            continue
        csr_starts.append(len(csr_indices))
        for r in assigns:
            csr_indices.append(qty_offset + r)
            csr_values.append(1.0)
        d = float(inst.demands[w])
        row_lower.append(d)
        row_upper.append(d)

    n_rows = len(row_lower)
    h.addRows(
        n_rows, np.array(row_lower), np.array(row_upper),
        len(csr_indices),
        np.array(csr_starts, dtype=np.int32),
        np.array(csr_indices, dtype=np.int32),
        np.array(csr_values),
    )


if __name__ == "__main__":
    from instance import create_instance
    inst = create_instance()
    print(f"Instance: {inst.n_pools} pools, {inst.n_workloads} workloads, "
          f"{inst.n_assignments:,} assignments")
    print("Running HiGHS MILP (this may take >6 minutes)...")
    result = solve_milp_highs(inst)
    print(f"HiGHS: {result['time_s']:.1f}s, ${result.get('objective', '?'):,}, {result['status']}")
