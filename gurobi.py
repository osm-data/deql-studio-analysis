"""
MILP Solver (Gurobi)
======================

Same MILP formulation as highs.py but using Gurobi. Gurobi's aggressive
pre-root heuristics produce early incumbents before the root LP is solved,
giving a realistic trajectory: heuristic at ~8s -> optimal at ~26s.

Requires a Gurobi license (academic free or trial).
"""

from __future__ import annotations

import time

import numpy as np

from instance import AllocationInstance


def solve_milp_gurobi(inst: AllocationInstance, time_limit: float = 600.0) -> dict:
    """MILP formulation solved with Gurobi, capturing incumbents."""
    import gurobipy as gp
    from gurobipy import GRB
    import scipy.sparse as sp

    t0 = time.perf_counter()

    n = inst.n_assignments
    big_m = float(inst.capacities.max())

    m = gp.Model("gpu_alloc")
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", time_limit)

    # Variables: active (binary) + quantity (continuous)
    active = m.addMVar(n, vtype=GRB.BINARY, name="active")
    quantity = m.addMVar(n, lb=0, ub=big_m, name="quantity")

    # Objective: minimize sum(cost * quantity)
    m.setObjective(inst.costs.astype(np.float64) @ quantity, GRB.MINIMIZE)

    # Big-M linking: quantity <= big_m * active
    m.addConstr(quantity <= big_m * active)

    # Supply constraints: sum(quantity) by pool <= capacity
    pool_matrix = sp.csr_matrix(
        (np.ones(n), (inst.assign_pool, np.arange(n))),
        shape=(inst.n_pools, n)
    )
    m.addMConstr(pool_matrix, quantity, GRB.LESS_EQUAL,
                 inst.capacities.astype(np.float64))

    # Demand constraints: sum(quantity) by workload == demand
    wl_matrix = sp.csr_matrix(
        (np.ones(n), (inst.assign_workload, np.arange(n))),
        shape=(inst.n_workloads, n)
    )
    m.addMConstr(wl_matrix, quantity, GRB.EQUAL,
                 inst.demands.astype(np.float64))

    t_build = time.perf_counter() - t0

    # Track incumbents
    incumbents: list[tuple[float, float]] = []

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            elapsed = time.perf_counter() - t0
            incumbents.append((elapsed, obj))

    t_solve_start = time.perf_counter()
    m.optimize(callback)
    solve_time = time.perf_counter() - t_solve_start
    total_time = time.perf_counter() - t0

    if m.Status == GRB.OPTIMAL:
        status = "optimal"
    elif m.Status == GRB.TIME_LIMIT:
        status = "timeout"
    else:
        status = f"status_{m.Status}"

    obj = int(round(m.ObjVal)) if m.SolCount > 0 else float("inf")

    return {
        "solver": "gurobi",
        "time_s": total_time,
        "build_s": t_build,
        "solve_s": solve_time,
        "objective": obj,
        "status": status,
        "incumbents": [
            {"time_s": round(t, 1), "objective": int(round(o))}
            for t, o in incumbents
        ],
    }


if __name__ == "__main__":
    from instance import create_instance
    inst = create_instance()
    print(f"Instance: {inst.n_pools} pools, {inst.n_workloads} workloads, "
          f"{inst.n_assignments:,} assignments")
    print("Running Gurobi MILP...")
    result = solve_milp_gurobi(inst)
    print(f"Gurobi: {result['time_s']:.1f}s, ${result['objective']:,}, {result['status']}")
    for inc in result["incumbents"]:
        print(f"  incumbent: t={inc['time_s']:.1f}s, obj=${inc['objective']:,}")
