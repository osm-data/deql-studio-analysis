"""
Min-Cost Network Flow (MCF) Solver
====================================

Solves the GPU allocation as a min-cost flow problem on a bipartite
supply-demand graph. This is the formulation the DeQL engine selects
after detecting transportation structure in the query.

Polynomial time via network simplex (OR-Tools).
"""

from __future__ import annotations

import time

import numpy as np
from ortools.graph.python import min_cost_flow

from instance import AllocationInstance


def solve_mcf(inst: AllocationInstance) -> dict:
    """Solve via min-cost network flow on bipartite graph."""
    t0 = time.perf_counter()

    smcf = min_cost_flow.SimpleMinCostFlow()

    n_pools = inst.n_pools
    n_wl = inst.n_workloads

    # Nodes: 0..n_pools-1 are pools, n_pools..n_pools+n_wl-1 are workloads
    start_nodes = inst.assign_pool.astype(np.int64)
    end_nodes = (n_pools + inst.assign_workload).astype(np.int64)
    arc_caps = inst.capacities[inst.assign_pool].astype(np.int64)
    unit_costs = inst.costs.astype(np.int64)

    smcf.add_arcs_with_capacity_and_unit_cost(start_nodes, end_nodes, arc_caps, unit_costs)

    # Node supplies: positive = source (pool), negative = sink (workload)
    n_nodes = n_pools + n_wl
    supplies = np.zeros(n_nodes, dtype=np.int64)
    supplies[:n_pools] = inst.capacities
    supplies[n_pools:] = -inst.demands
    smcf.set_nodes_supplies(np.arange(n_nodes, dtype=np.int64), supplies)

    t_build = time.perf_counter() - t0
    t_solve_start = time.perf_counter()
    status = smcf.solve()
    solve_time = time.perf_counter() - t_solve_start
    total_time = time.perf_counter() - t0

    if status == smcf.OPTIMAL:
        obj = float(smcf.optimal_cost())
        return {
            "solver": "mcf",
            "time_s": total_time,
            "build_s": t_build,
            "solve_s": solve_time,
            "objective": int(obj),
            "status": "optimal",
        }
    else:
        return {
            "solver": "mcf",
            "time_s": total_time,
            "objective": float("inf"),
            "status": "failed",
        }


if __name__ == "__main__":
    from instance import create_instance
    inst = create_instance()
    print(f"Instance: {inst.n_pools} pools, {inst.n_workloads} workloads, "
          f"{inst.n_assignments:,} assignments")
    result = solve_mcf(inst)
    print(f"MCF: {result['time_s']:.2f}s, ${result['objective']:,}, {result['status']}")
