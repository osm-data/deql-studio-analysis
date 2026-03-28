"""
Greedy Baseline: Demand-Priority
==================================

Sorts workloads by demand descending (largest first), then assigns each to the
cheapest compatible pool with enough remaining capacity. Force-assigns to
cheapest pool on overflow.

Fast but infeasible -- violates ~10% of capacity constraints.
"""

from __future__ import annotations

import time

import numpy as np

from instance import AllocationInstance


def solve_greedy(inst: AllocationInstance) -> dict:
    """Sort workloads by demand descending, assign to cheapest pool with capacity."""
    t0 = time.perf_counter()

    # Build workload -> list of (cost, pool_idx)
    options: list[list[tuple[int, int]]] = [[] for _ in range(inst.n_workloads)]
    for i in range(inst.n_assignments):
        w = int(inst.assign_workload[i])
        p = int(inst.assign_pool[i])
        c = int(inst.costs[i])
        options[w].append((c, p))

    for w in range(inst.n_workloads):
        options[w].sort()

    # Sort workloads by demand descending (large items first)
    order = np.argsort(-inst.demands)

    remaining_capacity = inst.capacities.copy()
    total_cost = 0
    assignment = np.full(inst.n_workloads, -1, dtype=np.int64)

    for w in order:
        w = int(w)
        demand = int(inst.demands[w])
        assigned = False

        for cost, pool in options[w]:
            if remaining_capacity[pool] >= demand:
                assignment[w] = pool
                remaining_capacity[pool] -= demand
                total_cost += cost * demand
                assigned = True
                break

        if not assigned and options[w]:
            cost, pool = options[w][0]
            assignment[w] = pool
            remaining_capacity[pool] -= demand
            total_cost += cost * demand

    solve_time = time.perf_counter() - t0

    # Count violations
    pool_usage = np.zeros(inst.n_pools, dtype=np.int64)
    for w in range(inst.n_workloads):
        if assignment[w] >= 0:
            pool_usage[assignment[w]] += inst.demands[w]

    n_violations = int(np.sum(pool_usage > inst.capacities))

    return {
        "solver": "greedy",
        "time_s": solve_time,
        "objective": int(total_cost),
        "status": "infeasible" if n_violations > 0 else "optimal",
        "feasible": n_violations == 0,
        "n_violations": n_violations,
        "n_pools": inst.n_pools,
    }


if __name__ == "__main__":
    from instance import create_instance
    inst = create_instance()
    print(f"Instance: {inst.n_pools} pools, {inst.n_workloads} workloads, "
          f"{inst.n_assignments:,} assignments")
    result = solve_greedy(inst)
    print(f"Greedy: {result['time_s']:.2f}s, ${result['objective']:,}, "
          f"{result['status']} ({result['n_violations']}/{result['n_pools']} violations)")
