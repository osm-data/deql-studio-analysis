"""
GPU Allocation Instance Generator
===================================

Generates a GPU-to-workload allocation problem matching the DeQL Studio demo:
1,000 GPU pools x 4,000 workloads, ~2.7M candidate assignments after
hardware-compatibility filtering (WHERE mem_gb >= min_mem).

All values are integers (required by OR-Tools SimpleMinCostFlow).
Supply is balanced: sum(capacities) == sum(demands).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MEM_TIERS = [24, 40, 80]


@dataclass
class AllocationInstance:
    n_pools: int
    n_workloads: int
    capacities: np.ndarray      # int64, shape (n_pools,)
    demands: np.ndarray         # int64, shape (n_workloads,)
    assign_pool: np.ndarray     # int64, index into pools
    assign_workload: np.ndarray # int64, index into workloads
    costs: np.ndarray           # int64, per-assignment unit cost

    @property
    def n_assignments(self) -> int:
        return len(self.costs)


def create_instance(
    n_pools: int = 1000,
    n_workloads: int = 4000,
    seed: int = 42,
    density: float = 0.7,
) -> AllocationInstance:
    """Generate a balanced GPU allocation instance with mem_gb >= min_mem filter.

    Args:
        n_pools: Number of GPU pools.
        n_workloads: Number of workloads.
        seed: Random seed for reproducibility.
        density: Probability of each (pool, workload) pair existing before filtering.
    """
    rng = np.random.default_rng(seed)

    # Pool properties -- bias toward high mem_gb so most edges survive filtering
    raw_caps = rng.integers(100, 1000, size=n_pools)
    pool_mem = rng.choice(MEM_TIERS, size=n_pools, p=[0.05, 0.25, 0.70])
    pool_cost = np.round(rng.uniform(1.50, 6.50, size=n_pools), 2)

    # Workload properties -- bias toward low min_mem for feasibility
    raw_demands = rng.integers(20, 200, size=n_workloads)
    workload_min_mem = rng.choice(MEM_TIERS, size=n_workloads, p=[0.60, 0.30, 0.10])

    # Balance: scale demands so total demand == total supply
    total_supply = int(raw_caps.sum())
    scaled = raw_demands / raw_demands.sum() * total_supply
    demands = np.maximum(scaled.astype(np.int64), 1)
    demands[-1] += total_supply - int(demands.sum())
    capacities = raw_caps.astype(np.int64)
    assert int(capacities.sum()) == int(demands.sum()), "Supply-demand imbalance"

    # Generate candidate assignments with mem_gb >= min_mem filter
    all_pool = []
    all_workload = []
    all_cost = []
    for p in range(n_pools):
        for w in range(n_workloads):
            if rng.random() < density:
                if pool_mem[p] >= workload_min_mem[w]:
                    all_pool.append(p)
                    all_workload.append(w)
                    all_cost.append(int(rng.integers(1, 101)))

    assign_pool = np.array(all_pool, dtype=np.int64)
    assign_workload = np.array(all_workload, dtype=np.int64)
    costs = np.array(all_cost, dtype=np.int64)

    # Feasibility: every workload must be reachable
    reachable_workloads = set(assign_workload.tolist())
    max_pool_mem = int(max(pool_mem))
    for w in range(n_workloads):
        if w not in reachable_workloads:
            compatible = [p for p in range(n_pools) if pool_mem[p] >= workload_min_mem[w]]
            if not compatible:
                workload_min_mem[w] = max_pool_mem
                compatible = [p for p in range(n_pools) if pool_mem[p] >= workload_min_mem[w]]
            p = max(compatible, key=lambda p: capacities[p])
            assign_pool = np.append(assign_pool, p)
            assign_workload = np.append(assign_workload, w)
            costs = np.append(costs, int(rng.integers(1, 101)))

    # Every pool must also be reachable
    reachable_pools = set(assign_pool.tolist())
    for p in range(n_pools):
        if p not in reachable_pools:
            compatible = [w for w in range(n_workloads) if pool_mem[p] >= workload_min_mem[w]]
            if not compatible:
                compatible = list(range(n_workloads))
            w = max(compatible, key=lambda w: demands[w])
            assign_pool = np.append(assign_pool, p)
            assign_workload = np.append(assign_workload, w)
            costs = np.append(costs, int(rng.integers(1, 101)))

    return AllocationInstance(
        n_pools=n_pools,
        n_workloads=n_workloads,
        capacities=capacities,
        demands=demands,
        assign_pool=assign_pool,
        assign_workload=assign_workload,
        costs=costs,
    )
