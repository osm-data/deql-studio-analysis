"""
Network-structure detection: does a general-purpose solver exploit
transportation structure on its own?
=====================================================================

Reproduces the demo paper's point that a general-purpose MILP/LP solver does not
switch to a specialized network algorithm on this problem, even when asked to, so
the fair like-for-like baseline (its LP relaxation) is still far slower than
min-cost flow.

Two checks on the 3XL instance (1,000 pools x 4,000 workloads, ~2.7M arcs, seed 42):

  1. NetworkAlg discriminator (Gurobi). Solve the LP relaxation of the
     transportation problem three ways: primal simplex with Gurobi's network mode
     OFF (NetworkAlg=0), primal simplex with network mode ON (NetworkAlg=1), and
     dual simplex. If the two primal runs take the SAME number of simplex
     iterations, NetworkAlg=1 is a no-op on this model: Gurobi does not engage a
     distinct network algorithm just because the parameter is set.

  2. Fair LP-relaxation baseline. Time the LP relaxation (Gurobi dual simplex and
     HiGHS simplex) and min-cost flow (mcf.py) on the same instance. The LP
     relaxation is the best a general-purpose solver reaches here -- its presolve
     drops the binaries because the constraint matrix is totally unimodular -- and
     it is still far slower than MCF. This is the comparison behind the paper's
     30x figure.

Gurobi requires a license; if it is unavailable, the Gurobi checks are skipped and
the HiGHS LP + MCF baseline still run. Solve times vary by machine; the iteration
counts in check 1 do not.

Usage:
    uv run network_detection.py
    uv run network_detection.py --skip-gurobi
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import scipy.sparse as sp

from instance import AllocationInstance, create_instance


def transportation_lp(inst: AllocationInstance):
    """Node-arc incidence matrix of the transportation LP over reachable nodes.

    Supply rows (one per reachable pool): sum of arc flows <= capacity.
    Demand rows (one per reachable workload): sum of arc flows == demand.
    Returns (A, n_supply, n_demand, rhs_capacities, rhs_demands).
    """
    n = inst.n_assignments
    ap, aw = inst.assign_pool, inst.assign_workload
    pool_used = np.bincount(ap, minlength=inst.n_pools) > 0
    wl_used = np.bincount(aw, minlength=inst.n_workloads) > 0
    pool_row = -np.ones(inst.n_pools, np.int64)
    pool_row[pool_used] = np.arange(int(pool_used.sum()))
    n_supply = int(pool_used.sum())
    wl_row = -np.ones(inst.n_workloads, np.int64)
    wl_row[wl_used] = np.arange(int(wl_used.sum()))
    n_demand = int(wl_used.sum())
    rows = np.concatenate([pool_row[ap], n_supply + wl_row[aw]])
    cols = np.concatenate([np.arange(n), np.arange(n)])
    A = sp.csr_matrix((np.ones(2 * n), (rows, cols)), shape=(n_supply + n_demand, n))
    return (A, n_supply, n_demand,
            inst.capacities[pool_used].astype(float),
            inst.demands[wl_used].astype(float))


def gurobi_netalg_discriminator(inst: AllocationInstance, time_limit: float = 600.0) -> list[dict]:
    """Check 1: does Gurobi's NetworkAlg engage a distinct algorithm? Compare the
    simplex iteration counts of primal NETOFF vs primal NETON vs dual simplex."""
    import gurobipy as gp
    from gurobipy import GRB

    A, n_supply, n_demand, cap, dem = transportation_lp(inst)
    n = inst.n_assignments
    sense = np.array([GRB.LESS_EQUAL] * n_supply + [GRB.EQUAL] * n_demand)
    rhs = np.concatenate([cap, dem])
    obj = inst.costs.astype(float)
    ub = float(inst.capacities.max())

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()

    configs = [
        ("primal_network_off", {"Method": 0, "NetworkAlg": 0}),
        ("primal_network_on", {"Method": 0, "NetworkAlg": 1}),
        ("dual_simplex", {"Method": 1, "NetworkAlg": 0}),
    ]
    results = []
    for name, params in configs:
        m = gp.Model(env=env)
        m.Params.LogToConsole = 0
        m.Params.TimeLimit = time_limit
        x = m.addMVar(n, lb=0.0, ub=ub, obj=obj)
        m.ModelSense = GRB.MINIMIZE
        m.addMConstr(A, x, sense, rhs)
        m.update()
        for k, v in params.items():
            m.setParam(k, v)
        m.optimize()
        results.append({
            "config": name,
            "params": params,
            "iterations": int(m.IterCount),
            "runtime_s": round(m.Runtime, 2),
            "objective": int(round(m.ObjVal)) if m.SolCount else None,
        })
        m.dispose()
    env.dispose()
    return results


def gurobi_lp_relaxation(inst: AllocationInstance) -> dict:
    """Time Gurobi's LP relaxation (dual simplex, deterministic)."""
    import gurobipy as gp
    from gurobipy import GRB

    A, n_supply, n_demand, cap, dem = transportation_lp(inst)
    n = inst.n_assignments
    sense = np.array([GRB.LESS_EQUAL] * n_supply + [GRB.EQUAL] * n_demand)
    rhs = np.concatenate([cap, dem])
    obj = inst.costs.astype(float)
    ub = float(inst.capacities.max())

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model(env=env)
    m.Params.Method = 1  # dual simplex (deterministic timing)
    t0 = time.perf_counter()
    x = m.addMVar(n, lb=0.0, ub=ub, obj=obj)
    m.ModelSense = GRB.MINIMIZE
    m.addMConstr(A, x, sense, rhs)
    m.update()
    t_build = time.perf_counter() - t0
    t1 = time.perf_counter()
    m.optimize()
    t_solve = time.perf_counter() - t1
    out = {
        "solver": "gurobi-lp-relaxation (dual simplex)",
        "build_s": round(t_build, 3),
        "solve_s": round(t_solve, 3),
        "objective": int(round(m.ObjVal)) if m.SolCount else None,
    }
    m.dispose()
    env.dispose()
    return out


def highs_lp_relaxation(inst: AllocationInstance) -> dict:
    """Time HiGHS's LP relaxation (simplex); license-free."""
    import highspy

    from highs import _add_supply_demand

    n = inst.n_assignments
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("solver", "simplex")
    ub = float(inst.capacities.max())
    t0 = time.perf_counter()
    h.addVars(n, np.zeros(n), np.full(n, ub))
    h.changeColsCost(n, np.arange(n, dtype=np.int32), inst.costs.astype(np.float64))
    h.changeObjectiveSense(highspy.ObjSense.kMinimize)
    _add_supply_demand(h, inst, qty_offset=0)
    t_build = time.perf_counter() - t0
    t1 = time.perf_counter()
    h.run()
    t_solve = time.perf_counter() - t1
    status = h.getModelStatus()
    obj = (h.getInfoValue("objective_function_value")[1]
           if status == highspy.HighsModelStatus.kOptimal else None)
    return {
        "solver": "highs-lp-relaxation (simplex)",
        "build_s": round(t_build, 3),
        "solve_s": round(t_solve, 3),
        "objective": int(round(obj)) if obj is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Network-structure detection check (demo paper D5/D6)")
    parser.add_argument("--skip-gurobi", action="store_true",
                        help="Skip the Gurobi checks (requires a license)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    inst = create_instance(seed=args.seed)
    print(f"Instance: {inst.n_pools:,} pools x {inst.n_workloads:,} workloads, "
          f"{inst.n_assignments:,} arcs (seed {args.seed})\n")
    report = {
        "instance": {
            "n_pools": inst.n_pools,
            "n_workloads": inst.n_workloads,
            "n_assignments": int(inst.n_assignments),
            "seed": args.seed,
        }
    }

    gurobi_available = not args.skip_gurobi
    if gurobi_available:
        try:
            import gurobipy  # noqa: F401
        except ImportError:
            print("Gurobi not available (gurobipy not installed); skipping Gurobi checks.\n")
            gurobi_available = False

    # Check 1: NetworkAlg discriminator
    if gurobi_available:
        print("Check 1: Gurobi NetworkAlg discriminator (LP relaxation)")
        disc = gurobi_netalg_discriminator(inst)
        report["netalg_discriminator"] = disc
        for r in disc:
            print(f"  {r['config']:<20} iterations={r['iterations']:>10,} "
                  f"runtime={r['runtime_s']:>6}s  obj={r['objective']}")
        off = next(r for r in disc if r["config"] == "primal_network_off")["iterations"]
        on = next(r for r in disc if r["config"] == "primal_network_on")["iterations"]
        verdict = ("NetworkAlg=1 is a no-op: identical iteration count, so Gurobi does "
                   "not engage a network algorithm on this model."
                   if off == on else
                   "NetworkAlg=1 changes the iteration count: a distinct algorithm engaged.")
        report["netalg_verdict"] = {"primal_off_iters": off, "primal_on_iters": on,
                                    "no_op": off == on, "text": verdict}
        print(f"  => {verdict}\n")
    else:
        print("Check 1 (NetworkAlg discriminator) skipped: requires Gurobi.\n")

    # Check 2: fair LP-relaxation baseline vs MCF
    print("Check 2: fair LP-relaxation baseline vs min-cost flow")
    from mcf import solve_mcf
    mcf = solve_mcf(inst)
    print(f"  MCF (network simplex)    solve={mcf['solve_s']:.3f}s  "
          f"total={mcf['time_s']:.3f}s  obj=${mcf['objective']:,}")
    report["mcf"] = mcf

    hl = highs_lp_relaxation(inst)
    print(f"  HiGHS LP relaxation      solve={hl['solve_s']:.3f}s  obj={hl['objective']}")
    report["highs_lp"] = hl

    if gurobi_available:
        gl = gurobi_lp_relaxation(inst)
        print(f"  Gurobi LP relaxation     solve={gl['solve_s']:.3f}s  obj={gl['objective']}")
        report["gurobi_lp"] = gl

    with open("network_detection_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nWrote network_detection_results.json")


if __name__ == "__main__":
    main()
