# DeQL Studio: Declarative Decision-Making over Relational Data

**Authors:** Matteo Brucato, Fjodor Kholodkov, Soren Little, Jakob Mayer, Duc Nguyen
**Affiliation:** OSM Data
**Venue:** VLDB 2026 Demo Track

This repository contains the benchmark scripts for reproducing the solver comparison results reported in the paper.

## Quick start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run all solvers with Gurobi, skipping HiGHS (saves ~6 minutes)
uv run --extra gurobi run_all.py --skip-highs

# Full run including HiGHS (~7 minutes)
uv run --extra gurobi run_all.py
```

Gurobi requires a [license](https://www.gurobi.com/academia/academic-program-and-licenses/) (free for academics) and the `gurobi` extra. Passing `--extra gurobi` (as above) installs and uses it; **without `--extra gurobi`, gurobipy is not installed and the Gurobi runs are skipped automatically.** Drop the flag to run only the open-source solvers (MCF, HiGHS, greedy).

## What this runs

The benchmark generates a GPU allocation instance (1,000 pools, 4,000 workloads, ~2.7M candidate assignments) and solves it with four approaches:

| Script | Formulation | Expected result |
|--------|------------|-----------------|
| `greedy.py` | Cheapest-pool heuristic | ~4s, $856K, **infeasible** (107 violations) |
| `mcf.py` | Min-cost network flow | ~0.9s, $554,053, **optimal** |
| `gurobi.py` | MILP (Gurobi) | ~26s, $554,053, **optimal** |
| `highs.py` | MILP (HiGHS) | >6min, $554,053, **optimal** |

MCF reaches the same optimal cost about 30x faster than the conventional MILP formulation a user writes without analyzing the problem's structure. Even the LP relaxation that a general-purpose solver's presolve reaches (the constraint matrix is totally unimodular, so the binary variables drop out) is far slower than MCF, and Gurobi's network-simplex parameter does not engage on this model; `network_detection.py` reproduces both points. The DeQL engine selects MCF automatically by detecting transportation structure from the query.

## Individual scripts

Each solver can also be run standalone:

```bash
uv run greedy.py
uv run mcf.py
uv run --extra gurobi gurobi.py
uv run highs.py
```

## Network-structure detection

`network_detection.py` reproduces two points behind the 30x comparison:

- Gurobi's `NetworkAlg=1` is a no-op on this model: the LP relaxation takes the
  same number of simplex iterations with the network parameter on as off, so a
  general-purpose solver does not engage a network algorithm even when asked to.
- The fair like-for-like baseline (the LP relaxation, solved by Gurobi or HiGHS)
  is still far slower than min-cost flow.

```bash
uv run --extra gurobi network_detection.py   # both checks (check 1 needs Gurobi)
uv run network_detection.py --skip-gurobi     # HiGHS LP + MCF baseline only
```

On the demo instance, check 1 prints identical simplex iteration counts for
`NetworkAlg` off vs on (212,986 in our run), and check 2 shows MCF (~0.9s) beating
the Gurobi (~3.8s) and HiGHS (~44s) LP relaxations, all at the same $554,053 optimum.
