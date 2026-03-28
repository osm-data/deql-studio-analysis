# DeQL Studio: Declarative Decision-Making over Relational Data

**Authors:** Matteo Brucato, Fjodor Kholodkov, Soren Little, Jakob Mayer, Duc Nguyen
**Affiliation:** OSM Data
**Submitted to:** VLDB 2026 Demo Track

This repository contains the benchmark scripts for reproducing the solver comparison results reported in the paper.

## Quick start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run all solvers (skip HiGHS to save ~6 minutes)
uv run run_all.py --skip-highs

# Full run including HiGHS (~7 minutes)
uv run run_all.py
```

Gurobi requires a [license](https://www.gurobi.com/academia/academic-program-and-licenses/) (free for academics). If not installed, it is skipped automatically.

## What this runs

The benchmark generates a GPU allocation instance (1,000 pools, 4,000 workloads, ~2.7M candidate assignments) and solves it with four approaches:

| Script | Formulation | Expected result |
|--------|------------|-----------------|
| `greedy.py` | Cheapest-pool heuristic | ~4s, $856K, **infeasible** (107 violations) |
| `mcf.py` | Min-cost network flow | ~0.9s, $554,053, **optimal** |
| `gurobi.py` | MILP (Gurobi) | ~26s, $554,053, **optimal** |
| `highs.py` | MILP (HiGHS) | >6min, $554,053, **optimal** |

MCF reaches the same optimal cost 30x faster than the fastest general-purpose solver. The DeQL engine selects MCF automatically by detecting transportation structure from the query.

## Individual scripts

Each solver can also be run standalone:

```bash
uv run greedy.py
uv run mcf.py
uv run gurobi.py
uv run highs.py
```
