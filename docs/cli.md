# `marswalk` CLI — end-to-end mission demo

The `marswalk` console command (entry point `marswalk.cli:main`) runs the whole
planner pipeline on synthetic terrain and prints a judge-facing mission report.
It drives the **real** modules — nothing in the output is hand-authored:

1. **Three-route comparison** (`planner.routes.compare_routes`) — shortest vs
   slope-aware vs full-constraint, with rejection reasons.
2. **Walk-back mission constraint** (`planner.mission.compare_routes_with_walkback`)
   across the three standard contingencies (nominal / rover-loss / suit-degraded),
   showing the decision flip.
3. **Route stability** (`validation.stability.route_stability`) over a DEM error
   ensemble (C2): how often the recommended route changes under plausible error.

## Usage

```bash
pip install -e .
marswalk mission                 # full text report on default ridge terrain
marswalk mission --json          # machine-readable
marswalk mission --terrain random --seed 3
marswalk mission --start 0,0 --goal 39,0 --base 0,0 --rover 39,0 --conservative
```

Running without a subcommand defaults to `mission`. Without installing, use
`PYTHONPATH=src python -m marswalk.cli mission`.

## Key flags

| Flag | Meaning |
|---|---|
| `--terrain ridge\|random` | synthetic terrain generator |
| `--start`, `--goal` | `row,col` endpoints (default corner → opposite corner) |
| `--base`, `--rover` | walk-back haven cells (default base=start, rover=goal) |
| `--reserve` | walk-back time reserve, seconds |
| `--conservative` | use the conservative (tighter) consumables set |
| `--stability-n` | DEM realizations to replan over |
| `--sill-m2`, `--range-m` | DEM error variance / correlation length (PLACEHOLDERS) |
| `--json` | emit JSON instead of text |

## Validity

Offline, synthetic terrain only. **All physics and threshold parameters are
PLACEHOLDERS** (see [`ASSUMPTIONS.md`](../ASSUMPTIONS.md)) and the DEM error
model must be fitted from real residuals by an owner. The command demonstrates
the decision machinery; it is **UNTESTED ON REAL DATA**.
