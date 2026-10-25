# The price of protecting the night shift

Code, instances and raw results for

> M. Z. Koç. *The Price of Protecting the Night Shift: Goal Programming and Reliable
> Counterfactual Pricing for Weaving Scheduling with Drawing-in Crews.*
> Submitted to Annals of Operations Research (special issue MOPGP'25).

The study schedules weaving beams on parallel looms. Drawing-in (the long changeover
between configuration families) is done by a few crews that work only in the day and
evening shifts. The paper measures what this "night rule" costs in tardiness, drawing-in
work and night work, using lexicographic and Chebyshev goal programming. The models are
solved with CP-SAT (exact, small instances), dispatching rules and GP-RKO (a rule-seeded,
goal-programming random-key optimizer).

## Contents

| Path | Content |
|---|---|
| `code/weaving_gp_v9.py` | Current version: evaluator, CP-SAT model, rules, GP-RKO, BRKGA, BRKGA-S, SEED-LS, all experiments, with efficiency restoration (priority 3) and experiment D6 |
| `code/weaving_gp_v8.py` | Version used for the D3 runs and polishing (run D) and for D5 (run E) |
| `code/weaving_gp_v7.py` | Version used for D2 (run A) |
| `code/irace_tune.R`, `code/irace_tune_brkga.R` | irace configuration of GP-RKO and of the seeded BRKGA |
| `code/brkga_s_cfg.json` | Tuned BRKGA-S configuration (read at start-up) |
| `results/run*/` | Log and one CSV row per run for every experiment |

Instances are not stored separately: every instance is generated deterministically by
`gen_instance` / `d3_instances` from the calibration parameters and the seeds given in the
paper. The CSV files carry the instance name, the factor levels, the scheme, the policy,
the method, the replication and all objective values and indicators.

## Which run produced which result

| Experiment (paper) | Run | Command |
|---|---|---|
| D2 algorithm comparison (sets M, L) | A | `python -u weaving_gp_v7.py --only=D1,D2` |
| D3 price map: runs and polishing | D | `python -u weaving_gp_v8.py --only=D3` |
| D5 convergence | E | `python -u weaving_gp_v8.py --only=D1,D5,D4` (reads the D3 checkpoint) |
| D3 restoration and night stages, tables; D1; D6; D4 | F | `python -u weaving_gp_v9.py --only=D1,D3,D6,D4 --resume` (continues from `v8_D3_ckpt.pkl`) |
| Electricity bill under the EPDK tariff (D3 best-known schedules, D4) | G | `python -u weaving_gp_v9.py --reprice` (reads `v9_D3_ckpt.pkl`) |

`weaving_gp_v9.py` contains two changes made after run F that do not affect any search: the
EPDK prices (used in run G) and an ulp-level tolerance for the hit rates of the method tables
(the D1 hit rates in the article are computed with it). The irace scripts call
`weaving_gp_v8.py`; the tuning target is the same in all versions.

D1 and D4 of runs A and E are superseded by run F. Runs D to F used illustrative
time-of-use prices; the electricity bill is a reported indicator only, so run G re-evaluates
the same schedules under the EPDK tariff of 4 April 2026 (the schedules are identical) and
gives the bill figures reported in the article. `python -u weaving_gp_v9.py --only=D3`
runs D3 from scratch in one version (about 6 hours on 18 cores).

A quick test of the whole pipeline (10 to 15 minutes):

```bash
python -u weaving_gp_v9.py --quick
```

## Requirements

Python 3.12, `ortools` 9.15.6755, `matplotlib`; R with the `irace` package for tuning only.

```bash
python -m pip install -r requirements.txt
```

The experiments were run on an Intel Core i9-13900H (14 cores, 20 threads, 32 GB) with 18
worker processes (`--workers=18`) and 16 CP-SAT threads (`--cp-workers=16`). Results with
other numbers of workers are identical for the heuristics; CP-SAT with another number of
threads may return a different optimal schedule and takes a different time.

## Licence and citation

MIT licence (see `LICENSE`). Please cite the article and the archived version of this
repository (see `CITATION.cff`).
