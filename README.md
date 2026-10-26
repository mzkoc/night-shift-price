# The price of protecting the night shift

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23649332.svg)](https://doi.org/10.5281/zenodo.23649332)

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
| `code/reproduce_tables.py` | Recomputes Tables 5 to 7 of the article and Table S4 of Online Resource 1 from the exported best-known schedules, and verifies them with the evaluator |
| `results/best_known_D3/` | The D3 best-known schedules: objective values and indicators (`v9_D3_bks.csv`) and their random-key vectors (`v9_D3_bks_keys.json`) |
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
| Export of the D3 best-known schedules | H | `python -u weaving_gp_v9.py --export` (reads `v9_D3_ckpt.pkl`) |

`weaving_gp_v9.py` contains two changes made after run F that do not affect any search: the
EPDK prices (used in run G) and an ulp-level tolerance for the hit rates of the method tables
(the D1 hit rates in the article are computed with it). The irace scripts call
`weaving_gp_v8.py`; the tuning target is the same in all versions.

D1 and D4 of runs A and E are superseded by run F. Runs D to F used illustrative
time-of-use prices; the electricity bill is a reported indicator only, so run G re-evaluates
the same schedules under the EPDK tariff of 4 April 2026 (the schedules are identical) and
gives the bill figures reported in the article. `python -u weaving_gp_v9.py --only=D3`
runs D3 from scratch in one version (about 6 hours on 18 cores).

## Reproducing the tables of the price experiment

The prices are differences between pooled best-known schedules, not between single runs,
so they cannot be recomputed from the run-level CSV files. The best-known schedules of D3
are therefore exported in a portable form (`results/best_known_D3/`): one row per
instance, scheme and policy with the objective values and indicators, and the random-key
vector of every schedule. From the `code` folder,

```bash
python reproduce_tables.py ../results/best_known_D3/v9_D3_bks.csv --verify
```

recomputes Tables 5 to 7 and Table S4 and, with `--verify`, regenerates the instances,
decodes every key vector with the evaluator and checks that it gives the exported
objective values (480 of 480 schedules are reproduced exactly; the expected output is in
`results/best_known_D3/reproduce_tables_output.txt`). The checkpoint files (`.pkl`) from which the export was made are not
included: they depend on the Python version and are not a safe exchange format.

## Quick test

A quick test of the whole pipeline (10 to 15 minutes):

```bash
python -u weaving_gp_v9.py --quick
```

## Requirements

Python 3.12 or later, `ortools` 9.15.6755 and `matplotlib`; R with the `irace` package for
tuning only. The quick test was run in a clean virtual environment created from
`requirements.txt`.

```bash
python -m pip install -r requirements.txt
```

The experiments were run on an Intel Core i9-13900H (14 cores, 20 threads, 32 GB) with 18
worker processes (`--workers=18`) and 16 CP-SAT threads (`--cp-workers=16`). Results with
other numbers of workers are identical for the heuristics; CP-SAT with another number of
threads may return a different optimal schedule and takes a different time.

## Licence and citation

The code is released under the MIT licence (see `LICENSE`); the results, logs and exported
schedules in `results/` under the Creative Commons Attribution 4.0 licence (CC BY 4.0).
Please cite the article and the archived version of this repository (see `CITATION.cff`).
