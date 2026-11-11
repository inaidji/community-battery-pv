# Community versus individual batteries for PV households

Code and results for the paper "Community versus individual batteries for PV households:
how control, tariff design and billing shape group-level peak import and savings", in preparation.

## Layout

| Path | Content |
|---|---|
| `code/config.py` | tariffs (AER 2026-27 DMO caps), battery, cost and sensitivity parameters |
| `code/sim.py` | data loading, groups/partitions, dispatch (rule, LP, tie-breaks, heuristics, day-ahead MPC), billing, metrics |
| `code/exp.py`, `code/20_run_all.py` | experiment pipeline: stage 1 per-home dispatch (cached), stage 2 group evaluation |
| `code/30_tables.py`, `code/31_figures.py` | tables (`results/v2_tables/`) and figures (`results/figures_v2/`) |
| `code/06_fairness.py`, `code/32_fairness_summary.py` | exact Shapley benefit-sharing analysis |
| `tests/` | pytest checks (limits, state of charge, bill never worse with LP, non-negative losses) |
| `results/v2/` | raw per-group results (CSV) behind every number in the paper |

## Reproduce

1. `pip install -r requirements.txt` (Python 3.11; HiGHS via SciPy).
2. Put the dataset in `data/extracted/` (see `data/README.md`).
3. `cd code && python 20_run_all.py` - about 38 min (stage 1) + 115 min (stage 2) on 7 cores.
4. `python 06_fairness.py && python 32_fairness_summary.py`, then `python 30_tables.py && python 31_figures.py`.
5. `python -m pytest tests` from the repository root.

The tables and figures in the paper can be regenerated from the shipped `results/v2/*.csv` without re-running the
simulation (steps 4 only; the data are needed only for `06_fairness.py` and the tests).

## Notes

- Tariff energy prices are from the Australian Energy Regulator's 2026-27 DMO final determination; the feed-in
  tariff, demand charge and export limit are assumptions (varied in the sensitivity analysis).
- Much of the code was written with the help of a generative AI assistant (see the paper's Generative AI
  statement); results should be checked against the shipped CSVs and the tests.
- Licence: MIT for the code (copyright Ilyes Naidji). The dataset is CC BY and
  keeps its own licence.

## Citation

Archived release: https://doi.org/10.5281/zenodo.24129444 (see `CITATION.cff`). Add the paper citation after publication.
