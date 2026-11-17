# Community versus individual batteries for PV households

Code, configuration, tests and result files for the paper "Community versus individual batteries for PV
households: how control, tariff design and billing shape imports at a shared connection and savings in a simulation
study on Australian data" (in preparation).
The manuscript sources (`paper/`), the review documents and the superseded first version (`legacy/`) are private and not part of
the public repository; `make all` skips the paper build when `paper/` is absent.

## One command

```
pip install -r requirements.txt      # Python 3.11, pinned versions
make all                             # data check -> tests -> experiments -> tables -> figures -> numbers -> checks
```

`make all` takes about 11 hours on 7 cores (measured on a clean clone, 2026-11-16/17) (`make all JOBS=4` to change the core count). Individual steps:

| Target | What it does |
|---|---|
| `make data` | verifies SHA-256 of the three Ausgrid archives in `data/raw/` and extracts them (`tools/prepare_data.py`) |
| `make test` | 20 unit tests and a three-home smoke run |
| `make experiments` | main experiment set (`code/20_run_all.py`) |
| `make extra` | reactive day-ahead controllers, capped rules, 5 tie-break draws, C-rate and efficiency (`code/21_run_extra.py`) |
| `make nextgen` | second dataset: preprocessing (`code/40_prepare_nextgen.py`), strict and imputed versions (`code/22_run_nextgen.py`) |
| `make review` | experiments added after the independent review (`code/23_run_review.py`) |
| `make fairness` | exact Shapley analysis |
| `make tables` | all tables, statistical tests, interval coverage check (`results/v2_tables/`) |
| `make figures` | vector figures, readable in greyscale (`results/figures_v2/`) |
| `make numbers` | number registry and LaTeX macros/tables (`code/35_latex.py`) |
| `make check` | every number in the text is traceable (`code/36_check_numbers.py`); results match the current code (`tools/check_provenance.py`) |

## Layout

| Path | Content |
|---|---|
| `config.yaml` | every assumption and experiment setting (`code/config.py` only loads it) |
| `code/sim.py` | data loading, partitions, battery, controllers, billing, metrics |
| `code/exp.py` | pipeline: stage 1 per-home dispatch (cached), stage 2 group evaluation, provenance records |
| `code/analysis_lib.py` | estimators and confidence intervals |
| `results/v2/` | raw group-level results (CSV) with a `.meta.json` provenance record each; `results/run_log.jsonl` |
| `results/v2_tables/` | tables; `numbers_registry.csv` lists every number used in the paper with its source file |
| `results/figures_v2/` | figures (PDF and 300 dpi PNG) |
| `tools/` | raw-data checksums and extraction, provenance check |
| `tests/` | unit tests |

## Data

- Ausgrid Solar Home Electricity Data (CC BY): Ausgrid's page has been removed; the archives were retrieved from the
  Internet Archive. Checksums are in `tools/prepare_data.py`.
- NextGen energy storage trial, ACT, 2018 (CC BY 4.0): Zenodo, doi:10.5281/zenodo.14885589. MD5 sums in
  `tools/nextgen_md5.txt`. Place the CSV files in `data/raw_nextgen/`.

Raw data are never modified; all preprocessing is scripted.

## Caveat on caches

Stage-1 per-home results are cached in `data/processed/` under the scenario name only. If you change a tariff,
battery or PV parameter in `config.yaml`, run `make clean-results` before `make all`; otherwise stale cached
results may be reused.

## Licence

MIT (see `LICENSE`).

## Citation

See `CITATION.cff`. Archived on Zenodo; version 1.0.0: https://doi.org/10.5281/zenodo.24129444.
Version 2.0.0 is the code and results behind the revised paper. Add the paper citation after publication.
