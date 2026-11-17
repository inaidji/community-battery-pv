# Single entry point. `make all` regenerates every result, table, figure and the paper from the raw data.
# Override the interpreter or core count:  make all PY=python3 JOBS=4
PY   ?= python3
JOBS ?= 7
export CB_JOBS := $(JOBS)

.PHONY: all data test smoke experiments extra nextgen review fairness tables figures numbers paper check clean-results

all: data test experiments extra nextgen review fairness tables figures numbers check paper

data:            ## verify checksums of the raw archives and extract them
	$(PY) tools/prepare_data.py

test:            ## unit tests + 3-home smoke run
	$(PY) -m pytest -q tests

experiments:     ## main experiment set (about 2.5 h on 7 cores)
	mkdir -p results/v2 results/v2_tables results/figures_v2
	cd code && $(PY) 20_run_all.py > ../results/v2/run_resume.log 2>&1

extra:           ## experiments added after the audit (about 45 min on 7 cores)
	cd code && $(PY) 21_run_extra.py stage1 > ../results/v2/run_extra_stage1.log 2>&1 && $(PY) 21_run_extra.py stage2 > ../results/v2/run_extra_stage2.log 2>&1

nextgen:         ## second dataset (NextGen ACT 2018): strict version (main) and imputed version (sensitivity); about 35 min
	cd code && $(PY) 40_prepare_nextgen.py > ../results/v2/prepare_nextgen.log 2>&1 && $(PY) 22_run_nextgen.py > ../results/v2/run_nextgen_strict.log 2>&1 && $(PY) 22_run_nextgen.py imputed > ../results/v2/run_nextgen.log 2>&1

review:          ## experiments added after the independent review (about 4 h)
	cd code && $(PY) 23_run_review.py stage1 > ../results/v2/run_review_stage1.log 2>&1 && $(PY) 23_run_review.py stage2 > ../results/v2/run_review_stage2.log 2>&1 && $(PY) 23_run_review.py coord > ../results/v2/run_review_coord.log 2>&1

fairness:        ## exact Shapley analysis
	cd code && $(PY) 06_fairness.py && $(PY) 32_fairness_summary.py

tables:          ## all result tables (CSV) and statistical tests
	cd code && $(PY) 30_tables.py > ../results/v2_tables/tables.log && $(PY) 33_derived.py > ../results/v2_tables/derived.log && $(PY) 34_stats.py > ../results/v2_tables/stats.log && $(PY) 37_nextgen_tables.py > ../results/v2_tables/nextgen.log 2>&1 && $(PY) 38_coverage.py > ../results/v2_tables/coverage.log && $(PY) 39_review_tables.py > ../results/v2_tables/review.log

figures:
	cd code && $(PY) 31_figures.py

numbers:         ## LaTeX tables and number macros generated from results/
	cd code && $(PY) 35_latex.py

paper:           ## private; needs a TeX installation. Builds main.pdf, supplementary.pdf and the Frontiers version (skipped if paper/ is absent)
	@if [ ! -f paper/main.tex ]; then echo "paper/ not present (public repository): skipping the paper build"; exit 0; fi; \
	cd paper && pdflatex -interaction=nonstopmode main >/dev/null && bibtex main >/dev/null && pdflatex -interaction=nonstopmode main >/dev/null && pdflatex -interaction=nonstopmode main >/dev/null && \
	pdflatex -interaction=nonstopmode supplementary >/dev/null && \
	$(PY) make_frontiers.py && cd frontiers && pdflatex -interaction=nonstopmode frontiers >/dev/null && bibtex frontiers >/dev/null && pdflatex -interaction=nonstopmode frontiers >/dev/null && pdflatex -interaction=nonstopmode frontiers >/dev/null

check:           ## every number in the paper text must be traceable to results/; results must match the current code
	cd code && $(PY) 36_check_numbers.py
	$(PY) tools/check_provenance.py

clean-results:   ## remove caches and results (NOT raw data) to force a full re-run
	rm -rf data/processed results/v2/*.csv results/v2/*.meta.json results/v2/*.log results/v2_tables/* results/figures_v2/* results/fairness*.csv results/run_log.jsonl paper/generated
