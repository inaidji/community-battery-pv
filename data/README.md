# Data

The raw data are not stored in this repository. Both datasets keep their own licences.

## Ausgrid Solar Home Electricity Data (300 homes, 1 July 2010 - 30 June 2013, CC BY)

Ausgrid's original web page has been removed; the three yearly archives were retrieved from the Internet Archive
copy of the Ausgrid "Solar home electricity data" release (also listed at
https://data.nsw.gov.au/data/dataset/solar-home-electricty-data). Place them unchanged in `data/raw/`:

- `Solar-home-half-hour-data_1-July-2010-to-30-June-2011.zip`
- `Solar-home-half-hour-data_1-July-2011-to-30-June-2012.zip`
- `Solar-home-half-hour-data_1-July-2012-to-30-June-2013.zip`

`make data` (`tools/prepare_data.py`) checks their SHA-256 sums and extracts them to `data/extracted/`.

## NextGen energy storage trial, ACT, 2018 (CC BY 4.0)

Zenodo, doi:10.5281/zenodo.14885589. Place the per-home CSV files unchanged in `data/raw_nextgen/`; their MD5 sums are
in `tools/nextgen_md5.txt`. `code/40_prepare_nextgen.py` builds the strict (no imputed day) and imputed versions.

Processed data and caches are written to `data/processed/` (remove with `make clean-results`).
