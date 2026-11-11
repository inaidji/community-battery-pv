# Data

The Ausgrid Solar Home Electricity Data (300 homes, 1 July 2010 - 30 June 2013, CC BY) is not stored in this
repository. Ausgrid's original web page has been removed; the files were retrieved from the Internet Archive
copy of the Ausgrid "Solar home electricity data" release (see the data availability statement and
`paper/refs.bib`, entry `ausgrid2014notes`; also listed at
https://data.nsw.gov.au/data/dataset/solar-home-electricty-data).

Place the three yearly CSV files in `data/extracted/` with these names (the code reads them with `skiprows=1`):

- `2010-2011 Solar home electricity data.csv`
- `2011-2012 Solar home electricity data v2.csv`
- `2012-2013 Solar home electricity data v2.csv`

Processed caches are written to `data/processed/` automatically.
