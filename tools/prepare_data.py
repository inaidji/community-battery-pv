"""Verify the raw Ausgrid archives and extract them. Raw files are never modified.

The three zips are the Ausgrid "Solar home electricity data" yearly releases, retrieved from the
Internet Archive because Ausgrid's page has been removed. The original file URL of the 2012-13 archive is
https://www.ausgrid.com.au/-/media/Documents/Data-to-share/Solar-home-electricity-data/Solar-home-half-hour-data---1-July-2012-to-30-June-2013.zip
(Internet Archive snapshot of 26 December 2025). The URLs of the other two years follow the same naming
pattern but were not recorded at download time: [TO VERIFY] before publication.

Usage: python tools/prepare_data.py           (checks SHA-256, extracts to data/extracted/)
"""
import hashlib
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "extracted"
SHA256 = {
    "Solar-home-half-hour-data_1-July-2010-to-30-June-2011.zip": "0fdfbaaba469b91a54477d4061ee226127d95863888c2ee0bf0b02db8b491055",
    "Solar-home-half-hour-data_1-July-2011-to-30-June-2012.zip": "e16f9143147c28e12f9686cf15869ac08a09724ac9074c9341d1bf0d5669155e",
    "Solar-home-half-hour-data_1-July-2012-to-30-June-2013.zip": "6949ffee7ef8e2260f229f8a7e3b992390187facaaf023bb933b811a11cd1a11",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, want in SHA256.items():
        f = RAW / name
        if not f.exists():
            print(f"MISSING  {name}  (place it in data/raw/; see data/README.md)")
            ok = False
            continue
        got = hashlib.sha256(f.read_bytes()).hexdigest()
        if got != want:
            print(f"CHECKSUM MISMATCH  {name}\n  expected {want}\n  got      {got}")
            ok = False
            continue
        with zipfile.ZipFile(f) as z:
            for m in z.namelist():
                if not (OUT / Path(m).name).exists():
                    z.extract(m, OUT)
        print(f"ok       {name}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
