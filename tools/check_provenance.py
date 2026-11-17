"""List stage-2 result files whose recorded code/config hash differs from the current code, or that have no
record (files written before run logging was added on 2026-11-15). Exit code 1 if any file is stale."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import config as cfg  # noqa: E402
import exp  # noqa: E402

cur, stale, norec, ok = exp.code_sha(), [], [], 0
for f in sorted((ROOT / "results" / "v2").glob("*.csv")):
    m = f.with_suffix(".meta.json")
    if not m.exists():
        norec.append(f.name)
        continue
    r = json.loads(m.read_text())
    if r["code_sha"] != cur or r["config_sha"] != cfg.CONFIG_SHA:
        stale.append(f.name)
    else:
        ok += 1
print(f"current code {cur}, config {cfg.CONFIG_SHA}: {ok} up to date, {len(stale)} stale, {len(norec)} without record")
for n in stale:
    print("  STALE     ", n)
if "-v" in sys.argv:
    for n in norec:
        print("  NO RECORD ", n)
sys.exit(1 if stale or norec else 0)
