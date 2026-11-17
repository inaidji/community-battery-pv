"""Configuration loader. All values live in ../config.yaml; this module exposes them in upper case
(cfg.BATT_C_RATE, cfg.TOU_PEAK_C, ...) so that no constant is defined in code. Derived values are below."""
import hashlib
import os
from pathlib import Path

import yaml

CONFIG_FILE = Path(__file__).resolve().parents[1] / "config.yaml"
_raw = CONFIG_FILE.read_bytes()
CONFIG_SHA = hashlib.sha256(_raw).hexdigest()[:12]
_cfg = yaml.safe_load(_raw)
for _section in _cfg.values():
    for _k, _v in _section.items():
        globals()[_k.upper()] = tuple(_v) if isinstance(_v, list) else _v

# derived
BATT_COST_COMMUNITY_NET_AUD_PER_KWH = BATT_COST_COMMUNITY_AUD_PER_KWH - ARENA_MAX_GRANT_AUD_PER_KWH  # noqa: F821
N_JOBS = int(os.environ.get("CB_JOBS", N_JOBS))  # noqa: F821
