"""Run log: one JSON line per run in results/run_log.jsonl (script, arguments, git commit, dirty flag,
configuration hash, seeds, number of tasks, duration, host). Written by the run scripts."""
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "run_log.jsonl"


def _git(*a):
    try:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def write(script, t0, n_tasks, **extra):
    import numpy, scipy, pandas
    import config as cfg
    rec = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), script=script, argv=sys.argv[1:],
               git_commit=_git("rev-parse", "--short", "HEAD"), git_dirty=bool(_git("status", "--porcelain", "--", "code", "config.yaml")),
               config_sha=getattr(cfg, "CONFIG_SHA", ""), n_jobs=getattr(cfg, "N_JOBS", None), n_tasks=n_tasks,
               duration_s=round(time.time() - t0, 1), python=platform.python_version(),
               numpy=numpy.__version__, scipy=scipy.__version__, pandas=pandas.__version__, host=platform.platform(), **extra)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec
