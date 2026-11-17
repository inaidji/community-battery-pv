"""Additional experiments approved after the audit (PLAN.md 3.1-3.3). Resumable like 20_run_all.py.

  ctrl2  reactive day-ahead controllers, import-cap variants of the operational rules, and the persistence
         controller without day-1 foresight (supersedes mpc_persist in ctrl_)            [PLAN 3.1]
  seeds  four more independent draws of the random tie-break (lp_random_s1..s4)          [PLAN 3.2]
  batt   battery C-rate (0.25, 1.0) and one-way efficiency (0.90, 0.98), ToU, n=25       [PLAN 3.3]

Usage: python 21_run_extra.py stage1 | stage2
"""
import sys
import time

from joblib import Parallel, delayed

import config as cfg
import exp
from exp import spec

import config as cfg
import runlog

N_JOBS = cfg.N_JOBS
PV = cfg.PV_SCALES
SEED = lambda n: 1000 + n
CTRL2 = ["mpc_persist", "mpc_react_perfect", "mpc_react_persist", "mpc_react_persist_cap", "jit_cap", "spread_cap"]
SEEDS = [f"lp_random_s{k}" for k in range(1, 5)]
BATT = [{"c_rate": 0.25}, {"c_rate": 1.0}, {"eta": 0.90}, {"eta": 0.98}]
TOU = [spec(pv=pv, tname="tou", cap=5.0) for pv in PV]
BSPECS = [spec(pv=pv, tname="tou", cap=5.0, batt=b) for b in BATT for pv in PV]

s1_tasks = [(sp, c) for sp in TOU for c in SEEDS + CTRL2] + [(sp, c) for sp in BSPECS for c in ("lp", "rule")]
RL = ["rule", "lp"]
s2_jobs = [(sp, 25, 4, CTRL2, CTRL2, "ctrl2_") for sp in TOU] + [(sp, 25, 4, SEEDS, SEEDS, "seeds_") for sp in TOU]
for sp in BSPECS:
    cc = RL + (["lp_minpeak"] if ("c_rate" in sp["batt"] and sp["pv"] == 1.0) else [])
    s2_jobs.append((sp, 25, 4, RL, cc, "batt_"))


def s1(sp, c):
    exp.stage1(sp, c)
    return 1


def s2(job):
    sp, n, K, cb, cc, tag = job
    exp.stage2(sp, n, SEED(n), K, cb, cc, tag=tag)
    return 1


if __name__ == "__main__":
    t0 = time.time()
    what = sys.argv[1]
    if what == "stage1":
        print(f"stage 1: {len(s1_tasks)} tasks", flush=True)
        Parallel(n_jobs=N_JOBS, verbose=5)(delayed(s1)(sp, c) for sp, c in s1_tasks)
    else:
        print(f"stage 2: {len(s2_jobs)} jobs", flush=True)
        Parallel(n_jobs=N_JOBS, verbose=5)(delayed(s2)(j) for j in s2_jobs)
    print(f"done ({time.time()-t0:.0f}s)", flush=True)
    runlog.write("21_run_extra.py", t0, len(s1_tasks) + len(s2_jobs))
