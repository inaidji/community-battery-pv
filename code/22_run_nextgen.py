"""Second dataset (NextGen, ACT, 2018; 99 homes with 6 kWp PV on average): same scenarios, tariffs and
controllers as the main study, 5 kWh per home, measured PV (no scaling). Groups: blocks of a fixed random
order (no postcode in the data); n=10 -> 9 groups per partition, n=25 -> 3 groups per partition.
The Ausgrid tariffs are applied unchanged so that only the data differ (these are ACT homes on another network).

Usage: python 22_run_nextgen.py            main version: no imputed day (52 homes, days complete for all homes)
       python 22_run_nextgen.py imputed    sensitivity: 99 homes, missing days imputed (results tagged ng_)
"""
import sys
import time

from joblib import Parallel, delayed

import exp
from exp import spec

import config as cfg
import runlog

N_JOBS = cfg.N_JOBS
STRICT = len(sys.argv) < 2 or sys.argv[1] != "imputed"
Y = "nextgen-2018s" if STRICT else "nextgen-2018"
TAG = "ngs" if STRICT else "ng"
TARIFFS = ("flat", "tou", "tou_demand", "tou_demand24")
CB = ["rule", "lp", "lp_random", "jit", "spread", "mpc_react_persist", "mpc_react_persist_cap"]
CC = CB + ["lp_minpeak", "lp_cap"]
RL = ["rule", "lp"]
s1 = [(spec(year=Y, tname=t), c) for t in TARIFFS for c in RL] + [(spec(year=Y, tname="tou"), c) for c in CB if c not in RL]
s2 = [(spec(year=Y, tname=t), n, 4, RL, RL, f"{TAG}_main_") for t in TARIFFS for n in (10, 25)]
s2 += [(spec(year=Y, tname="tou"), 10, 4, CB, CC, f"{TAG}_ctrl_")]


def f1(sp, c):
    exp.stage1(sp, c)


def f2(j):
    sp, n, K, cb, cc, tag = j
    exp.stage2(sp, n, 1000 + n, K, cb, cc, tag=tag)


if __name__ == "__main__":
    t0 = time.time()
    Parallel(n_jobs=N_JOBS, verbose=5)(delayed(f1)(sp, c) for sp, c in s1)
    print(f"stage 1 done ({time.time()-t0:.0f}s)", flush=True)
    Parallel(n_jobs=N_JOBS, verbose=5)(delayed(f2)(j) for j in s2)
    print(f"done ({time.time()-t0:.0f}s)", flush=True)
    runlog.write("22_run_nextgen.py", t0, len(s1) + len(s2))
