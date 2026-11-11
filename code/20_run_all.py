"""Run every experiment of the revision (resumable: stage-1 and stage-2 outputs are cached on disk).

Experiments (all 2012-13 unless stated; n = homes per group; groups from non-overlapping partitions):
  main     3 sizes x 5 tariffs x 2 PV x 3 battery sizes, rule + optimal, B / Bagg / C
  years    2010-11 and 2011-12, n=25, 5 kWh/home
  sens     tariff-parameter variants (feed-in, demand charge, export limit, market-offer ToU rates)
  clmix    controlled load excluded; 1/3 of homes without PV
  ctrl     controller comparison (tie-breaks, heuristics, day-ahead MPC, peak cap) at n=25, ToU
"""
import sys
import time

from joblib import Parallel, delayed

import config as cfg
import exp
from exp import spec

N_JOBS = 7
PV = cfg.PV_SCALES
SEED = lambda n: 1000 + n

# ----- spec lists -----------------------------------------------------------------------------
MAIN = [spec(pv=pv, tname=t, cap=c) for pv in PV for t in cfg.TARIFFS for c in cfg.BATT_KWH_PER_HOME_SWEEP]
YEARS = [spec(year=y, pv=pv, tname=t, cap=5.0) for y in ("2010-11", "2011-12") for pv in PV for t in cfg.TARIFFS]
SENS_VARIANTS = [("flat", {"fit": 0.0}), ("flat", {"fit": 10.0}), ("tou", {"fit": 0.0}), ("tou", {"fit": 10.0}),
                 ("tou_demand", {"demand_aud": 0.2}), ("tou_demand", {"demand_aud": 0.8}),
                 ("tou_demand24", {"demand_aud": 0.2}), ("tou_demand24", {"demand_aud": 0.8}),
                 ("tou_exportlimit", {"export_limit": 3.0}), ("tou_exportlimit", {"export_limit": 5.0}),
                 ("tou", {"peak_c": cfg.TOU_PEAK_C_MARKET, "offpeak_c": cfg.TOU_OFFPEAK_C_MARKET})]
SENS = [spec(pv=pv, tname=t, tkw=kw, cap=5.0) for t, kw in SENS_VARIANTS for pv in PV]
CL = [spec(loadvar="gc", pv=pv, tname=t, cap=5.0) for t in ("flat", "tou") for pv in PV]
NOPV0 = [spec(pv=0.0, tname=t, cap=5.0) for t in ("flat", "tou")]
CTRL_B = ["rule", "lp", "lp_random", "lp_cap", "jit", "spread", "mpc_perfect", "mpc_persist"]
CTRL_C = CTRL_B + ["lp_minpeak"]
CTRL = [spec(pv=pv, tname="tou", cap=5.0) for pv in PV]

# ----- stage 1 tasks (spec, controller) ------------------------------------------------------
COST = {"lp": 0, "lp_random": 0, "lp_cap": 0, "mpc_perfect": 1, "mpc_persist": 1, "rule": 2, "jit": 2, "spread": 2}
tasks = {}
def need(sps, ctrls):
    for sp in sps:
        for c in ctrls:
            tasks[(exp.spec_label(sp), c)] = (sp, c)
need(MAIN + YEARS + SENS + CL + NOPV0, ["rule", "lp"])
need(CTRL, CTRL_B)
task_list = sorted(tasks.values(), key=lambda t: COST[t[1]])


def s1(sp, c):
    exp.stage1(sp, c)
    return 1


def s2(job):
    sp, n, K, cb, cc, mix, tag = job
    exp.stage2(sp, n, SEED(n), K, cb, cc, nopv_every=mix, tag=tag)
    return 1


if __name__ == "__main__":
    t0 = time.time()
    print(f"stage 1: {len(task_list)} (spec, controller) tasks", flush=True)
    Parallel(n_jobs=N_JOBS, verbose=5)(delayed(s1)(sp, c) for sp, c in task_list)
    print(f"stage 1 done ({time.time()-t0:.0f}s)", flush=True)

    jobs = []
    RL = ["rule", "lp"]
    for sp in MAIN:
        for n, K in ((10, 3), (25, 4), (50, 4)):
            jobs.append((sp, n, K, RL, RL, 0, "main_"))
    for sp in YEARS:
        jobs.append((sp, 25, 4, RL, RL, 0, "years_"))
    for sp in SENS:
        jobs.append((sp, 25, 4, RL, RL, 0, "sens_"))
    for sp in CL:
        jobs.append((sp, 25, 4, RL, RL, 0, "cl_"))
    for t in ("flat", "tou"):
        for pv in PV:
            jobs.append((spec(pv=pv, tname=t, cap=5.0), 25, 4, RL, RL, 3, "mix_"))
    for sp in CTRL:
        jobs.append((sp, 25, 4, CTRL_B, CTRL_C, 0, "ctrl_"))
    jobs.sort(key=lambda j: 0 if j[6] == "ctrl_" else 1)          # slowest (minpeak) first
    print(f"stage 2: {len(jobs)} jobs", flush=True)
    Parallel(n_jobs=N_JOBS, verbose=5)(delayed(s2)(j) for j in jobs)
    print(f"done ({time.time()-t0:.0f}s)", flush=True)
