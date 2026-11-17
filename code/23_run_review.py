"""Experiments added in response to the independent review (REVIEW.md, RESPONSE_TO_REVIEW.md), E1-E7.

  rv_ctrl_   E1-E3  controller study re-run with tail metrics and new variants (ToU, n=25, 5 kWh/home, PV x1 and x3):
                    day-ahead without the half-full end constraint (e0) and with a 21:00-21:00 window (h21);
                    peak-maximising member of the cost-optimal set (lp_maxpeak); caps set from the previous
                    year's peak (capP90/100/110) and from yesterday's maximum (dcap)
  rv_rn_     E4     revenue-neutral demand-charge tariffs (tou_demand_rn, tou_demand24_rn)
  rv_sweep_  E5     ToU shape: price ratio 1.5 and 4 (off-peak rate unchanged), window 17-21 h and 13-21 h,
                    and a free midday window (tou_free)
  ctrl10_    E6     Ausgrid controller study with groups of 10 (like-for-like with the second dataset)
  rv_coord_  E7     coordinated individual batteries with individual bills (2 partitions, PV x1)

Usage: python 23_run_review.py stage1|stage2|coord
"""
import sys
import time

from joblib import Parallel, delayed

import config as cfg
import exp
import runlog
import sim
from exp import spec

N_JOBS = cfg.N_JOBS
PVS = (1.0, 3.0)
RL = ["rule", "lp"]
NEW_B = ["mpc_react_persist_e0", "mpc_react_persist_h21"]
CB = ["rule", "lp", "jit", "spread", "mpc_react_persist"] + NEW_B
CC = ["rule", "lp", "lp_minpeak", "lp_maxpeak", "lp_cap", "jit", "jit_cap", "spread", "mpc_react_persist", "mpc_react_persist_cap",
      "mpc_react_persist_e0", "mpc_react_persist_h21", "mpc_react_persist_h21_cap", "mpc_react_persist_dcap",
      "mpc_react_persist_capP90", "mpc_react_persist_capP100", "mpc_react_persist_capP110"]
OFF = cfg.TOU_OFFPEAK_C
SWEEP = [("tou", {"peak_c": round(1.5 * OFF, 4)}), ("tou", {"peak_c": round(4.0 * OFF, 4)}), ("tou", {"peak_start": 17}),
         ("tou", {"peak_start": 13}), ("tou_free", {})]
RN = ["tou_demand_rn", "tou_demand24_rn"]
CB10 = ["rule", "lp", "lp_random", "jit", "spread", "mpc_react_persist", "mpc_react_persist_cap"]
CC10 = CB10 + ["lp_minpeak", "lp_cap"]

s1 = [(spec(pv=pv, tname="tou"), c) for pv in PVS for c in NEW_B]
s1 += [(spec(pv=pv, tname=t), c) for pv in PVS for t in RN for c in RL]
s1 += [(spec(pv=pv, tname=t, tkw=kw), c) for pv in PVS for t, kw in SWEEP for c in RL]
s2 = [(spec(pv=pv, tname="tou"), 25, 4, CB, CC, "rv_ctrl_") for pv in PVS]
s2 += [(spec(pv=pv, tname=t), 25, 4, RL, RL, "rv_rn_") for pv in PVS for t in RN]
s2 += [(spec(pv=pv, tname=t, tkw=kw), 25, 4, RL, RL + ["lp_minpeak"], "rv_sweep_") for pv in PVS for t, kw in SWEEP]
s2 += [(spec(pv=1.0, tname="tou"), 10, 3, CB10, CC10, "ctrl10_")]


def f1(sp, c):
    exp.stage1(sp, c)


def f2(j):
    sp, n, K, cb, cc, tag = j
    exp.stage2(sp, n, cfg.PARTITION_SEED_BASE + n, K, cb, cc, tag=tag)


def coord_one(part, gi, idx):
    return exp.coordinated_individual(spec(pv=1.0, tname="tou"), [(part, gi, idx)])


if __name__ == "__main__":
    what = sys.argv[1]
    t0 = time.time()
    if what == "stage1":
        Parallel(n_jobs=N_JOBS, verbose=5)(delayed(f1)(sp, c) for sp, c in s1)
        n = len(s1)
    elif what == "stage2":
        Parallel(n_jobs=N_JOBS, verbose=5)(delayed(f2)(j) for j in sorted(s2, key=lambda j: -len(j[4]) - len(j[3])))
        n = len(s2)
    else:
        import pandas as pd
        out = exp.OUT_DIR / "rv_coord_2012-13_gc+cl_pv1_tou_base_cap5_n25_K2.csv"
        n = 0
        if not out.exists():
            d = exp.get_data(spec())
            parts = sim.make_partitions(d, 25, 4, seed=cfg.PARTITION_SEED_BASE + 25)[:2]
            groups = [(k, gi, idx) for k, part in enumerate(parts) for gi, idx in enumerate(part)]
            n = len(groups)
            dfs = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(coord_one)(*g) for g in groups)
            pd.concat(dfs, ignore_index=True).to_csv(out, index=False)
            import json
            out.with_suffix(".meta.json").write_text(json.dumps(dict(
                spec=spec(pv=1.0, tname="tou"), n=25, partitions_seed=cfg.PARTITION_SEED_BASE + 25, n_partitions=2,
                scenario="Bcoord_lp (exp.coordinated_individual)", code_sha=exp.code_sha(), config_sha=cfg.CONFIG_SHA,
                duration_s=round(time.time() - t0, 1), written=time.strftime("%Y-%m-%dT%H:%M:%S")), default=str))
    print(f"done ({time.time()-t0:.0f}s)", flush=True)
    runlog.write("23_run_review.py", t0, n)
