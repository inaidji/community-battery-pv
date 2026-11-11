"""Fairness of sharing a community battery's savings among members (n = 10 homes, exact Shapley).

Coalition value v(S) = sum_{i in S} bill_A_i - bill_C(S): the saving when the homes in S pool into one
connection with a community battery of 5 kWh x |S| (rule-based dispatch, the realistic controller).
Allocation rules compared: equal split, proportional to consumption, proportional to standalone bill,
Shapley. Benchmark for each home: its saving with its own 5 kWh battery (scenario B).
Output: results/fairness.csv (10 non-overlapping groups of 10 homes)
"""
import itertools
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from math import factorial

import config as cfg
import sim

N, CAP, N_GROUPS = 10, 5.0, 10
TARIFFS = ("flat", "tou", "tou_demand")


def one(gi, idx, tname, pv_scale):
    d = sim.load_headline()
    tariff = sim.make_tariff(tname, d["dates"])
    load, pv = d["load"][idx], d["pv"][idx] * pv_scale
    bill_A = np.array([sim.bill(load[i] - pv[i], tariff) for i in range(N)])
    bill_B = np.array([sim.bill(sim.home_flow(load[i], pv[i], CAP, "rule", tariff)[0], tariff)
                       for i in range(N)])
    # value of every coalition (bitmask)
    v = np.zeros(1 << N)
    for mask in range(1, 1 << N):
        members = [i for i in range(N) if mask >> i & 1]
        flow, _, _ = sim.group_flow(load[members], pv[members], CAP * len(members), "rule", tariff)
        v[mask] = bill_A[members].sum() - sim.bill(flow, tariff)
    # exact Shapley
    shap = np.zeros(N)
    for i in range(N):
        for mask in range(1 << N):
            if mask >> i & 1:
                continue
            s = bin(mask).count("1")
            w = factorial(s) * factorial(N - s - 1) / factorial(N)
            shap[i] += w * (v[mask | 1 << i] - v[mask])
    total = v[(1 << N) - 1]
    alloc = {
        "equal": np.full(N, total / N),
        "prop_consumption": total * load.sum(axis=1) / load.sum(),
        "prop_bill": total * bill_A / bill_A.sum(),
        "shapley": shap,
    }
    rows = []
    for rule, a in alloc.items():
        for i in range(N):
            rows.append(dict(group=gi, tariff=tname, pv_scale=pv_scale, home=int(idx[i]), rule=rule,
                             allocation=a[i], own_battery_saving=bill_A[i] - bill_B[i],
                             annual_load=load[i].sum(), annual_pv=pv[i].sum(), total_saving=total,
                             stand_alone_bill=bill_A[i]))
    return rows


def main():
    t0 = time.time()
    d = sim.load_headline()
    groups = sim.make_partitions(d, N, 1, seed=1000 + N)[0][:N_GROUPS]      # non-overlapping groups
    jobs = [(gi, idx, t, pv) for gi, idx in enumerate(groups) for t in TARIFFS for pv in cfg.PV_SCALES]
    res = Parallel(n_jobs=7, verbose=5)(delayed(one)(*j) for j in jobs)
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(sim.ROOT / "results" / "fairness.csv", index=False)
    print(f"done {len(df)} rows ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
