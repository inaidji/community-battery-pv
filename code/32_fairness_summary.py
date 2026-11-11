"""Fairness summary from results/fairness.csv -> fairness_summary.csv"""
import numpy as np
import pandas as pd
import sim
R=sim.ROOT/"results"
pd.set_option("display.width",250)
# ---------------- fairness ----------------
f = pd.read_csv(R / "fairness.csv")
f["better_off"] = f.allocation >= f.own_battery_saving
f["per_kwh_load"] = f.allocation / f.annual_load


def gini(x):
    x = np.sort(np.asarray(x)); n = len(x)
    return (2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()) if x.sum() > 0 else np.nan


rows = []
for (t, pv, rule, g), d in f.groupby(["tariff", "pv_scale", "rule", "group"]):
    rows.append(dict(tariff=t, pv_scale=pv, rule=rule, group=g, share_better_off=d.better_off.mean(),
                     min_alloc=d.allocation.min(), max_alloc=d.allocation.max(), gini=gini(d.allocation),
                     corr_load=np.corrcoef(d.allocation, d.annual_load)[0, 1],
                     corr_pv=np.corrcoef(d.allocation, d.annual_pv)[0, 1],
                     total=d.total_saving.iloc[0], own_total=d.own_battery_saving.sum(),
                     neg_share=(d.allocation < 0).mean()))
fs = pd.DataFrame(rows).groupby(["tariff", "pv_scale", "rule"]).mean(numeric_only=True).drop(columns="group").round(3).reset_index()
fs.to_csv(R / "fairness_summary.csv", index=False)
print("\n=== FAIRNESS (n=10 homes, 5 groups, rule-based). share_better_off = homes whose allocation >= own-battery saving ===")
print(fs.to_string(index=False))
