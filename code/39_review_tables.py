"""Tables for the experiments added after the independent review (23_run_review.py). Writes results/v2_tables/rv_*.csv.

  rv_controllers.csv  controller study with tail metrics (n=25): peak, p99, p99.9, hours at/above 90 % and 99.9 % of
                      the no-battery peak, saving, cycles; share of groups by position of the annual peak
  rv_ctrl10.csv       the same for groups of 10 (like-for-like with the second dataset)
  rv_rn.csv           revenue-neutral demand-charge tariffs: bills, savings, decomposition, netting share, peak change
  rv_sweep.csv        ToU shape sweep and free-midday variant
  rv_coord.csv        coordinated individual batteries with individual bills
"""
import numpy as np
import pandas as pd

import config as cfg
import sim
from analysis_lib import load, paired, pct_change, diff

OUT = sim.ROOT / "results" / "v2_tables"


def ctrl_table(d, name):
    rows = []
    for scn in sorted(s for s in d.scenario.unique() if s.startswith(("B_", "C_"))):
        who, c = scn.split("_", 1)
        Q = [("d_peak_pct", "peak_import_kw", pct_change(scn, "A")), ("d_p99_pct", "p99_import_kw", pct_change(scn, "A")),
             ("d_p999_pct", "p999_import_kw", pct_change(scn, "A")), ("saving", "bill_aud", diff("A", scn)),
             ("h_ge90", "h_ge90", lambda w, s=scn: w[s]), ("h_ge999", "h_ge999", lambda w, s=scn: w[s]),
             ("cycles", "cycles", lambda w, s=scn: w[s]), ("peak_export_kw", "peak_export_kw", lambda w, s=scn: w[s])]
        for qty, metric, fn in Q:
            r = paired(d, metric, fn)
            r["who"], r["ctrl"], r["quantity"] = who, c, qty
            rows.append(r)
        g = d[d.scenario == scn]
        for pv, gg in g.groupby("pv"):
            h = gg.peak_import_hour.to_numpy()
            for q, v in (("pk_before_pct", 100 * np.mean(h < cfg.TOU_PEAK_START_H)), ("pk_after_pct", 100 * np.mean(h >= cfg.TOU_PEAK_END_H)),
                         ("pk_in_pct", 100 * np.mean((h >= cfg.TOU_PEAK_START_H) & (h < cfg.TOU_PEAK_END_H)))):
                rows.append(pd.DataFrame([dict(pv=pv, mean=v, lo=np.nan, hi=np.nan, who=who, ctrl=c, quantity=q)]))
    ga = d[d.scenario == "A"]
    for pv, gg in ga.groupby("pv"):
        h = gg.peak_import_hour.to_numpy()
        for q, v in (("pk_before_pct", 100 * np.mean(h < cfg.TOU_PEAK_START_H)), ("pk_after_pct", 100 * np.mean(h >= cfg.TOU_PEAK_END_H)),
                     ("pk_in_pct", 100 * np.mean((h >= cfg.TOU_PEAK_START_H) & (h < cfg.TOU_PEAK_END_H)))):
            rows.append(pd.DataFrame([dict(pv=pv, mean=v, lo=np.nan, hi=np.nan, who="A", ctrl="none", quantity=q)]))
    for qty, metric in (("h_ge90", "h_ge90"), ("h_ge999", "h_ge999"), ("peak_export_kw", "peak_export_kw"), ("peak_import_kw", "peak_import_kw"),
                        ("p99_import_kw", "p99_import_kw")):
        r = paired(d, metric, lambda w: w["A"])
        r["who"], r["ctrl"], r["quantity"] = "A", "none", qty
        rows.append(r)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT / name, index=False)
    return out


def rn_table(d):
    rows = []
    for mode in ("rule", "lp"):
        for k, fn in {"sav_B": diff("A", f"B_{mode}"), "sav_Bagg": diff("A", f"Bagg_{mode}"), "sav_C": diff("A", f"C_{mode}"),
                      "billing_effect": diff(f"B_{mode}", f"Bagg_{mode}"), "pooling_effect": diff(f"Bagg_{mode}", f"C_{mode}"),
                      "netting": diff("A", "A_agg"), "bill_A": lambda w: w["A"], "bill_C": lambda w, m=mode: w[f"C_{m}"]}.items():
            r = paired(d, "bill_aud", fn); r["quantity"], r["mode"] = k, mode; rows.append(r)
        for who in ("B", "C"):
            r = paired(d, "peak_import_kw", pct_change(f"{who}_{mode}", "A")); r["quantity"], r["mode"] = f"d_peak_{who}", mode; rows.append(r)
            r = paired(d, "p99_import_kw", pct_change(f"{who}_{mode}", "A")); r["quantity"], r["mode"] = f"d_p99_{who}", mode; rows.append(r)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT / "rv_rn.csv", index=False)
    return out


def sweep_table(d):
    rows = []
    for scn in ("B_rule", "B_lp", "C_rule", "C_lp", "C_lp_minpeak"):
        for qty, metric, fn in (("d_peak_pct", "peak_import_kw", pct_change(scn, "A")), ("d_p99_pct", "p99_import_kw", pct_change(scn, "A")),
                                ("saving", "bill_aud", diff("A", scn))):
            r = paired(d, metric, fn); r["scenario"], r["quantity"] = scn, qty; rows.append(r)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT / "rv_sweep.csv", index=False)
    return out


def coord_table(d):
    rows = []
    for scn in ("B_lp", "Bcoord_lp"):
        for qty, metric, fn in (("d_peak_pct", "peak_import_kw", pct_change(scn, "A")), ("d_p99_pct", "p99_import_kw", pct_change(scn, "A")),
                                ("saving", "bill_aud", diff("A", scn)), ("h_ge90", "h_ge90", lambda w, s=scn: w[s])):
            r = paired(d, metric, fn); r["scenario"], r["quantity"] = scn, qty; rows.append(r)
    r = paired(d, "h_ge90", lambda w: w["A"]); r["scenario"], r["quantity"] = "A", "h_ge90"; rows.append(r)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT / "rv_coord.csv", index=False)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 260, "display.max_rows", 400, "display.max_columns", 40)
    t = ctrl_table(load("rv_ctrl_"), "rv_controllers.csv")
    print(t.pivot_table(index=["pv", "who", "ctrl"], columns="quantity", values="mean").round(1).to_string())
    t10 = ctrl_table(load("ctrl10_"), "rv_ctrl10.csv")
    print(t10[t10.quantity.isin(["d_peak_pct", "d_p99_pct", "saving"])].pivot_table(index=["who", "ctrl"], columns="quantity", values="mean").round(1).to_string())
    rn = rn_table(load("rv_rn_"))
    print(rn.pivot_table(index=["tariff", "pv", "mode"], columns="quantity", values="mean").round(0).to_string())
    sw = sweep_table(load("rv_sweep_"))
    print(sw.pivot_table(index=["tariff", "tkw", "pv"], columns=["quantity", "scenario"], values="mean").round(1).to_string())
    print(coord_table(load("rv_coord_")).pivot_table(index="scenario", columns="quantity", values="mean").round(1).to_string())
