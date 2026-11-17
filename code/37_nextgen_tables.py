"""Tables for the second dataset (NextGen ACT 2018; see 40_prepare_nextgen.py, 22_run_nextgen.py).
Same estimators as for the Ausgrid data (analysis_lib). Two versions of the data are tabulated:
  ngs_*  MAIN: no imputed day (52 homes, 277 days measured for all homes). Bills, savings and reverse-flow hours
         are totals over the retained days; columns *_annual scale them by 365 / retained days.
  ng_*   sensitivity: 99 homes, full year, missing days imputed from the same weekday.
Writes results/v2_tables/{ngs,ng}_{main,controllers,stats}.csv.

  ng_main.csv         peak change, reverse-flow hours, savings and their decomposition by tariff and group size
  ng_controllers.csv  controller study at ToU, n = 10
  ng_stats.csv        paired exact Wilcoxon tests within partitions (max p over partitions), Holm-corrected
"""
import numpy as np
import pandas as pd
from scipy import stats

import sim
from analysis_lib import load, wide, paired, est, pct_change, diff
import importlib
S34 = importlib.import_module("34_stats")

OUT = sim.ROOT / "results" / "v2_tables"


def main_table(m, tag, scale):
    rows = []
    q = [("d_peak_pct", "peak_import_kw", pct_change, ("B_rule", "A")), ("d_peak_pct", "peak_import_kw", pct_change, ("B_lp", "A")),
         ("d_peak_pct", "peak_import_kw", pct_change, ("C_rule", "A")), ("d_peak_pct", "peak_import_kw", pct_change, ("C_lp", "A"))]
    for name, metric, f, (a, b) in q:
        r = paired(m, metric, f(a, b))
        r["quantity"], r["scenario"] = name, a
        rows.append(r)
    for scn in ("B_rule", "B_lp", "C_rule", "C_lp"):
        r = paired(m, "p99_import_kw", pct_change(scn, "A"))
        r["quantity"], r["scenario"] = "d_p99_pct", scn
        rows.append(r)
    for scn in ("A", "B_rule", "B_lp", "C_rule", "C_lp"):
        for q in ("peak_import_kw", "peak_export_kw"):
            r = paired(m, q, lambda w, s=scn: w[s])
            r["quantity"], r["scenario"] = q, scn
            rows.append(r)
        r = paired(m, "reverse_flow_h", lambda w, s=scn: w[s])
        r["quantity"], r["scenario"] = "reverse_flow_h", scn
        rows.append(r)
        r = paired(m, "bill_aud", lambda w, s=scn: w[s])
        r["quantity"], r["scenario"] = "bill_aud", scn
        rows.append(r)
    for mode in ("rule", "lp"):
        for k, fn in {"sav_B": diff("A", f"B_{mode}"), "sav_Bagg": diff("A", f"Bagg_{mode}"), "sav_C": diff("A", f"C_{mode}"),
                      "billing_effect": diff(f"B_{mode}", f"Bagg_{mode}"), "pooling_effect": diff(f"Bagg_{mode}", f"C_{mode}"),
                      "netting": diff("A", "A_agg"), "storage_C": diff("A_agg", f"C_{mode}")}.items():
            r = paired(m, "bill_aud", fn)
            r["quantity"], r["scenario"] = k, mode
            rows.append(r)
    out = pd.concat(rows)
    out["mean_annual"] = np.where(out.quantity.isin(["d_peak_pct", "d_p99_pct", "peak_import_kw", "peak_export_kw"]), out["mean"], out["mean"] * scale)
    out.to_csv(OUT / f"{tag}_main.csv", index=False)
    return out


def ctrl_table(c, tag, scale):
    rows = []
    for scn in sorted(s for s in c.scenario.unique() if s.startswith(("B_", "C_"))):
        who, ct = scn.split("_", 1)
        for qty, metric, fn in (("d_peak_pct", "peak_import_kw", pct_change(scn, "A")), ("saving", "bill_aud", diff("A", scn)),
                                ("d_p99_pct", "p99_import_kw", pct_change(scn, "A")), ("cycles", "cycles", lambda w, s=scn: w[s])):
            r = paired(c, metric, fn)
            r["who"], r["ctrl"], r["quantity"] = who, ct, qty
            rows.append(r)
    out = pd.concat(rows)
    out["mean_annual"] = np.where(out.quantity.isin(["d_peak_pct", "d_p99_pct", "peak_import_kw", "peak_export_kw"]), out["mean"], out["mean"] * scale)
    out.to_csv(OUT / f"{tag}_controllers.csv", index=False)
    return out


def tests(m, c, tag):
    Q = "p99_import_kw"
    T = [("Cpk|annual maximum: community, solver default, vs no battery [pp]", m, "tou", "peak_import_kw", pct_change("C_lp", "A")),
         ("Bpk|annual maximum: individual, solver default, vs no battery [pp]", m, "tou", "peak_import_kw", pct_change("B_lp", "A")),
         ("CvB|annual maximum: community vs individual, solver default [pp]", m, "tou", "peak_import_kw",
          lambda w: 100 * (w["C_lp"] - w["B_lp"]) / w["A"]),
         ("Cp99|99th percentile: community, solver default, vs no battery [pp]", m, "tou", Q, pct_change("C_lp", "A")),
         ("minp99|99th percentile: community, peak-minimising, vs no battery [pp]", c, "tou", Q, pct_change("C_lp_minpeak", "A")),
         ("sprp99|99th percentile: community, spread top-up, vs no battery [pp]", c, "tou", Q, pct_change("C_spread", "A")),
         ("dap99|99th percentile: community, day-ahead (persistence), vs no battery [pp]", c, "tou", Q, pct_change("C_mpc_react_persist", "A")),
         ("dem24|annual maximum: community vs no battery, 24 h demand charge [pp]", m, "tou_demand24", "peak_import_kw", pct_change("C_lp", "A")),
         ("dem24p99|99th percentile: community vs no battery, 24 h demand charge [pp]", m, "tou_demand24", Q, pct_change("C_lp", "A")),
         ("rev|reverse-flow hours: community vs individual [h]", m, "tou", "reverse_flow_h", diff("C_lp", "B_lp")),
         ("billCvB|bill: community vs individual, solver default [A$]", m, "tou", "bill_aud", diff("C_lp", "B_lp")),
         ("billing|bill: billing effect (B_agg vs B) [A$]", m, "tou", "bill_aud", diff("Bagg_lp", "B_lp")),
         ("pooling|bill: pooling effect (C vs B_agg) [A$]", m, "tou", "bill_aud", diff("C_lp", "Bagg_lp"))]
    rows = []
    for name, d, tar, metric, fn in T:
        d = d[(d.tariff == tar) & (d.n == 10)]
        w = wide(d, metric)
        w["_v"] = fn(w)
        ps = []
        for _, g in w.groupby("part"):
            v = g["_v"].dropna().to_numpy()
            ps.append(S34.wilcoxon_p(v))
        mean, lo, hi, G, K = est(w["_v"].to_numpy(), w["part"].to_numpy())
        rows.append(dict(id=name.split("|")[0], contrast=name.split("|")[1], n=10, mean_diff=mean, lo=lo, hi=hi, p_max=max(ps), p_min=min(ps), partitions=K, groups=G))
    out = pd.DataFrame(rows)
    o = np.argsort(out.p_max.to_numpy())
    adj = np.maximum.accumulate((len(out) - np.arange(len(out))) * out.p_max.to_numpy()[o]).clip(max=1.0)
    out.loc[out.index[o], "p_holm"] = adj
    out.to_csv(OUT / f"{tag}_stats.csv", index=False)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_rows", 300, "display.max_columns", 40)
    for tag, year in (("ngs", "nextgen-2018s"), ("ng", "nextgen-2018")):
        n_days = len(sim.load_headline(year)["dates"])
        scale = 365.0 / n_days
        print(f"\n===== {tag}: {year}, {n_days} days, annual scale x{scale:.4f} =====")
        m, c = load(f"{tag}_main_"), load(f"{tag}_ctrl_")
        t = main_table(m, tag, scale)
        print(t.pivot_table(index=["quantity", "scenario"], columns=["tariff", "n"], values="mean_annual").round(1).to_string())
        pk = t[t.quantity == "d_peak_pct"]
        print(pk[["tariff", "n", "scenario", "mean", "lo", "hi", "groups", "partitions"]].round(1).to_string(index=False))
        ct = ctrl_table(c, tag, scale)
        ct["v"] = ct.apply(lambda r: f"{r['mean']:.1f} [{r.lo:.1f}, {r.hi:.1f}]", axis=1)
        print(ct.pivot_table(index="ctrl", columns=["quantity", "who"], values="v", aggfunc="first").to_string())
        print(tests(m, c, tag).round(4).to_string(index=False))
