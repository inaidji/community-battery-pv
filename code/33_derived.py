"""Derived tables: every number used in the paper that is not in 30_tables.py output.
Reads results/v2 and writes CSVs to results/v2_tables/. No number in the paper may be computed elsewhere.

  bills.csv            mean annual group bill by scenario (n=25, 5 kWh/home)
  decomposition.csv    savings, billing / pooling effects, netting, storage-only (B_agg and C vs A_agg), netting share
  economics2.csv       simple payback on two cost bases, NPV break-even ratio (rule, lp; all tariffs)
  controllers_all.csv  all controllers incl. reactive day-ahead, capped rules, 5 tie-break draws (ToU, n=25)
  econ_controllers.csv payback by controller (ToU)
  seeds.csv            spread of the random tie-break over 5 independent draws
  batt.csv             battery C-rate / efficiency sensitivity
"""
import numpy as np
import pandas as pd

import config as cfg
import sim
from analysis_lib import load, load_controllers, wide, paired, est, pct_change, diff, KEYS

OUT = sim.ROOT / "results" / "v2_tables"
PH = cfg.BATT_COST_HOME_AUD_PER_KWH
PC_GROSS = cfg.BATT_COST_COMMUNITY_AUD_PER_KWH
PC_NET = cfg.BATT_COST_COMMUNITY_NET_AUD_PER_KWH


def base(main):
    m = main[(main.n == 25) & (main.cap == 5.0)]
    return m


def bills(main):
    m = base(main)
    rows = []
    for scn in sorted(m.scenario.unique()):
        r = paired(m, "bill_aud", lambda w, s=scn: w[s])
        r["scenario"] = scn
        rows.append(r)
    out = pd.concat(rows)
    out.to_csv(OUT / "bills.csv", index=False)
    return out


def decomposition(main):
    """All quantities are means over partitions of within-partition means (analysis_lib.est).
    netting_share = mean netting saving / mean community saving (a ratio of means; the only definition used)."""
    m = base(main)
    rows = []
    for mode in ("rule", "lp"):
        q = {
            "sav_B": diff("A", f"B_{mode}"), "sav_Bagg": diff("A", f"Bagg_{mode}"), "sav_C": diff("A", f"C_{mode}"),
            "billing_effect": diff(f"B_{mode}", f"Bagg_{mode}"), "pooling_effect": diff(f"Bagg_{mode}", f"C_{mode}"),
            "netting": diff("A", "A_agg"),
            "storage_Bagg": diff("A_agg", f"Bagg_{mode}"), "storage_C": diff("A_agg", f"C_{mode}"),
        }
        for k, fn in q.items():
            if k == "sav_Bagg" or k.endswith("Bagg") or k in ("billing_effect", "pooling_effect"):
                mm = m[m.tariff != "tou_exportlimit"]
            else:
                mm = m
            r = paired(mm, "bill_aud", fn)
            r["quantity"], r["mode"] = k, mode
            rows.append(r)
    d = pd.concat(rows)
    piv = d.pivot_table(index=["tariff", "pv", "mode"], columns="quantity", values="mean").reset_index()
    piv["netting_share_pct"] = 100 * piv["netting"] / piv["sav_C"]
    piv["billing_share_of_gap_pct"] = 100 * piv["billing_effect"] / (piv["sav_C"] - piv["sav_B"])
    piv.to_csv(OUT / "decomposition.csv", index=False)
    return piv


def _econ_row(sB, sC, cycB, cycC, cap_tot):
    r = cfg.DISCOUNT_RATE
    ann = lambda y: (1 - (1 + r) ** -y) / r
    life = lambda c: min(cfg.BATT_LIFE_MAX_YEARS, cfg.BATT_CYCLE_LIFE / max(c, 1e-9))
    KB = cap_tot * PH
    return dict(sav_B=sB, sav_C=sC, payback_B=KB / sB, payback_C_gross=cap_tot * PC_GROSS / sC,
                payback_C_net=cap_tot * PC_NET / sC,
                breakeven_ratio=(sC * ann(life(cycC)) - sB * ann(life(cycB)) + KB) / KB,
                ratio_gross=PC_GROSS / PH, ratio_net=PC_NET / PH)


def economics2(main):
    m = base(main)
    wb, wc = wide(m, "bill_aud"), wide(m, "cycles")
    rows = []
    for (t, pv), g in wb.groupby(["tariff", "pv"]):
        gc = wc[(wc.tariff == t) & (wc.pv == pv)]
        for mode in ("rule", "lp"):
            sB = est((g["A"] - g[f"B_{mode}"]).to_numpy(), g["part"].to_numpy())[0]
            sC = est((g["A"] - g[f"C_{mode}"]).to_numpy(), g["part"].to_numpy())[0]
            cB = est(gc[f"B_{mode}"].to_numpy(), gc["part"].to_numpy())[0]
            cC = est(gc[f"C_{mode}"].to_numpy(), gc["part"].to_numpy())[0]
            rows.append(dict(tariff=t, pv=pv, mode=mode, **_econ_row(sB, sC, cB, cC, 125.0)))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "economics2.csv", index=False)
    return out


def controllers_all(ct):
    rows = []
    scns = sorted(s for s in ct.scenario.unique() if s.startswith(("B_", "C_")))
    for scn in scns:
        who, c = scn.split("_", 1)
        for qty, metric, fn in (("d_peak_pct", "peak_import_kw", pct_change(scn, "A")),
                                ("saving", "bill_aud", diff("A", scn)),
                                ("cycles", "cycles", lambda w, s=scn: w[s]),
                                ("d_p99_pct", "p99_import_kw", pct_change(scn, "A")),
                                ("peak_export_kw", "peak_export_kw", lambda w, s=scn: w[s]),
                                ("export_kwh", "export_kwh", lambda w, s=scn: w[s]),
                                ("coincidence", "chg_coincidence", lambda w, s=scn: w[s]),
                                ("chg_peak_kw", "chg_peak_kw", lambda w, s=scn: w[s])):
            r = paired(ct, metric, fn)
            r["who"], r["ctrl"], r["quantity"] = who, c, qty
            rows.append(r)
    out = pd.concat(rows)
    out.to_csv(OUT / "controllers_all.csv", index=False)
    p = out.pivot_table(index=["pv", "ctrl"], columns=["who", "quantity"], values="mean")
    e = []
    for (pv, c), r in p.iterrows():
        if ("B", "saving") in r.index and not np.isnan(r[("B", "saving")]):
            e.append(dict(pv=pv, ctrl=c, **_econ_row(r[("B", "saving")], r[("C", "saving")],
                                                      r[("B", "cycles")], r[("C", "cycles")], 125.0)))
    pd.DataFrame(e).to_csv(OUT / "econ_controllers.csv", index=False)
    s = out[out.ctrl.str.startswith("lp_random") & out.quantity.isin(["d_peak_pct", "saving"])]
    sd = s.groupby(["pv", "who", "quantity"])["mean"].agg(["mean", "std", "min", "max", "count"]).reset_index()
    sd.to_csv(OUT / "seeds.csv", index=False)
    return out, sd


def tails(main, ct):
    """Upper tail of import and export quantities that the annual maximum does not show (n=25, 5 kWh/home):
    99th percentile of half-hourly import (change vs no battery), peak export (kW), exported energy (kWh),
    and the share of groups whose annual peak import falls before / inside / after the 15:00-21:00 window."""
    m = base(main)
    rows = []
    for scn in ("B_rule", "B_lp", "C_rule", "C_lp"):
        r = paired(m, "p99_import_kw", pct_change(scn, "A")); r["scenario"], r["quantity"] = scn, "d_p99_pct"; rows.append(r)
    for scn in ("A", "B_rule", "B_lp", "C_rule", "C_lp"):
        for q in ("peak_export_kw", "export_kwh", "peak_import_kw"):
            r = paired(m, q, lambda w, s=scn: w[s]); r["scenario"], r["quantity"] = scn, q; rows.append(r)
    out = pd.concat(rows)
    out.to_csv(OUT / "tails.csv", index=False)
    hrs = []
    d = pd.concat([ct, m[(m.tariff == "tou") & m.scenario.isin(["A"])]], ignore_index=True)
    for (pv, scn), g in d.groupby(["pv", "scenario"]):
        g = g.drop_duplicates(["part", "group"])
        h = g.peak_import_hour.to_numpy()
        hrs.append(dict(pv=pv, scenario=scn, groups=len(h), before_window_pct=100 * np.mean(h < cfg.TOU_PEAK_START_H),
                        in_window_pct=100 * np.mean((h >= cfg.TOU_PEAK_START_H) & (h < cfg.TOU_PEAK_END_H)),
                        after_window_pct=100 * np.mean(h >= cfg.TOU_PEAK_END_H), modal_hour=float(pd.Series(h).mode().iloc[0]),
                        modal_share_pct=100 * float((h == pd.Series(h).mode().iloc[0]).mean())))
    hh = pd.DataFrame(hrs)
    hh.to_csv(OUT / "peak_hours.csv", index=False)
    return out, hh


def batt(bt, main):
    b0 = base(main)
    b0 = b0[b0.tariff == "tou"]
    d = pd.concat([bt, b0], ignore_index=True)
    rows = []
    for scn in ("B_rule", "B_lp", "C_rule", "C_lp", "C_lp_minpeak"):
        if scn not in set(d.scenario):
            continue
        dd = d[d.batt.isin(d[d.scenario == scn].batt.unique())] if scn == "C_lp_minpeak" else d
        for qty, metric, fn in (("d_peak_pct", "peak_import_kw", pct_change(scn, "A")), ("saving", "bill_aud", diff("A", scn))):
            if scn == "C_lp_minpeak":
                dd2 = dd[(dd.pv == 1.0) & (dd.batt != "base")]
            else:
                dd2 = dd
            r = paired(dd2, metric, fn)
            r["scenario"], r["quantity"] = scn, qty
            rows.append(r)
    out = pd.concat(rows)
    out.to_csv(OUT / "batt.csv", index=False)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_rows", 300, "display.max_columns", 40)
    main = load("main_")
    b = bills(main)
    print(b[(b.tariff.isin(["tou", "tou_demand24"])) & (b.pv == 1.0)].pivot_table(index="scenario", columns="tariff", values="mean").round(0))
    print(decomposition(main).round(1).to_string(index=False))
    print(economics2(main).round(2).to_string(index=False))
    ct = load_controllers()
    out, sd = controllers_all(ct)
    tl, hh = tails(main, ct)
    print(tl[(tl.tariff.isin(["tou", "tou_demand24"]))].pivot_table(index=["pv", "scenario"], columns=["tariff", "quantity"], values="mean").round(1).to_string())
    print(hh.round(0).to_string(index=False))
    print(out[out.quantity.isin(["d_peak_pct", "saving"])].pivot_table(index=["pv", "ctrl"], columns=["who", "quantity"], values="mean").round(1).to_string())
    print(sd.round(2).to_string(index=False))
    print(pd.read_csv(OUT / "econ_controllers.csv").round(2).to_string(index=False))
    print(batt(load("batt_"), main).pivot_table(index=["pv", "batt"], columns=["scenario", "quantity"], values="mean").round(1).to_string())
