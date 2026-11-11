"""Compute every table of the revised paper from results/v2 (prints them and saves CSVs to results/v2_tables)."""
import numpy as np
import pandas as pd

import config as cfg
import sim
from analysis_lib import load, wide, paired, est, pct_change, diff, fmt_ci, KEYS

OUT = sim.ROOT / "results" / "v2_tables"
OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 260, "display.max_columns", 60, "display.max_rows", 500)
TARIFFS = list(cfg.TARIFFS)


def save(df, name):
    df.to_csv(OUT / f"{name}.csv", index=False)
    return df


def sel(df, **kw):
    for k, v in kw.items():
        df = df[df[k] == v]
    return df


def table_peak(main):
    """Peak-import change vs no battery (%) per tariff/PV: B and C for rule and cost-minimising (lp)."""
    m = sel(main, n=25, cap=5.0)
    rows = []
    for scn in ("B_rule", "B_lp", "C_rule", "C_lp"):
        r = paired(m, "peak_import_kw", pct_change(scn, "A"))
        r["scenario"] = scn
        rows.append(r)
    out = pd.concat(rows)
    base = paired(m, "peak_import_kw", lambda w: w["A"])
    base["scenario"] = "A_peak_kw"
    return save(pd.concat([out, base]), "peak_change")


def table_size(main):
    """Peak change of C_lp / B_lp vs battery size and group size (tou)."""
    m = sel(main, tariff="tou")
    rows = []
    for scn in ("B_lp", "C_lp", "C_rule"):
        r = paired(m, "peak_import_kw", pct_change(scn, "A"))
        r["scenario"] = scn
        rows.append(r)
    return save(pd.concat(rows), "peak_change_by_size")


def table_reverse(main):
    m = sel(main, n=25, cap=5.0)
    rows = []
    for scn in ("A", "B_rule", "B_lp", "C_rule", "C_lp"):
        r = paired(m, "reverse_flow_h", lambda w, s=scn: w[s])
        r["scenario"] = scn
        rows.append(r)
    return save(pd.concat(rows), "reverse_flow")


def table_savings(main):
    """Annual saving decomposition per group (A$): individual, billing effect, pooling effect, community."""
    m = sel(main, n=25, cap=5.0)
    rows = []
    for mode in ("rule", "lp"):
        specs = {
            "sav_B": lambda w, c=mode: w["A"] - w[f"B_{c}"],
            "sav_Bagg": lambda w, c=mode: w["A"] - w[f"Bagg_{c}"],
            "sav_C": lambda w, c=mode: w["A"] - w[f"C_{c}"],
            "billing_effect": lambda w, c=mode: w[f"B_{c}"] - w[f"Bagg_{c}"],        # B -> Bagg
            "pooling_effect": lambda w, c=mode: w[f"Bagg_{c}"] - w[f"C_{c}"],        # Bagg -> C
            "netting_A": lambda w: w["A"] - w["A_agg"],
            "storage_only_C": lambda w, c=mode: w["A_agg"] - w[f"C_{c}"],
        }
        for k, fn in specs.items():
            r = paired(m, "bill_aud", fn)
            r["quantity"], r["mode"] = k, mode
            rows.append(r)
    return save(pd.concat(rows), "savings")


def table_ctrl(ctrl):
    """Controller comparison (tou, n=25, 5 kWh): peak change, saving, coincidence, cycles."""
    rows = []
    cts = sorted({s.split("_", 1)[1] for s in ctrl.scenario.unique() if s.startswith("C_")})
    for who in ("B", "C"):
        for c in cts:
            scn = f"{who}_{c}"
            if scn not in set(ctrl.scenario):
                continue
            for qty, metric, fn in (
                ("d_peak_pct", "peak_import_kw", pct_change(scn, "A")),
                ("saving", "bill_aud", diff("A", scn)),
                ("peak_kw", "peak_import_kw", lambda w, s=scn: w[s]),
                ("cycles", "cycles", lambda w, s=scn: w[s]),
                ("coincidence", "chg_coincidence", lambda w, s=scn: w[s]),
                ("chg_peak_kw", "chg_peak_kw", lambda w, s=scn: w[s]),
                ("exports_kwh", "export_kwh", lambda w, s=scn: w[s]),
            ):
                r = paired(ctrl, metric, fn)
                r["who"], r["ctrl"], r["quantity"] = who, c, qty
                rows.append(r)
    return save(pd.concat(rows), "controllers")


def table_years(years, main):
    y = pd.concat([years, sel(main, n=25, cap=5.0).assign(year="2012-13")])
    rows = []
    for scn in ("B_lp", "C_lp"):
        r = paired(y, "peak_import_kw", pct_change(scn, "A"))
        r["scenario"], r["quantity"] = scn, "d_peak_pct"
        rows.append(r)
    for scn in ("A", "B_lp", "C_lp"):
        r = paired(y, "reverse_flow_h", lambda w, s=scn: w[s])
        r["scenario"], r["quantity"] = scn, "reverse_flow_h"
        rows.append(r)
    r = paired(y, "bill_aud", lambda w: 100 * (w["A"] - w["A_agg"]) / (w["A"] - w["C_rule"]))
    r["scenario"], r["quantity"] = "C_rule", "netting_share_pct"
    rows.append(r)
    r = paired(y, "peak_import_kw", lambda w: w["A"])
    r["scenario"], r["quantity"] = "A", "peak_kw"
    rows.append(r)
    return save(pd.concat(rows), "years")


def table_sens(sens, main):
    base_map = {"flat": ("flat", "base"), "tou": ("tou", "base"), "tou_demand": ("tou_demand", "base"),
                "tou_demand24": ("tou_demand24", "base"), "tou_exportlimit": ("tou_exportlimit", "base")}
    b = sel(main, n=25, cap=5.0)
    s = pd.concat([sens, b], ignore_index=True)
    rows = []
    for scn in ("B_lp", "C_lp"):
        r = paired(s, "peak_import_kw", pct_change(scn, "A"))
        r["scenario"], r["quantity"] = scn, "d_peak_pct"
        rows.append(r)
    for mode in ("rule", "lp"):
        for nm, fn in (("sav_B", lambda w, c=mode: w["A"] - w[f"B_{c}"]),
                       ("sav_C", lambda w, c=mode: w["A"] - w[f"C_{c}"]),
                       ("sav_C_storage", lambda w, c=mode: w["A_agg"] - w[f"C_{c}"])):
            r = paired(s, "bill_aud", fn)
            r["scenario"], r["quantity"] = mode, nm
            rows.append(r)
    return save(pd.concat(rows), "sensitivity")


def table_clmix(cl, mix, main):
    b = sel(main, n=25, cap=5.0)
    b = b[b.tariff.isin(["flat", "tou"])].assign(loadvar="gc+cl")
    rows = []
    for tag, df in (("base", b), ("no_controlled_load", cl), ("third_without_pv", mix)):
        for scn in ("B_lp", "C_lp"):
            r = paired(df, "peak_import_kw", pct_change(scn, "A"))
            r["scenario"], r["quantity"], r["variant"] = scn, "d_peak_pct", tag
            rows.append(r)
        for mode in ("rule",):
            for nm, fn in (("sav_B", lambda w, c=mode: w["A"] - w[f"B_{c}"]),
                           ("sav_C", lambda w, c=mode: w["A"] - w[f"C_{c}"]),
                           ("sav_C_storage", lambda w, c=mode: w["A_agg"] - w[f"C_{c}"])):
                r = paired(df, "bill_aud", fn)
                r["scenario"], r["quantity"], r["variant"] = mode, nm, tag
                rows.append(r)
    return save(pd.concat(rows), "clmix")


def table_econ(main):
    """Payback (years) from MEAN annual savings, and break-even community cost ratio (price ratio at which
    community and individual NPV are equal). Home battery A$/kWh and community A$/kWh from config."""
    m = sel(main, cap=5.0)
    w = wide(m, "bill_aud")
    wc = wide(m, "cycles")
    keys = [k for k in KEYS if k in w.columns]
    ph, pc = cfg.BATT_COST_HOME_AUD_PER_KWH, cfg.BATT_COST_COMMUNITY_AUD_PER_KWH
    r_disc = cfg.DISCOUNT_RATE
    ann = lambda y: (1 - (1 + r_disc) ** -y) / r_disc
    life = lambda cyc: np.minimum(cfg.BATT_LIFE_MAX_YEARS, cfg.BATT_CYCLE_LIFE / max(cyc, 1e-9))
    rows = []
    for key, g in w.groupby(keys, dropna=False):
        gc = wc.loc[g.index]
        n = key[keys.index("n")]
        cap_tot = 5.0 * n
        for mode in ("rule", "lp"):
            sB = (g["A"] - g[f"B_{mode}"]).mean()
            sC = (g["A"] - g[f"C_{mode}"]).mean()
            lB, lC = life(gc[f"B_{mode}"].mean()), life(gc[f"C_{mode}"].mean())
            capexB, capexC = cap_tot * ph, cap_tot * pc
            rho_match = (sC * ann(lC) - sB * ann(lB) + capexB) / capexB   # community price/home price at equal NPV
            rows.append(dict(zip(keys, key), mode=mode, sav_B=sB, sav_C=sC, payback_B=capexB / sB if sB > 0 else np.inf,
                             payback_C_home_price=capexB / sC if sC > 0 else np.inf,
                             payback_C_arena=capexC / sC if sC > 0 else np.inf,
                             npv_B=sB * ann(lB) - capexB, npv_C_arena=sC * ann(lC) - capexC,
                             rho_match=rho_match, actual_ratio=pc / ph))
    return save(pd.DataFrame(rows), "economics")


if __name__ == "__main__":
    main = load("main_")
    print("main rows", len(main))
    print("\n=== PEAK CHANGE (%), n=25, 5 kWh/home; mean [CI over one non-overlapping partition] ===")
    pk = table_peak(main)
    show = pk[(pk.scenario != "A_peak_kw")].copy()
    show["ci"] = show.apply(fmt_ci, axis=1)
    print(show.pivot_table(index=["tariff", "pv"], columns="scenario", values="ci", aggfunc="first").to_string())
    print("no-battery peak kW:\n", pk[pk.scenario == "A_peak_kw"][["tariff", "pv", "mean"]].round(1).to_string(index=False))
    print("\n=== PEAK CHANGE (tou) vs size ===")
    sz = table_size(main)
    sz["ci"] = sz.apply(fmt_ci, axis=1)
    print(sz[sz.scenario == "C_lp"].pivot_table(index=["n", "cap"], columns="pv", values="ci", aggfunc="first").to_string())
    print(sz[sz.scenario == "B_lp"].pivot_table(index=["n", "cap"], columns="pv", values="ci", aggfunc="first").to_string())
    print("\n=== REVERSE-FLOW hours (n=25, 5 kWh), by tariff-independent ===")
    rv = table_reverse(main)
    print(rv[rv.tariff == "flat"][["pv", "scenario", "mean", "lo", "hi"]].round(0).to_string(index=False))
    print("\n=== SAVINGS decomposition (A$/yr, n=25, 5 kWh) ===")
    sv = table_savings(main)
    print(sv.pivot_table(index=["tariff", "pv", "mode"], columns="quantity", values="mean").round(0).to_string())
    print("\n=== ECONOMICS ===")
    ec = table_econ(main)
    print(ec[(ec.n == 25)][["tariff", "pv", "mode", "sav_B", "sav_C", "payback_B", "payback_C_home_price", "payback_C_arena",
                            "rho_match", "actual_ratio"]].round(2).to_string(index=False))
    try:
        ctrl = load("ctrl_")
        print("\n=== CONTROLLERS ===")
        ct = table_ctrl(ctrl)
        ct["ci"] = ct.apply(lambda r: fmt_ci(r, 2 if r.quantity in ("coincidence",) else 1), axis=1)
        print(ct.pivot_table(index=["pv", "ctrl", "who"], columns="quantity", values="mean").round(2).to_string())
    except FileNotFoundError:
        print("(ctrl not ready)")
    for pre, fn, args in (("years_", table_years, (main,)), ("sens_", table_sens, (main,))):
        try:
            d = load(pre)
            t = fn(d, *args)
            print(f"\n=== {pre} ===")
            t["ci"] = t.apply(fmt_ci, axis=1)
            print(t.head(60).to_string(index=False))
        except FileNotFoundError:
            print(f"({pre} not ready)")
    try:
        cl, mix = load("cl_"), load("mix_")
        t = table_clmix(cl, mix, main)
        t["ci"] = t.apply(fmt_ci, axis=1)
        print("\n=== CL / mixed ===")
        print(t.pivot_table(index=["tariff", "pv", "quantity", "scenario"], columns="variant", values="mean", aggfunc="first").round(1).to_string())
    except FileNotFoundError:
        print("(cl/mix not ready)")
