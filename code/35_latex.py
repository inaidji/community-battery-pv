"""Results -> paper link. Reads results/ (tables written by 30/32/33/34/37) and config.yaml and writes

  paper/generated/numbers.tex     one macro per number:  \\R{key} (results)  and  \\K{key} (configuration)
  paper/generated/tab_*.tex       LaTeX tables
  results/v2_tables/numbers_registry.csv   key, value, text, source file  (used by 36_check_numbers.py)

No number in the paper is typed by hand: the text uses \\R{...}/\\K{...}; 36_check_numbers.py enforces it.
Key scheme (t = flat|tou|dem|dem24, pv = 1|3, scen = Brule|Blp|Crule|Clp, mode = rule|lp):
  pk.t.pv.scen[.lo|.hi]        peak change vs no battery, %          (n = 25, 5 kWh/home)
  sz.n.cap.scen[.lo|.hi]       same by group size / battery size (ToU, PV x1)
  rev.pv.scen  revcut.pv.scen  reverse-flow hours (ToU) and reduction vs no battery (ratio of means, %)
  ct.pv.who.ctrl.q             controller study: q = peak|sav|coinc|cyc|chgkw
  seed.pv.who.q                random tie-break over 5 draws: mean|sd|min|max
  dc.t.pv.mode.q               savings and decomposition, A$ thousand
  bill.t.pv.scen               group bill, A$ thousand
  ec.t.pv.mode.q  ecc.pv.ctrl.q   payback (pbB, pbCg, pbCn) and break-even ratio (be)
  bt.pv.batt.scen.q            C-rate / efficiency sensitivity
  st.fam.id.q                  tests: d (mean difference), p (max p over partitions), ph (Holm)
  sn.* cm.* yr.* fr.*          sensitivity, controlled load / mixed groups, years, fairness
  ngs.* ng.*                   second dataset, strict (main) and imputed (sensitivity) versions
  dat.* rt.*                   data facts, run times
A suffix ":0", ":1" or ":2" selects the number of decimals (default given per quantity).
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import config as cfg
import sim

T = sim.ROOT / "results" / "v2_tables"
G = sim.ROOT / "paper" / "generated"
G.mkdir(parents=True, exist_ok=True)
REG = {}          # key -> (value, text, source)
TN = {"flat": "flat", "tou": "tou", "tou_demand": "dem", "tou_demand24": "dem24", "tou_exportlimit": "exp"}
SC = {"B_rule": "Brule", "B_lp": "Blp", "C_rule": "Crule", "C_lp": "Clp", "A": "A", "A_agg": "Aagg",
      "Bagg_rule": "Baggrule", "Bagg_lp": "Bagglp", "C_lp_minpeak": "Cmin"}


def fmt(v, d):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "---"
    s = f"{abs(v):.{d}f}"
    if float(s) == 0:
        return s
    ip, _, fp = s.partition(".")
    if len(ip) > 3:
        ip = f"{int(ip):,}".replace(",", "\\,")
    s = ip + ("." + fp if fp else "")
    return ("\\ensuremath{-}" if v < 0 else "") + s


def put(key, v, d=1, src="", alts=(0, 1, 2)):
    v = float(v)
    REG[key] = (v, fmt(v, d), src)
    for a in alts:
        REG[f"{key}:{a}"] = (v, fmt(v, a), src)


def put3(key, r, d=1, src="", scale=1.0):
    put(key, r["mean"] * scale, d, src)
    put(key + ".lo", r["lo"] * scale, d, src)
    put(key + ".hi", r["hi"] * scale, d, src)


def pvk(x):
    return str(int(round(float(x))))


def headline(d):
    return d[(d.n == 25) & (d.cap == 5.0) & (d.tkw == "base")] if "tkw" in d else d


# ---------------------------------------------------------------------------------------------------------
def numbers():
    # ---- configuration (\K) ----
    raw = yaml.safe_load(cfg.CONFIG_FILE.read_text())
    K = {}
    for sec in raw.values():
        for k, v in sec.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                K[k] = v
    K.update(ratio_gross=cfg.BATT_COST_COMMUNITY_AUD_PER_KWH / cfg.BATT_COST_HOME_AUD_PER_KWH,
             ratio_net=cfg.BATT_COST_COMMUNITY_NET_AUD_PER_KWH / cfg.BATT_COST_HOME_AUD_PER_KWH,
             batt_cost_community_net_aud_per_kwh=cfg.BATT_COST_COMMUNITY_NET_AUD_PER_KWH,
             cap_total_25=25 * 5.0, power_total_25=25 * 5.0 * cfg.BATT_C_RATE, eta_roundtrip=cfg.BATT_ETA_ONE_WAY ** 2,
             demand_c_per_kw_day=100 * cfg.DEMAND_AUD_PER_KW_DAY, discount_pct=100 * cfg.DISCOUNT_RATE,
             arena_grant_per_wh=cfg.ARENA_MAX_GRANT_AUD_PER_KWH / 1000, community_cost_per_wh=cfg.BATT_COST_COMMUNITY_AUD_PER_KWH / 1000,
             tou_ratio=cfg.TOU_PEAK_C / cfg.TOU_OFFPEAK_C)

    # ---- data facts ----
    for y, tag in (("2012-13", "aus"), ("nextgen-2018s", "ngs"), ("nextgen-2018", "ng")):
        d = sim.load_headline(y)
        days = len(d["dates"])
        put(f"dat.{tag}.homes", len(d["ids"]), 0, f"data/processed ({y})")
        put(f"dat.{tag}.days", days, 0, f"data/processed ({y})")
        put(f"dat.{tag}.load", d["load"].sum(1).mean() * 365 / days, 0, f"data/processed ({y})")
        put(f"dat.{tag}.pv", d["pv"].sum(1).mean() * 365 / days, 0, f"data/processed ({y})")
        put(f"dat.{tag}.kwp", float(np.mean(d["kwp"])), 2, f"data/processed ({y})")
        put(f"dat.{tag}.T", d["load"].shape[1], 0, f"data/processed ({y})")
        if tag == "aus":
            put("dat.aus.kwp3", 3 * float(np.mean(d["kwp"])), 1, "data/processed")
        if tag != "aus":
            put(f"dat.{tag}.battcap", float(np.mean(d["battcap"])), 1, f"data/processed ({y})")
            put(f"dat.{tag}.imputed", float(np.sum(d["imputed_days"])), 0, f"data/processed ({y})")
            put(f"dat.{tag}.homedays", days * len(d["ids"]), 0, f"data/processed ({y})")
            bym = pd.Series(1, index=pd.DatetimeIndex(d["dates"])).groupby(pd.DatetimeIndex(d["dates"]).month).sum()
            put(f"dat.{tag}.mindays", bym.min(), 0, f"data/processed ({y})")
            put(f"dat.{tag}.maxdays", bym.max(), 0, f"data/processed ({y})")

    # ---- peak change, headline ----
    pc = headline(pd.read_csv(T / "peak_change.csv"))
    for _, r in pc.iterrows():
        if r.scenario == "A_peak_kw":
            if r.tariff == "tou":
                put(f"pkw.{pvk(r.pv)}.A", r["mean"], 1, "peak_change.csv")
        elif r.tariff in TN and r.tariff != "tou_exportlimit":
            put3(f"pk.{TN[r.tariff]}.{pvk(r.pv)}.{SC[r.scenario]}", r, 1, "peak_change.csv")
    put("dat.aus.groups25", pc.groups.iloc[0], 0, "peak_change.csv")
    put("dat.aus.partitions", pc.partitions.iloc[0], 0, "peak_change.csv")
    sz = pd.read_csv(T / "peak_change_by_size.csv")
    for _, r in sz[(sz.tariff == "tou")].iterrows():
        put3(f"sz.{pvk(r.pv)}.{int(r.n)}.{r.cap:g}.{SC[r.scenario]}", r, 1, "peak_change_by_size.csv")
        put(f"dat.aus.groups{int(r.n)}", r.groups, 0, "peak_change_by_size.csv")
        put(f"dat.aus.partitions{int(r.n)}", r.partitions, 0, "peak_change_by_size.csv")

    # ---- reverse flow ----
    rf = pd.read_csv(T / "reverse_flow.csv")
    rf = rf[rf.tariff == "tou"]
    for _, r in rf.iterrows():
        put3(f"rev.{pvk(r.pv)}.{SC[r.scenario]}", r, 0, "reverse_flow.csv")
    for pv, g in rf.groupby("pv"):
        a = g[g.scenario == "A"]["mean"].iloc[0]
        for _, r in g.iterrows():
            if r.scenario != "A":
                put(f"revcut.{pvk(pv)}.{SC[r.scenario]}", 100 * (1 - r["mean"] / a), 0, "reverse_flow.csv")

    # ---- controllers ----
    ca = pd.read_csv(T / "controllers_all.csv")
    Q = {"d_peak_pct": ("peak", 1, 1), "saving": ("sav", 1, 1e-3), "coincidence": ("coinc", 2, 1), "cycles": ("cyc", 0, 1),
         "chg_peak_kw": ("chgkw", 1, 1)}
    for _, r in ca.iterrows():
        if r.quantity not in Q:
            continue
        q, d, s = Q[r.quantity]
        put3(f"ct.{pvk(r.pv)}.{r.who}.{r.ctrl}.{q}", r, d, "controllers_all.csv", s)
    sd = pd.read_csv(T / "seeds.csv")
    for _, r in sd.iterrows():
        q = "peak" if r.quantity == "d_peak_pct" else "sav"
        s = 1 if q == "peak" else 1e-3
        for c, name in (("mean", "mean"), ("std", "sd"), ("min", "min"), ("max", "max")):
            put(f"seed.{pvk(r.pv)}.{r.who}.{q}.{name}", r[c] * s, 1, "seeds.csv")
        put("seed.draws", r["count"], 0, "seeds.csv")
    ecc = pd.read_csv(T / "econ_controllers.csv")
    for _, r in ecc.iterrows():
        for c, k in (("payback_B", "pbB"), ("payback_C_gross", "pbCg"), ("payback_C_net", "pbCn"), ("breakeven_ratio", "be")):
            put(f"ecc.{pvk(r.pv)}.{r.ctrl}.{k}", r[c], 2 if k == "be" else 1, "econ_controllers.csv")
    # ---- savings decomposition, bills, economics ----
    dc = pd.read_csv(T / "decomposition.csv")
    for _, r in dc.iterrows():
        if r.tariff not in TN:
            continue
        k = f"dc.{TN[r.tariff]}.{pvk(r.pv)}.{r['mode']}"
        for c, name in (("sav_B", "B"), ("sav_Bagg", "Bagg"), ("sav_C", "C"), ("netting", "net"), ("billing_effect", "billing"),
                        ("pooling_effect", "pooling"), ("storage_Bagg", "stB"), ("storage_C", "stC")):
            if not np.isnan(r[c]):
                put(f"{k}.{name}", r[c] / 1000, 1, "decomposition.csv")
        put(f"{k}.gap", (r.sav_C - r.sav_B) / 1000, 2, "decomposition.csv")
        put(f"{k}.nshare", r.netting_share_pct, 0, "decomposition.csv")
        if not np.isnan(r.billing_share_of_gap_pct):
            put(f"{k}.bshare", r.billing_share_of_gap_pct, 0, "decomposition.csv")
    bl = headline(pd.read_csv(T / "bills.csv"))
    for _, r in bl.iterrows():
        if r.tariff in TN and r.scenario in SC:
            put(f"bill.{TN[r.tariff]}.{pvk(r.pv)}.{SC[r.scenario]}", r["mean"] / 1000, 1, "bills.csv")
    ec = pd.read_csv(T / "economics2.csv")
    for _, r in ec.iterrows():
        if r.tariff in TN:
            for c, k in (("payback_B", "pbB"), ("payback_C_gross", "pbCg"), ("payback_C_net", "pbCn"), ("breakeven_ratio", "be")):
                put(f"ec.{TN[r.tariff]}.{pvk(r.pv)}.{r['mode']}.{k}", r[c], 2 if k == "be" else 1, "economics2.csv")

    # ---- battery parameters ----
    bt = pd.read_csv(T / "batt.csv")
    for _, r in bt.iterrows():
        q, d, s = ("peak", 1, 1) if r.quantity == "d_peak_pct" else ("sav", 1, 1e-3)
        put3(f"bt.{pvk(r.pv)}.{r.batt.replace('_', '')}.{SC[r.scenario]}.{q}", r, d, "batt.csv", s)

    # ---- statistical tests ----
    stt = pd.read_csv(T / "stats_tests.csv")
    for _, r in stt.iterrows():
        k = f"st.{r.family}.{r.id}"
        put(k + ".d", r.mean_diff, 1, "stats_tests.csv")
        put(k + ".blo", r.boot_lo, 1, "stats_tests.csv")
        put(k + ".bhi", r.boot_hi, 1, "stats_tests.csv")
        put(k + ".p", r.p_max, 3, "stats_tests.csv", alts=(2, 3, 4))
        put(k + ".ph", r.p_holm, 3, "stats_tests.csv", alts=(2, 3, 4))
    STID = list(stt[stt.family == "pv1"].id)
    put("st.ncontrasts", len(STID), 0, "stats_tests.csv")
    put("st.phmax", max(REG[f"st.{f}.{i}.ph"][0] for f in ("pv1", "pv3") for i in STID if i != "Bpk"), 3, "stats_tests.csv", alts=(2, 3))
    put("st.phmax1", max(REG[f"st.pv1.{i}.ph"][0] for i in STID if i != "Bpk"), 3, "stats_tests.csv", alts=(2, 3))
    put("st.pmin11", 2 / 2 ** 11, 4, "exact Wilcoxon, 11 pairs", alts=(3, 4))
    put("st.pmin9", 2 / 2 ** 9, 4, "exact Wilcoxon, 9 pairs", alts=(3, 4))
    put("st.pmin5", 2 / 2 ** 5, 4, "exact Wilcoxon, 5 pairs", alts=(3, 4))

    # ---- sensitivity, controlled load / mixed groups, years ----
    sn = pd.read_csv(T / "sensitivity.csv")
    for _, r in sn.iterrows():
        tk = re.sub(r"[^a-z0-9]", "", str(r.tkw))
        q = SC.get(r.scenario, r.scenario) if r.quantity == "d_peak_pct" else f"{r.quantity.replace('sav_', 'sav').replace('_', '')}{r.scenario}"
        s, d = (1, 1) if r.quantity == "d_peak_pct" else (1e-3, 1)
        put(f"sn.{TN[r.tariff]}.{pvk(r.pv)}.{tk}.{q}", r["mean"] * s, d, "sensitivity.csv")
    cm = pd.read_csv(T / "clmix.csv")
    V = {"base": "base", "no_controlled_load": "nocl", "third_without_pv": "mix"}
    for _, r in cm.iterrows():
        q = SC.get(r.scenario, r.scenario) if r.quantity == "d_peak_pct" else r.quantity.replace("sav_", "sav").replace("_", "")
        s = 1 if r.quantity == "d_peak_pct" else 1e-3
        put(f"cm.{V[r.variant]}.{TN[r.tariff]}.{pvk(r.pv)}.{q}", r["mean"] * s, 1, "clmix.csv")
    yr = pd.read_csv(T / "years.csv")
    for _, r in yr.iterrows():
        if r.tariff in ("tou", "tou_demand24"):
            q = {"d_peak_pct": "pk", "reverse_flow_h": "rev", "netting_share_pct": "nshare", "peak_kw": "pkw"}[r.quantity]
            put(f"yr.{r.year}.{TN[r.tariff]}.{pvk(r.pv)}.{q}.{SC.get(r.scenario, r.scenario)}", r["mean"], 0, "years.csv")

    # ---- fairness ----
    fs = pd.read_csv(sim.ROOT / "results" / "fairness_summary.csv")
    for _, r in fs.iterrows():
        k = f"fr.{TN[r.tariff]}.{pvk(r.pv_scale)}.{r['rule'].replace('prop_', 'p').replace('consumption', 'cons')}"
        put(k + ".worse", 100 * (1 - r.share_better_off), 0, "fairness_summary.csv")
        put(k + ".neg", 100 * r.neg_share, 0, "fairness_summary.csv")
        put(k + ".corrpv", r.corr_pv, 2, "fairness_summary.csv")
        put(k + ".corrload", r.corr_load, 2, "fairness_summary.csv")
    sh = fs[fs["rule"] == "shapley"]
    put("fr.shapley.corrpv.min", sh.corr_pv.min(), 2, "fairness_summary.csv")
    put("fr.shapley.corrpv.max", sh.corr_pv.max(), 2, "fairness_summary.csv")
    put("fr.shapley.corrload.min", sh.corr_load.min(), 2, "fairness_summary.csv")
    put("fr.shapley.corrload.max", sh.corr_load.max(), 2, "fairness_summary.csv")
    pr = fs[(fs.pv_scale == 3.0) & fs["rule"].isin(["prop_consumption", "prop_bill", "equal"])]
    put("fr.3.simple.worse.min", 100 * (1 - pr.share_better_off.max()), 0, "fairness_summary.csv")
    put("fr.3.simple.worse.max", 100 * (1 - pr.share_better_off.min()), 0, "fairness_summary.csv")
    put("fr.3.shapley.worse.max", 100 * (1 - sh[sh.pv_scale == 3.0].share_better_off.min()), 0, "fairness_summary.csv")
    fa = pd.read_csv(sim.ROOT / "results" / "fairness.csv")
    put("fr.groups", fa["group"].nunique() if "group" in fa else np.nan, 0, "fairness.csv")

    # ---- second dataset ----
    for tag in ("ngs", "ng"):
        days = REG[f"dat.{tag}.days"][0]
        sc = 365.0 / days
        mn = pd.read_csv(T / f"{tag}_main.csv")
        mn = mn[mn.n == 10]
        put(f"{tag}.groups", mn.groups.max(), 0, f"{tag}_main.csv")
        put(f"{tag}.scale", sc, 3, f"{tag}_main.csv")
        for _, r in mn.iterrows():
            t = TN[r.tariff]
            if r.quantity == "d_peak_pct":
                put3(f"{tag}.pk.{t}.{SC[r.scenario]}", r, 1, f"{tag}_main.csv")
            elif r.quantity == "d_p99_pct":
                put3(f"{tag}.p99.{t}.{SC[r.scenario]}", r, 1, f"{tag}_main.csv")
            elif r.quantity in ("peak_import_kw", "peak_export_kw"):
                put(f"{tag}.{'pkw' if 'import' in r.quantity else 'pkexp'}.{t}.{SC[r.scenario]}", r["mean"], 1, f"{tag}_main.csv")
            elif r.quantity == "reverse_flow_h":
                put3(f"{tag}.rev.{t}.{SC[r.scenario]}", r, 0, f"{tag}_main.csv", sc)
            elif r.quantity == "bill_aud":
                put(f"{tag}.bill.{t}.{SC[r.scenario]}", r["mean"] * sc / 1000, 1, f"{tag}_main.csv")
            else:
                put3(f"{tag}.dc.{t}.{r.scenario}.{r.quantity.replace('sav_', '').replace('_effect', '').replace('storage_C', 'stC').replace('netting', 'net')}",
                     r, 1, f"{tag}_main.csv", sc / 1000)
        for t in ("flat", "tou", "dem", "dem24"):
            for mode in ("rule", "lp"):
                put(f"{tag}.dc.{t}.{mode}.nshare", 100 * REG[f"{tag}.dc.{t}.{mode}.net"][0] / REG[f"{tag}.dc.{t}.{mode}.C"][0], 0, f"{tag}_main.csv")
        a = REG[f"{tag}.rev.tou.A"][0]
        for s in ("Brule", "Blp", "Crule", "Clp"):
            put(f"{tag}.revcut.{s}", 100 * (1 - REG[f"{tag}.rev.tou.{s}"][0] / a), 0, f"{tag}_main.csv")
        cn = pd.read_csv(T / f"{tag}_controllers.csv")
        for _, r in cn.iterrows():
            if r.quantity == "d_peak_pct":
                put3(f"{tag}.ct.{r.who}.{r.ctrl}.peak", r, 1, f"{tag}_controllers.csv")
            elif r.quantity == "d_p99_pct":
                put3(f"{tag}.ct.{r.who}.{r.ctrl}.p99", r, 1, f"{tag}_controllers.csv")
            elif r.quantity == "saving":
                put3(f"{tag}.ct.{r.who}.{r.ctrl}.sav", r, 1, f"{tag}_controllers.csv", sc / 1000)
            else:
                put(f"{tag}.ct.{r.who}.{r.ctrl}.cyc", r["mean"] * sc, 0, f"{tag}_controllers.csv")
        ns = pd.read_csv(T / f"{tag}_stats.csv")
        for _, r in ns.iterrows():
            put(f"{tag}.st.{r.id}.d", r.mean_diff, 1, f"{tag}_stats.csv")
            put(f"{tag}.st.{r.id}.p", r.p_max, 3, f"{tag}_stats.csv", alts=(2, 3, 4))
            put(f"{tag}.st.{r.id}.ph", r.p_holm, 3, f"{tag}_stats.csv", alts=(2, 3, 4))
        put(f"{tag}.st.n", len(ns), 0, f"{tag}_stats.csv")
        put(f"{tag}.st.phmin", ns.p_holm.min(), 3, f"{tag}_stats.csv", alts=(1, 2, 3))

    # ---- experiments added after the review ----
    QN = {"d_peak_pct": ("peak", 1, 1, True), "d_p99_pct": ("p99", 1, 1, True), "d_p999_pct": ("p999", 1, 1, False), "saving": ("sav", 1, 1e-3, False),
          "h_ge90": ("h90", 0, 1, False), "h_ge999": ("h999", 0, 1, False), "cycles": ("cyc", 0, 1, False), "peak_export_kw": ("pkexp", 1, 1, False),
          "peak_import_kw": ("pkw", 1, 1, False), "p99_import_kw": ("p99kw", 1, 1, False), "pk_before_pct": ("before", 0, 1, False),
          "pk_after_pct": ("after", 0, 1, False), "pk_in_pct": ("inwin", 0, 1, False)}
    for fn, pre in (("rv_controllers.csv", "rv"), ("rv_ctrl10.csv", "r10")):
        t = pd.read_csv(T / fn)
        for _, r in t.iterrows():
            q, dd, sc_, three = QN[r.quantity]
            k = f"{pre}.{pvk(r.pv)}.{r.who}.{r.ctrl}.{q}"
            (put3(k, r, dd, fn, sc_) if three else put(k, r["mean"] * sc_, dd, fn))
    rn = pd.read_csv(T / "rv_rn.csv")
    RNQ = {"sav_B": "B", "sav_Bagg": "Bagg", "sav_C": "C", "billing_effect": "billing", "pooling_effect": "pooling", "netting": "net",
           "bill_A": "billA", "bill_C": "billC"}
    for _, r in rn.iterrows():
        k = f"rn.{TN[r.tariff.replace('_rn', '')]}.{pvk(r.pv)}.{r['mode']}"
        if r.quantity in RNQ:
            put(f"{k}.{RNQ[r.quantity]}", r["mean"] / 1000, 1, "rv_rn.csv")
        else:
            put3(f"{k}.{r.quantity.replace('d_peak_', 'pk').replace('d_p99_', 'p99')}", r, 1, "rv_rn.csv")
    for t in ("dem", "dem24"):
        for pv in ("1", "3"):
            for mode in ("rule", "lp"):
                k = f"rn.{t}.{pv}.{mode}"
                put(k + ".nshare", 100 * REG[k + ".net"][0] / REG[k + ".C"][0], 0, "rv_rn.csv")
    sw = pd.read_csv(T / "rv_sweep.csv")
    VN = {"peak_c36.058": "r15", "peak_c96.1548": "r40", "peak_start13": "w13", "peak_start17": "w17", "base": "free"}
    for _, r in sw.iterrows():
        q, dd, sc_, three = QN[r.quantity]
        k = f"sw.{VN[r.tkw]}.{pvk(r.pv)}.{SC[r.scenario]}.{q}"
        (put3(k, r, dd, "rv_sweep.csv", sc_) if three else put(k, r["mean"] * sc_, dd, "rv_sweep.csv"))
    co = pd.read_csv(T / "rv_coord.csv")
    for _, r in co.iterrows():
        q, dd, sc_, three = QN[r.quantity]
        k = f"co.{r.scenario.replace('_lp', '')}.{q}"
        (put3(k, r, dd, "rv_coord.csv", sc_) if three else put(k, r["mean"] * sc_, dd, "rv_coord.csv"))
    put("co.groups", co.groups.max() * co.partitions.max(), 0, "rv_coord.csv")
    tl = pd.read_csv(T / "tails.csv")
    for _, r in tl.iterrows():
        if r.tariff in TN and r.tkw == "base":
            q = {"d_p99_pct": "p99", "peak_export_kw": "pkexp", "export_kwh": "expmwh", "peak_import_kw": "pkw"}[r.quantity]
            sc_ = 1e-3 if q == "expmwh" else 1
            k = f"tl.{TN[r.tariff]}.{pvk(r.pv)}.{SC[r.scenario]}.{q}"
            (put3(k, r, 1, "tails.csv", sc_) if q == "p99" else put(k, r["mean"] * sc_, 1, "tails.csv"))
    # p99 for the controllers of the earlier study that were not re-run (random tie-break)
    for _, r in ca[ca.quantity == "d_p99_pct"].iterrows():
        put3(f"ct.{pvk(r.pv)}.{r.who}.{r.ctrl}.p99", r, 1, "controllers_all.csv")
    cv = pd.read_csv(T / "coverage.csv")
    for _, r in cv.iterrows():
        put(f"cov.{r.controller}.t", r.coverage_t_pct, 1, "coverage.csv")
        put(f"cov.{r.controller}.boot", r.coverage_bootstrap_pct, 1, "coverage.csv")
    put("cov.t.min", cv.coverage_t_pct.min(), 0, "coverage.csv"); put("cov.t.max", cv.coverage_t_pct.max(), 0, "coverage.csv")
    put("cov.boot.min", cv.coverage_bootstrap_pct.min(), 0, "coverage.csv"); put("cov.boot.max", cv.coverage_bootstrap_pct.max(), 0, "coverage.csv")
    put("cov.reps", cv.replications.max(), 0, "coverage.csv")
    # net present value (A$ thousand), life = min(max life, cycle life / cycles)
    ann = lambda y: (1 - (1 + cfg.DISCOUNT_RATE) ** -y) / cfg.DISCOUNT_RATE
    aL = ann(cfg.BATT_LIFE_MAX_YEARS)
    put("npv.annuity", aL, 2, "config.yaml")
    put("npv.maxcyc", cfg.BATT_CYCLE_LIFE / cfg.BATT_LIFE_MAX_YEARS, 0, "config.yaml")
    put("npv.cycmax.obs", max(v[0] for k, v in REG.items() if k.endswith(".cyc") and ":" not in k), 0, "controllers_all.csv, rv_controllers.csv")
    KB = 125 * cfg.BATT_COST_HOME_AUD_PER_KWH
    for _, r in ec.iterrows():
        if r.tariff in TN:
            k = f"npv.{TN[r.tariff]}.{pvk(r.pv)}.{r['mode']}"
            put(k + ".B", (r.sav_B * aL - KB) / 1000, 0, "economics2.csv")
            put(k + ".Cg", (r.sav_C * aL - 125 * cfg.BATT_COST_COMMUNITY_AUD_PER_KWH) / 1000, 0, "economics2.csv")
            put(k + ".Cn", (r.sav_C * aL - 125 * cfg.BATT_COST_COMMUNITY_NET_AUD_PER_KWH) / 1000, 0, "economics2.csv")
    for c_ in ("mpc_react_persist", "spread"):
        sB, sC = REG[f"rv.1.B.{c_}.sav"][0] * 1000, REG[f"rv.1.C.{c_}.sav"][0] * 1000
        put(f"npv.ctrl.{c_}.B", (sB * aL - KB) / 1000, 0, "rv_controllers.csv")
        put(f"npv.ctrl.{c_}.Cg", (sC * aL - 125 * cfg.BATT_COST_COMMUNITY_AUD_PER_KWH) / 1000, 0, "rv_controllers.csv")
        put(f"npv.ctrl.{c_}.Cn", (sC * aL - 125 * cfg.BATT_COST_COMMUNITY_NET_AUD_PER_KWH) / 1000, 0, "rv_controllers.csv")
        put(f"pb.ctrl.{c_}.B", KB / sB, 1, "rv_controllers.csv")
        put(f"pb.ctrl.{c_}.Cg", 125 * cfg.BATT_COST_COMMUNITY_AUD_PER_KWH / sC, 1, "rv_controllers.csv")
        put(f"pb.ctrl.{c_}.Cn", 125 * cfg.BATT_COST_COMMUNITY_NET_AUD_PER_KWH / sC, 1, "rv_controllers.csv")
    # years: intervals
    for _, r in yr.iterrows():
        if r.tariff == "tou" and r.quantity == "d_peak_pct":
            put3(f"yri.{r.year}.{pvk(r.pv)}.{SC[r.scenario]}", r, 1, "years.csv")
    # rated power of the block relative to the mean no-battery peak
    put("blk.aus25", 100 * 62.5 / REG["pkw.1.A"][0], 0, "peak_change.csv")
    put("blk.aus10", 100 * 25.0 / REG["r10.1.A.none.pkw"][0], 0, "rv_ctrl10.csv")
    put("blk.ngs10", 100 * 25.0 / REG["ngs.pkw.tou.A"][0], 0, "ngs_main.csv")
    put("blk.ng10", 100 * 25.0 / REG["ng.pkw.tou.A"][0], 0, "ng_main.csv")
    put("perhome.A", REG["pkw.1.A"][0] / 25, 1, "peak_change.csv")
    put("perhome.Clp", REG["tl.tou.1.Clp.pkw"][0] / 25, 1, "tails.csv")
    c0 = REG["rv.1.C.mpc_react_persist.sav"][0]
    for v in ("capP90", "capP100", "capP110", "dcap", "cap"):
        put(f"loss.{v}", 100 * (1 - REG[f"rv.1.C.mpc_react_persist_{v}.sav"][0] / c0), 1, "rv_controllers.csv")
    put("loss.spread", 100 * (1 - REG["rv.1.C.spread.sav"][0] / REG["rv.1.C.lp.sav"][0]), 1, "rv_controllers.csv")
    opt = ["lp", "lp_minpeak", "lp_maxpeak", "lp_cap"]
    for pv in ("1", "3"):
        put(f"set.{pv}.p99.min", min(REG[f"rv.{pv}.C.{c}.p99"][0] for c in opt), 0, "rv_controllers.csv")
        put(f"set.{pv}.p99.max", max(max(REG[f"rv.{pv}.C.{c}.p99"][0] for c in opt), REG[f"ct.{pv}.C.lp_random.p99"][0]), 0, "rv_controllers.csv, controllers_all.csv")

    for pv in ("1", "3"):
        put(f"set.{pv}.width", REG[f"rv.{pv}.C.lp_maxpeak.peak"][0] - REG[f"rv.{pv}.C.lp_minpeak.peak"][0], 0, "rv_controllers.csv")
    put("rng.nshare.all.max", max(REG["rn.dem24.1.lp.nshare"][0], REG["rn.dem.1.lp.nshare"][0], REG["dc.dem24.1.lp.nshare"][0], REG["dc.dem.1.lp.nshare"][0]), 0, "decomposition.csv, rv_rn.csv")

    # ---- ranges quoted in the text ----
    ops = ["lp", "lp_random", "jit", "mpc_react_perfect", "mpc_react_persist"]
    for pv in ("1", "3"):
        v = [REG[f"ct.{pv}.C.{c}.peak"][0] for c in ops]
        put(f"rng.{pv}.C.uncoord.min", min(v), 0, "controllers_all.csv")
        put(f"rng.{pv}.C.uncoord.max", max(v), 0, "controllers_all.csv")
    for mode in ("rule", "lp"):
        for grp, ts in (("energy", ("flat", "tou")), ("demand", ("dem", "dem24"))):
            v = [REG[f"dc.{t}.1.{mode}.nshare"][0] for t in ts]
            put(f"rng.nshare.{mode}.{grp}.min", min(v), 0, "decomposition.csv")
            put(f"rng.nshare.{mode}.{grp}.max", max(v), 0, "decomposition.csv")
        for grp, ts in (("energy", ("flat", "tou")), ("demand", ("dem", "dem24"))):
            v = [REG[f"ec.{t}.{pv}.{mode}.be"][0] for t in ts for pv in ("1", "3")]
            put(f"rng.be.{mode}.{grp}.min", min(v), 2, "economics2.csv")
            put(f"rng.be.{mode}.{grp}.max", max(v), 2, "economics2.csv")
    put("rng.capcost.1", (REG["ct.1.C.mpc_react_persist.sav"][0] - REG["ct.1.C.mpc_react_persist_cap.sav"][0]) * 1000, 0, "controllers_all.csv")
    put("rng.capcost.1.pct", 100 * (1 - REG["ct.1.C.mpc_react_persist_cap.sav"][0] / REG["ct.1.C.mpc_react_persist.sav"][0]), 1, "controllers_all.csv")
    put("rng.persistloss.B", 100 * (1 - REG["ct.1.B.mpc_react_persist.sav"][0] / REG["ct.1.B.lp.sav"][0]), 0, "controllers_all.csv")
    put("rng.persistloss.C", 100 * (1 - REG["ct.1.C.mpc_react_persist.sav"][0] / REG["ct.1.C.lp.sav"][0]), 0, "controllers_all.csv")
    put("rng.spreadloss.C", 100 * (1 - REG["ct.1.C.spread.sav"][0] / REG["ct.1.C.lp.sav"][0]), 0, "controllers_all.csv")
    put("rng.jitcaploss.C", 100 * (1 - REG["ct.1.C.jit_cap.sav"][0] / REG["ct.1.C.lp.sav"][0]), 0, "controllers_all.csv")
    for k, src_ in (("pk.dem24.1.Clp", "peak_change.csv"), ("pk.dem24.3.Clp", "peak_change.csv"), ("ct.1.C.lp_minpeak.peak", "controllers_all.csv"),
                    ("ct.3.C.lp_minpeak.peak", "controllers_all.csv"), ("st.pv3.capBill.d", "stats_tests.csv"),
                    ("ngs.pk.dem24.Clp", "ngs_main.csv"), ("ngs.ct.C.lp_minpeak.peak", "ngs_controllers.csv")):
        put("abs." + k, abs(REG[k][0]), 1, src_)
    allbe = lambda ts: [REG[f"ec.{t}.{pv}.{m}.be"][0] for t in ts for pv in ("1", "3") for m in ("rule", "lp")]
    put("rng.be.energy.min", min(allbe(("flat", "tou"))), 2, "economics2.csv")
    put("rng.be.energy.max", max(allbe(("flat", "tou"))), 2, "economics2.csv")
    put("rng.be.demand.min", min(allbe(("dem", "dem24"))), 2, "economics2.csv")
    put("rng.be.demand.max", max(allbe(("dem", "dem24"))), 2, "economics2.csv")
    put("rng.chgshare", 100 * REG["ct.1.C.lp.chgkw"][0] / REG["pkw.1.A"][0], 0, "controllers_all.csv, peak_change.csv")

    # ---- run times (minutes) ----
    logs = {"main1": ("run_resume.log", r"stage 1 done \((\d+)s\)"), "main": ("run_resume.log", r"^done \((\d+)s\)"),
            "extra1": ("run_extra_stage1.log", r"^done \((\d+)s\)"), "extra2": ("run_extra_stage2.log", r"^done \((\d+)s\)"),
            "ng": ("run_nextgen.log", r"^done \((\d+)s\)"), "ngs": ("run_nextgen_strict.log", r"^done \((\d+)s\)"),
            "rv1": ("run_review_stage1.log", r"^done \((\d+)s\)"), "rv2": ("run_review_stage2.log", r"^done \((\d+)s\)"),
            "rv3": ("run_review_coord.log", r"^done \((\d+)s\)")}
    for k, (f, pat) in logs.items():
        p = sim.ROOT / "results" / "v2" / f
        mm = re.findall(pat, p.read_text(), re.M) if p.exists() else []
        if mm:
            put(f"rt.{k}", int(mm[-1]) / 60, 0, f"results/v2/{f}")
    put("rt.rv", sum(REG[f"rt.{k}"][0] for k in ("rv1", "rv2", "rv3")), 0, "results/v2/run_review_*.log")
    put("rt.total", sum(REG[f"rt.{k}"][0] for k in ("main", "extra1", "extra2", "ng", "ngs", "rv")) / 60, 0, "results/v2/*.log")
    put("rt.jobs", cfg.N_JOBS, 0, "config.yaml")
    return K


def kfmt(v):
    if isinstance(v, int) or float(v).is_integer():
        return fmt(float(v), 0)
    s = f"{v:.4f}".rstrip("0")
    return s


def write_numbers(K):
    L = ["% GENERATED by code/35_latex.py -- do not edit.",
         "\\makeatletter",
         "\\newcommand{\\R}[1]{\\ifcsname R@#1\\endcsname\\csname R@#1\\endcsname\\else\\errmessage{Unknown result key: #1}\\fi}",
         "\\newcommand{\\K}[1]{\\ifcsname K@#1\\endcsname\\csname K@#1\\endcsname\\else\\errmessage{Unknown config key: #1}\\fi}"]
    for k, (v, s, src) in sorted(REG.items()):
        L.append(f"\\expandafter\\def\\csname R@{k}\\endcsname{{{s}}}")
    for k, v in sorted(K.items()):
        L.append(f"\\expandafter\\def\\csname K@{k}\\endcsname{{{kfmt(v)}}}")
        for a in (0, 1, 2):
            L.append(f"\\expandafter\\def\\csname K@{k}:{a}\\endcsname{{{fmt(float(v), a)}}}")
    L.append("\\makeatother")
    (G / "numbers.tex").write_text("\n".join(L) + "\n")
    rows = [dict(kind="R", key=k, value=v, text=s, source=src) for k, (v, s, src) in sorted(REG.items())]
    rows += [dict(kind="K", key=k, value=v, text=kfmt(v), source="config.yaml") for k, v in sorted(K.items())]
    pd.DataFrame(rows).to_csv(T / "numbers_registry.csv", index=False)


# ---------------------------------------------------------------------------------------------------------
def r(key):
    return REG[key][1]


def ci(key, d=None):
    k = (lambda s: f"{key}{s}" + (f":{d}" if d is not None else ""))
    return f"{REG[k('')][1]} [{REG[k('.lo')][1]}, {REG[k('.hi')][1]}]"


def table(name, lines):
    (G / f"tab_{name}.tex").write_text("\n".join(lines) + "\n")   # no comment line: the file is read inside a tabular


TLAB = {"flat": "Flat", "tou": "ToU", "dem": "ToU + demand (15--21~h)", "dem24": "ToU + demand (24~h)"}
CROWS = [("rule", "Self-consumption rule"), ("lp", "Cost-min., solver default"), ("lp_minpeak", "Cost-min., peak-minimising"),
         ("lp_maxpeak", "Cost-min., peak-maximising"), ("lp_cap", "Cost-min., import cap"), ("jit", "Just-in-time top-up"),
         ("jit_cap", "Just-in-time top-up, import cap"), ("spread", "Spread top-up"),
         ("mpc_react_persist", "Day-ahead, persistence"), ("mpc_react_persist_e0", "Day-ahead, no end-of-day reserve"),
         ("mpc_react_persist_h21", "Day-ahead, window from 21:00"),
         ("mpc_react_persist_capP100", "Day-ahead, cap at last year's peak"), ("mpc_react_persist_capP90", "Day-ahead, cap at 90\\% of last year's peak"),
         ("mpc_react_persist_dcap", "Day-ahead, cap at yesterday's maximum")]
CALL = [("rule", "Self-consumption rule"), ("lp", "Cost-min., solver default"), ("lp_random", "Cost-min., random tie-break"),
        ("lp_minpeak", "Cost-min., peak-minimising"), ("lp_cap", "Cost-min., import cap"), ("jit", "Just-in-time top-up"),
        ("jit_cap", "Just-in-time top-up, import cap"), ("spread", "Spread top-up"), ("mpc_react_perfect", "Day-ahead, perfect forecast"),
        ("mpc_react_persist", "Day-ahead, persistence forecast"), ("mpc_react_persist_cap", "Day-ahead, persistence, import cap"),
        ("mpc_persist", "Day-ahead, persistence, plan executed as set-points"), ("spread_cap", "Spread top-up, import cap")]
STLAB = {"Cpk": "Annual maximum: community vs no battery (pp)", "Bpk": "Annual maximum: individual vs no battery (pp)",
         "CvB": "Annual maximum: community vs individual (pp)", "Cp99": "99th percentile: community vs no battery (pp)",
         "minp99": "99th percentile: community, peak-minimising, vs no battery (pp)",
         "capp99": "99th percentile: community, import cap, vs no battery (pp)",
         "sprp99": "99th percentile: community, spread top-up, vs no battery (pp)",
         "sprVdef": "Annual maximum: spread top-up vs solver default, community (pp)",
         "dap99": "99th percentile: community, day-ahead, vs no battery (pp)",
         "dacapp99": "99th percentile: day-ahead, cap at last year's peak, vs no battery (pp)",
         "dcapp99": "99th percentile: day-ahead, cap at yesterday's maximum, vs no battery (pp)",
         "capBill": "Bill: day-ahead, cap at last year's peak vs no cap (A\\$)", "dcapBill": "Bill: day-ahead, cap at yesterday's maximum vs no cap (A\\$)",
         "dem24": "Annual maximum: community vs no battery, 24~h demand charge (pp)",
         "dem24p99": "99th percentile: community vs no battery, 24~h demand charge (pp)",
         "rev": "Reverse-flow hours: community vs individual (h)", "billCvB": "Bill: community vs individual (A\\$)",
         "billing": "Bill: billing effect, B$_{\\rm agg}$ vs B (A\\$)", "pooling": "Bill: pooling effect, C vs B$_{\\rm agg}$ (A\\$)"}


def opt(key, f=None):
    return (f or r)(key) if (key in REG or key + ".lo" in REG) and key in REG else "---"


def tables():
    L = []
    for t in ("flat", "tou", "dem", "dem24"):
        for pv in ("1", "3"):
            k = f"pk.{t}.{pv}"
            L.append(f"{TLAB[t] if pv == '1' else ''} & $\\times{pv}$ & {ci(k + '.Blp')} & {ci(k + '.Clp')} & {r(f'tl.{t}.{pv}.Blp.p99')} & {r(f'tl.{t}.{pv}.Clp.p99')}\\\\")
    table("peak", L)

    for pv, name in (("1", "ctrl"), ("3", "ctrl3")):
        L = [f"No battery & --- & --- & --- & --- & --- & {r(f'rv.{pv}.A.none.h90')} & ---\\\\", "\\midrule"]
        for c, lab in CROWS:
            b, cc = f"rv.{pv}.B.{c}", f"rv.{pv}.C.{c}"
            row = [lab] + ([r(b + ".peak"), r(b + ".p99"), r(b + ".sav")] if b + ".peak" in REG else ["---"] * 3)
            row += [ci(cc + ".peak"), ci(cc + ".p99"), r(cc + ".h90"), r(cc + ".sav")]
            L.append(" & ".join(row) + "\\\\")
        table(name, L)

    # Supplementary table: every controller of the original controller study (controllers_all.csv), peak change and saving
    L = []
    for c, lab in CALL:
        cells = [opt(f"ct.{pv}.{w}.{c}.{q}") for pv in ("1", "3") for w in "BC" for q in ("peak", "sav")]
        L.append(f"{lab} & " + " & ".join(cells) + "\\\\")
    table("ctrl_all", L)

    L = []
    for t in ("flat", "tou", "dem", "dem24"):
        f = lambda q: f"{r(f'dc.{t}.1.rule.{q}')} / {r(f'dc.{t}.1.lp.{q}')}"
        L.append(f"{TLAB[t]} & {f('B')} & {f('Bagg')} & {f('C')} & {r(f'dc.{t}.1.lp.net')} & {f('billing')} & {f('pooling')} & "
                 f"{r(f'dc.{t}.1.rule.nshare')} / {r(f'dc.{t}.1.lp.nshare')}\\\\")
    L.append("\\midrule")
    for t, lab in (("dem", "Revenue-neutral, demand (15--21~h)"), ("dem24", "Revenue-neutral, demand (24~h)")):
        f = lambda q: f"{r(f'rn.{t}.1.rule.{q}')} / {r(f'rn.{t}.1.lp.{q}')}"
        L.append(f"{lab} & {f('B')} & {f('Bagg')} & {f('C')} & {r(f'rn.{t}.1.lp.net')} & {f('billing')} & {f('pooling')} & "
                 f"{r(f'rn.{t}.1.rule.nshare')} / {r(f'rn.{t}.1.lp.nshare')}\\\\")
    table("sav", L)

    L = []
    for t in ("flat", "tou", "dem", "dem24"):
        row = [TLAB[t]]
        for mode in ("rule", "lp"):
            row += [r(f"ec.{t}.1.{mode}.{q}") for q in ("pbB", "pbCg", "pbCn", "be")]
        L.append(" & ".join(row) + "\\\\")
    L.append("\\midrule")
    L.append("ToU, spread top-up & --- & --- & --- & --- & " + " & ".join(r(f"pb.ctrl.spread.{q}") for q in ("B", "Cg", "Cn")) + f" & {r('ecc.1.spread.be')}\\\\")
    L.append("ToU, day-ahead (persistence) & --- & --- & --- & --- & " + " & ".join(r(f"pb.ctrl.mpc_react_persist.{q}") for q in ("B", "Cg", "Cn")) + f" & {r('ecc.1.mpc_react_persist.be')}\\\\")
    table("econ", L)
    L = []
    for t in ("flat", "tou", "dem", "dem24"):
        row = [TLAB[t]]
        for mode in ("rule", "lp"):
            row += [r(f"npv.{t}.1.{mode}.{q}") for q in ("B", "Cg", "Cn")]
        L.append(" & ".join(row) + "\\\\")
    L.append("\\midrule")
    for c_, lab in (("spread", "ToU, spread top-up"), ("mpc_react_persist", "ToU, day-ahead (persistence)")):
        L.append(f"{lab} & --- & --- & --- & " + " & ".join(r(f"npv.ctrl.{c_}.{q}") for q in ("B", "Cg", "Cn")) + "\\\\")
    table("npv", L)

    def cell(k):
        return ci(k) if k in REG else "---"
    rows = [("Annual maximum, community, solver default", "pk.tou.1.Clp", "sz.1.10.5.Clp", "{}.pk.tou.Clp"),
            ("Annual maximum, individual, solver default", "pk.tou.1.Blp", "sz.1.10.5.Blp", "{}.pk.tou.Blp"),
            ("Annual maximum, community, peak-minimising", "rv.1.C.lp_minpeak.peak", "r10.1.C.lp_minpeak.peak", "{}.ct.C.lp_minpeak.peak"),
            ("Annual maximum, community, day-ahead", "rv.1.C.mpc_react_persist.peak", "r10.1.C.mpc_react_persist.peak", "{}.ct.C.mpc_react_persist.peak"),
            ("Annual maximum, community, spread top-up", "rv.1.C.spread.peak", "r10.1.C.spread.peak", "{}.ct.C.spread.peak"),
            ("Annual maximum, community, 24~h demand charge", "pk.dem24.1.Clp", "none", "{}.pk.dem24.Clp"),
            ("99th percentile, community, solver default", "rv.1.C.lp.p99", "r10.1.C.lp.p99", "{}.p99.tou.Clp"),
            ("99th percentile, community, peak-minimising", "rv.1.C.lp_minpeak.p99", "r10.1.C.lp_minpeak.p99", "{}.ct.C.lp_minpeak.p99"),
            ("99th percentile, community, day-ahead", "rv.1.C.mpc_react_persist.p99", "r10.1.C.mpc_react_persist.p99", "{}.ct.C.mpc_react_persist.p99"),
            ("99th percentile, community, spread top-up", "rv.1.C.spread.p99", "r10.1.C.spread.p99", "{}.ct.C.spread.p99"),
            ("99th percentile, community, 24~h demand charge", "tl.dem24.1.Clp.p99", "none", "{}.p99.dem24.Clp")]
    L = [f"{lab} & {cell(a)} & {cell(b)} & {cell(c.format('ngs'))} & {cell(c.format('ng'))}\\\\" for lab, a, b, c in rows]
    L.append("\\midrule")
    L.append(f"Rated power of the community battery / mean no-battery peak (\\%) & {r('blk.aus25')} & {r('blk.aus10')} & {r('blk.ngs10')} & {r('blk.ng10')}\\\\")
    L.append(f"Reverse-flow hours per year: none / individual / community & {r('rev.1.A')} / {r('rev.1.Blp')} / {r('rev.1.Clp')} & --- & "
             f"{r('ngs.rev.tou.A')} / {r('ngs.rev.tou.Blp')} / {r('ngs.rev.tou.Clp')} & {r('ng.rev.tou.A')} / {r('ng.rev.tou.Blp')} / {r('ng.rev.tou.Clp')}\\\\")
    L.append(f"Peak export (kW): none / individual / community & {r('tl.tou.1.A.pkexp')} / {r('tl.tou.1.Blp.pkexp')} / {r('tl.tou.1.Clp.pkexp')} & --- & "
             f"{r('ngs.pkexp.tou.A')} / {r('ngs.pkexp.tou.Blp')} / {r('ngs.pkexp.tou.Clp')} & {r('ng.pkexp.tou.A')} / {r('ng.pkexp.tou.Blp')} / {r('ng.pkexp.tou.Clp')}\\\\")
    L.append(f"Netting share of the community saving (\\%) & {r('dc.tou.1.lp.nshare')} & --- & {r('ngs.dc.tou.lp.nshare')} & {r('ng.dc.tou.lp.nshare')}\\\\")
    table("nextgen", L)

    L = []
    for i, text in STLAB.items():
        d = 0 if "A\\$" in text or "(h)" in text else 1
        a = f"{r(f'st.pv1.{i}.d:{d}')} [{r(f'st.pv1.{i}.blo:{d}')}, {r(f'st.pv1.{i}.bhi:{d}')}]"
        ngd = f"{r(f'ng.st.{i}.d:{d}')} & {r(f'ng.st.{i}.p')} / {r(f'ng.st.{i}.ph')}" if f"ng.st.{i}.p" in REG else "--- & ---"
        L.append(f"{text} & {a} & {r(f'st.pv1.{i}.p')} / {r(f'st.pv1.{i}.ph')} & {r(f'st.pv3.{i}.d:{d}')} & "
                 f"{r(f'st.pv3.{i}.p')} / {r(f'st.pv3.{i}.ph')} & {ngd}\\\\")
    table("stats", L)

    L = []
    for v, lab in (("base", "ToU (base): peak price \\K{tou_ratio:1} times off-peak, window 15--21~h"), ("r15", "Peak price 1.5 times off-peak"),
                   ("r40", "Peak price 4 times off-peak"), ("w17", "Window 17:00--21:00"), ("w13", "Window 13:00--21:00"),
                   ("free", "Free import 11:00--14:00, no feed-in credit")):
        if v == "base":
            g = lambda scn, q: {"peak": r(f"pk.tou.1.{scn}"), "p99": r(f"tl.tou.1.{scn}.p99")}[q]
            mp, sv, mp99 = r("rv.1.C.lp_minpeak.peak"), r("dc.tou.1.lp.C"), r("rv.1.C.lp_minpeak.p99")
        else:
            g = lambda scn, q: r(f"sw.{v}.1.{scn}.{q}")
            mp, sv, mp99 = r(f"sw.{v}.1.Cmin.peak"), r(f"sw.{v}.1.Clp.sav"), r(f"sw.{v}.1.Cmin.p99")
        L.append(f"{lab} & {g('Blp', 'peak')} & {g('Blp', 'p99')} & {g('Clp', 'peak')} & {g('Clp', 'p99')} & {mp} & {mp99} & {sv}\\\\")
    table("sweep", L)

    L = []
    for n in (10, 25, 50):
        for cap in ("2.5", "5", "10"):
            L.append(f"{n} & {cap} & {ci(f'sz.1.{n}.{cap}.Blp')} & {ci(f'sz.1.{n}.{cap}.Clp')}\\\\")
    table("size", L)

    L = []
    for b, lab_ in (("crate0.25", "Rated power 0.25C"), ("base", "Rated power 0.5C, $\\eta=0.95$ (base)"), ("crate1", "Rated power 1C"),
                    ("eta0.9", "$\\eta=0.90$"), ("eta0.98", "$\\eta=0.98$")):
        mp = ci(f"bt.1.{b}.Cmin.peak") if f"bt.1.{b}.Cmin.peak" in REG else (ci("ct.1.C.lp_minpeak.peak") if b == "base" else "---")
        L.append(f"{lab_} & {ci(f'bt.1.{b}.Blp.peak')} & {ci(f'bt.1.{b}.Clp.peak')} & {mp} & {r(f'bt.1.{b}.Blp.sav')} & {r(f'bt.1.{b}.Clp.sav')}\\\\")
    table("batt", L)

    L = []
    for yv in ("2010-11", "2011-12", "2012-13"):
        L.append(f"{yv} & {ci(f'yri.{yv}.1.Blp')} & {ci(f'yri.{yv}.1.Clp')} & {ci(f'yri.{yv}.3.Blp')} & {ci(f'yri.{yv}.3.Clp')}\\\\")
    table("years", L)


if __name__ == "__main__":
    K = numbers()
    write_numbers(K)
    tables()
    print(f"{len(REG)} result macros, {len(K)} config macros -> {G}")
