"""Paper figures for the revised manuscript (reads results/v2 and results/v2_tables)."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg
import exp
import sim
from analysis_lib import load, paired, pct_change, diff
import importlib
T30 = importlib.import_module("30_tables")

FIG = sim.ROOT / "results" / "figures_v2"
FIG.mkdir(exist_ok=True)
H = cfg.DT_H

C = {"A": "#6b6b6b", "B": "#D55E00", "C": "#0072B2", "Bagg": "#E69F00", "green": "#009E73", "purple": "#CC79A7",
     "sky": "#56B4E9", "yellow": "#F0E442"}
TL = {"flat": "Flat", "tou": "ToU", "tou_demand": "ToU +\ndemand (15-21h)", "tou_demand24": "ToU +\ndemand (24 h)",
      "tou_exportlimit": "ToU +\nexport limit"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


# Greyscale safety: every fill colour has its own hatch, applied to all bars when a figure is saved, so that
# series can be told apart without colour. Colours are fixed per role in every figure:
# B (individual) vermilion '///', C (community) blue solid, B_agg / rule variants orange '...', sky '\\\\', grey 'xx'.
HATCH = {"#D55E00": "///", "#0072B2": "", "#E69F00": "...", "#56B4E9": "\\\\", "#6b6b6b": "xx", "#bdbdbd": "", "#009E73": "++",
         "#CC79A7": "oo", "#F0E442": "--"}


def _hatch(fig):
    from matplotlib.colors import to_hex
    from matplotlib.patches import Rectangle
    for ax in fig.axes:
        for pt in list(ax.patches) + [h for lg in [ax.get_legend()] if lg for h in lg.get_patches()]:
            if isinstance(pt, Rectangle):
                h = HATCH.get(to_hex(pt.get_facecolor()).upper().replace("#D55E00", "#D55E00"), None)
                key = to_hex(pt.get_facecolor())
                for k, v in HATCH.items():
                    if k.lower() == key.lower():
                        pt.set_hatch(v)
                        pt.set_edgecolor("black")
                        pt.set_linewidth(0.4)


def save(fig, name):
    _hatch(fig)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    print("saved", name)


def errbars(df):
    return np.vstack([df["mean"] - df["lo"], df["hi"] - df["mean"]])


# ---------------------------------------------------------------------------------------
def fig_day():
    """A winter weekday at the group connection: the same cost, different peaks."""
    sp = exp.spec(pv=1.0, tname="tou", cap=5.0)
    d = exp.get_data(sp)
    idx = sim.make_partitions(d, 25, 4, seed=1025)[0][0]
    tariff = exp.get_tariff(sp, d)
    load, pv = d["load"][idx], d["pv"][idx]
    net = (load - pv).sum(axis=0)
    cap_tot = 125.0
    flows = {"A": net}
    for c in ("lp", "lp_minpeak", "spread", "lp_cap"):
        flows[f"C_{c}"] = sim.group_flow(load, pv, cap_tot, c, tariff)[0]
    flows["B_lp"] = np.load(exp.cache_path(sp, "lp"))["flow"][idx].sum(axis=0)
    dates = pd.DatetimeIndex(d["dates"])
    peaks = np.array([np.maximum(flows["C_lp"][i * 48:(i + 1) * 48], 0).max() for i in range(len(dates))])
    cand = [i for i in range(len(dates)) if dates[i].month in (6, 7, 8) and dates[i].dayofweek < 5]
    day = max(cand, key=lambda i: peaks[i])
    sl = slice(day * 48, (day + 1) * 48)
    t = np.arange(48) * H
    panels = [("(A) Solver default", [("B_lp", C["B"], "--", "Individual batteries", "o"), ("C_lp", C["C"], "-", "Community battery", "s")]),
              ("(B) Same cost, single-maximum remedies", [("C_lp_minpeak", C["green"], "-", "Community, peak-minimising", "^"),
                                                         ("C_lp_cap", "#8a7a00", "--", "Community, import cap", "d")]),
              ("(C) Even-rate top-up", [("C_spread", C["purple"], "-", "Community, spread top-up", "x")])]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
    for ax, (ttl, series) in zip(axes, panels):
        ax.axvspan(cfg.TOU_PEAK_START_H, cfg.TOU_PEAK_END_H, color="#e6e6e6", lw=0, label="Peak-price window")
        ax.step(t, flows["A"][sl] / H, where="post", color="k", lw=2.2, label="No battery")
        for k, col, ls, lab, mkr in series:
            ax.step(t, flows[k][sl] / H, where="post", color=col, ls=ls, lw=1.3, label=lab, marker=mkr, ms=3.5, markevery=2)
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 6))
        ax.set_xlabel("Hour of day")
        ax.set_title(ttl, fontsize=9)
        ax.legend(frameon=False, fontsize=7, loc="upper left")
    axes[0].set_ylabel("Group import (kW)")
    fig.suptitle(f"{dates[day].strftime('%a %d %b %Y')}: 25 homes, 5 kWh per home, ToU tariff", fontsize=9, y=1.02)
    fig.tight_layout()
    save(fig, "fig1_day")


def fig_grid(main):
    m = T30.sel(main, n=25, cap=5.0)
    tars = ["flat", "tou", "tou_demand", "tou_demand24"]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.2))
    for c, pvs in enumerate(cfg.PV_SCALES):
        ax = axes[0, c]
        for j, (scn, col, lab) in enumerate((("B_lp", C["B"], "Individual (cost-minimising)"),
                                              ("C_lp", C["C"], "Community (cost-minimising)"))):
            r = paired(m, "peak_import_kw", pct_change(scn, "A"))
            r = r[r.pv == pvs].set_index("tariff").loc[tars].reset_index()
            ax.bar(np.arange(4) + (j - 0.5) * 0.38, r["mean"], 0.36, yerr=errbars(r), color=col, label=lab,
                   error_kw=dict(lw=0.8, capsize=2))
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(4))
        ax.set_xticklabels([TL[t].replace("demand (15-21h)","demand\n(15-21h)").replace("demand (24 h)","demand\n(24 h)") for t in tars], fontsize=6.5)
        ax.set_title(f"PV x{pvs:g}  ({pvs*1.68:.1f} kWp per home)", fontsize=9)
        if c == 0:
            ax.set_ylabel("Peak import vs no battery (%)")
            ax.legend(frameon=False, fontsize=7)
        ax = axes[1, c]
        order = [("A", C["A"], "No battery"), ("B_rule", C["Bagg"], "Individual, rule"), ("B_lp", C["B"], "Individual, cost-min."),
                 ("C_rule", C["sky"], "Community, rule"), ("C_lp", C["C"], "Community, cost-min.")]
        vals = []
        for scn, col, lab in order:
            r = paired(m[m.tariff == "tou"], "reverse_flow_h", lambda w, s=scn: w[s])
            r = r[r.pv == pvs].iloc[0]
            vals.append(r)
        ax.bar(range(5), [v["mean"] for v in vals], color=[o[1] for o in order],
               yerr=np.vstack([[v["mean"] - v["lo"] for v in vals], [v["hi"] - v["mean"] for v in vals]]),
               error_kw=dict(lw=0.8, capsize=2))
        ax.set_xticks(range(5))
        ax.set_xticklabels([o[2].replace(", ", "\n") for o in order], fontsize=7, rotation=35, ha="right")
        if c == 0:
            ax.set_ylabel("Reverse-flow hours per year\n(ToU tariff)")
    fig.tight_layout()
    save(fig, "fig2_grid")


def fig_ctrl(ctrl):
    ct = pd.read_csv(sim.ROOT / "results" / "v2_tables" / "controllers_all.csv")
    order = ["rule", "lp", "lp_random", "lp_minpeak", "lp_cap", "jit", "jit_cap", "spread", "mpc_react_perfect", "mpc_react_persist",
             "mpc_react_persist_cap"]
    names = {"rule": "Self-consumption\nrule", "lp": "Cost-min.\n(solver default)", "lp_random": "Cost-min.\nrandom tie-break",
             "lp_minpeak": "Cost-min.\npeak-minimising", "lp_cap": "Cost-min.\nimport cap", "jit": "Just-in-time\ntop-up",
             "jit_cap": "Just-in-time,\nimport cap", "spread": "Spread\ntop-up", "mpc_react_perfect": "Day-ahead,\nperfect forecast",
             "mpc_react_persist": "Day-ahead,\npersistence", "mpc_react_persist_cap": "Day-ahead, persist.,\nimport cap"}
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.6), sharex=True)
    for c, pvs in enumerate(cfg.PV_SCALES):
        for r, (qty, ylab) in enumerate((("d_peak_pct", "Peak import vs no battery (%)"), ("saving", "Annual saving (A$ thousand)"))):
            ax = axes[r, c]
            for j, who in enumerate(("B", "C")):
                sub = ct[(ct.pv == pvs) & (ct.quantity == qty) & (ct.who == who)].set_index("ctrl")
                sub = sub.reindex([o for o in order if o in sub.index])
                x = np.array([order.index(o) for o in sub.index]) + (j - 0.5) * 0.38
                sc = 1000.0 if qty == "saving" else 1.0
                e = errbars(sub) / sc
                ax.bar(x, sub["mean"] / sc, 0.36, yerr=e, color=C[who], error_kw=dict(lw=0.8, capsize=1.5),
                       label="Individual" if who == "B" else "Community")
            if qty == "d_peak_pct":
                ax.axhline(0, color="k", lw=0.6)
            if c == 0:
                ax.set_ylabel(ylab)
            if r == 0:
                ax.set_title(f"PV x{pvs:g}", fontsize=9)
    for ax in axes[1]:
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([names[o] for o in order], rotation=60, ha="right", fontsize=6.5)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "fig3_controllers")


def fig_mech(ctrl):
    ct = pd.read_csv(sim.ROOT / "results" / "v2_tables" / "controllers_all.csv")
    order = ["lp", "lp_random", "jit", "spread", "mpc_react_persist"]
    names = {"lp": "Cost-min.\n(solver default)", "lp_random": "Cost-min.\nrandom tie-break", "jit": "Just-in-time\ntop-up",
             "spread": "Spread\ntop-up", "mpc_react_persist": "Day-ahead,\npersistence"}
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.0))
    for ax, (qty, ylab) in zip(axes, (("coincidence", "Charging coincidence\nfactor (-)"),
                                      ("chg_peak_kw", "Peak simultaneous grid charging (kW)"),
                                      ("cycles", "Equivalent full cycles per year"))):
        for j, who in enumerate(("B", "C")):
            sub = ct[(ct.pv == 1.0) & (ct.quantity == qty) & (ct.who == who)].set_index("ctrl").reindex(order)
            ax.bar(np.arange(len(order)) + (j - 0.5) * 0.38, sub["mean"], 0.36, yerr=errbars(sub), color=C[who],
                   error_kw=dict(lw=0.8, capsize=1.5), label="Individual" if who == "B" else "Community")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([names[o] for o in order], rotation=60, ha="right", fontsize=6.5)
        ax.set_ylabel(ylab, fontsize=8)
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=7, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    save(fig, "fig4_mechanism")


def fig_remedies(main, ctrl):
    """Peak change and saving for the remedies, original PV size (left) and 3x PV (right)."""
    ct = pd.read_csv(sim.ROOT / "results" / "v2_tables" / "controllers_all.csv")
    m = T30.sel(main, n=25, cap=5.0)
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.2), sharex="col")
    items = [("Cost-min.\n(no remedy)", "ctrl", "lp"), ("Cost-min.,\nimport cap", "ctrl", "lp_cap"),
             ("Peak-minimising\ntie-break", "ctrl", "lp_minpeak"), ("Spread\ntop-up", "ctrl", "spread"),
             ("Day-ahead, persist.\n(no remedy)", "ctrl", "mpc_react_persist"), ("Day-ahead, persist.,\nimport cap", "ctrl", "mpc_react_persist_cap"),
             ("24 h demand\ncharge", "tariff", "tou_demand24")]
    for c, pvs in enumerate(cfg.PV_SCALES):
        for r, qty in enumerate(("d_peak_pct", "saving")):
            ax = axes[r, c]
            for j, who in enumerate(("B", "C")):
                xs, means, errs = [], [], []
                for i, (lab, kind, key) in enumerate(items):
                    if kind == "ctrl":
                        s = ct[(ct.pv == pvs) & (ct.quantity == qty) & (ct.who == who) & (ct.ctrl == key)]
                    else:
                        scn = f"{who}_lp"
                        fn = pct_change(scn, "A") if qty == "d_peak_pct" else diff("A", scn)
                        s = paired(m[m.tariff == key], "peak_import_kw" if qty == "d_peak_pct" else "bill_aud", fn)
                        s = s[s.pv == pvs]
                    if len(s) == 0:
                        continue
                    s = s.iloc[0]
                    sc = 1000.0 if qty == "saving" else 1.0
                    xs.append(i + (j - 0.5) * 0.38); means.append(s["mean"] / sc)
                    errs.append([(s["mean"] - s["lo"]) / sc, (s["hi"] - s["mean"]) / sc])
                ax.bar(xs, means, 0.36, yerr=np.array(errs).T, color=C[who], error_kw=dict(lw=0.8, capsize=1.5),
                       label="Individual" if who == "B" else "Community")
            if qty == "d_peak_pct":
                ax.axhline(0, color="k", lw=0.6)
            if c == 0:
                ax.set_ylabel("Peak import vs no battery (%)" if qty == "d_peak_pct" else "Saving (A$ thousand)\n(under each row's tariff)")
            if r == 0:
                ax.set_title(f"PV x{pvs:g}", fontsize=9)
    for ax in axes[1]:
        ax.set_xticks(range(len(items)))
        ax.set_xticklabels([i[0] for i in items], rotation=60, ha="right", fontsize=7)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "fig5_remedies")


def fig_decomp(main):
    sv = T30.table_savings(main)
    tars = ["flat", "tou", "tou_demand", "tou_demand24"]
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.0), sharey="row")
    for c, mode in enumerate(("rule", "lp")):
        for r, pvs in enumerate(cfg.PV_SCALES):
            ax = axes[r, c]
            g = sv[(sv["mode"] == mode) & (sv.pv == pvs) & (sv.n == 25) & (sv.cap == 5.0)]
            x = np.arange(len(tars))
            comp = {q: g[g.quantity == q].set_index("tariff").loc[tars, "mean"].to_numpy() / 1000 for q in
                    ("sav_B", "billing_effect", "pooling_effect")}
            ax.bar(x, comp["sav_B"], 0.6, color=C["B"], label="Individual batteries (B)")
            pos = comp["sav_B"].copy()
            ax.bar(x, comp["billing_effect"], 0.6, bottom=pos, color=C["Bagg"], label="+ billing as one connection")
            pos = pos + comp["billing_effect"]
            ax.bar(x, comp["pooling_effect"], 0.6, bottom=pos, color=C["C"], label="+ pooling the storage (= C)")
            ax.set_xticks(x)
            ax.set_xticklabels([TL[t].replace("demand (15-21h)","demand\n(15-21h)").replace("demand (24 h)","demand\n(24 h)") for t in tars], fontsize=6.5)
            if c == 0:
                ax.set_ylabel(f"PV x{pvs:g}\nSaving (A$ thousand per year)")
            if r == 0:
                ax.set_title("Rule-based control" if mode == "rule" else "Cost-minimising control", fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "fig6_decomposition")


def fig_econ(main):
    ec = T30.table_econ(main)
    ec = ec[(ec.n == 25)]
    tars = ["flat", "tou", "tou_demand", "tou_demand24"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)
    for ax, pvs in zip(axes, cfg.PV_SCALES):
        for j, (col, key, lab) in enumerate(((C["B"], "payback_B", f"Individual, A${cfg.BATT_COST_HOME_AUD_PER_KWH:.0f}/kWh"),
                                              (C["C"], "payback_C_arena", f"Community, A${cfg.BATT_COST_COMMUNITY_AUD_PER_KWH:,.0f}/kWh (ARENA)"))):
            g = ec[(ec.pv == pvs) & (ec["mode"] == "rule")].set_index("tariff").loc[tars]
            ax.bar(np.arange(len(tars)) + (j - 0.5) * 0.38, g[key].clip(upper=40), 0.36, color=col, label=lab)
        ax.axhline(cfg.BATT_LIFE_MAX_YEARS, color="k", lw=0.7, ls=":")
        ax.set_xticks(range(len(tars)))
        ax.set_xticklabels([TL[t].replace("demand (15-21h)","demand\n(15-21h)").replace("demand (24 h)","demand\n(24 h)") for t in tars], fontsize=6.5)
        ax.set_title(f"PV x{pvs:g}, rule-based control", fontsize=9)
    axes[0].set_ylabel("Simple payback (years)")
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    save(fig, "fig7_economics")


def fig_sens():
    s = pd.read_csv(sim.ROOT / "results" / "v2_tables" / "sensitivity.csv")
    s["tkw"] = s["tkw"].fillna("base")
    groups = [("Flat\nFiT", [("flat", "fit0"), ("flat", "base"), ("flat", "fit10")], ["0", "5", "10"]),
              ("ToU\nFiT", [("tou", "fit0"), ("tou", "base"), ("tou", "fit10")], ["0", "5", "10"]),
              ("ToU\nrates", [("tou", "base"), ("tou", "offpeak_c21.6-peak_c54")], ["AER", "market"]),
              ("Demand\n15-21h", [("tou_demand", "demand_aud0.2"), ("tou_demand", "base"), ("tou_demand", "demand_aud0.8")], ["0.2", "0.4", "0.8"]),
              ("Demand\n24 h", [("tou_demand24", "demand_aud0.2"), ("tou_demand24", "base"), ("tou_demand24", "demand_aud0.8")], ["0.2", "0.4", "0.8"]),
              ("Export\nlimit", [("tou_exportlimit", "export_limit3"), ("tou_exportlimit", "base"), ("tou_exportlimit", "export_limit5")], ["3", "1.5", "5"])]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.8))
    for c, pvs in enumerate(cfg.PV_SCALES):
        for r, (qty, scn_list) in enumerate((("d_peak_pct", [("B_lp", C["B"], "Individual"), ("C_lp", C["C"], "Community")]),
                                              ("sav", None))):
            ax = axes[r, c]
            xs, labs, gl = [], [], []
            pos = 0
            for gname, keys, vl in groups:
                gl.append((pos + (len(keys) - 1) / 2, gname))
                for (tn, kw), lab in zip(keys, vl):
                    xs.append((tn, kw, pos)); labs.append(lab); pos += 1
                pos += 1
            for tn, kw, p in xs:
                d = s[(s.tariff == tn) & (s.tkw == kw) & (s.pv == pvs)]
                if qty == "d_peak_pct":
                    for off, (scn, col, _) in zip((-0.2, 0.2), scn_list):
                        v = d[(d.quantity == "d_peak_pct") & (d.scenario == scn)]
                        if len(v):
                            ax.bar(p + off, v["mean"].iloc[0], 0.38, color=col)
                else:
                    for off, (q, mode, col) in zip((-0.27, 0, 0.27), (("sav_B", "rule", C["B"]), ("sav_C", "rule", C["sky"]), ("sav_C_storage", "rule", C["C"]))):
                        v = d[(d.quantity == q) & (d.scenario == mode)]
                        if len(v):
                            ax.bar(p + off, v["mean"].iloc[0] / 1000, 0.26, color=col)
            ax.set_xticks([x[2] for x in xs])
            ax.set_xticklabels(labs, fontsize=10, rotation=60)
            for gx, gn in gl:
                ax.text(gx, -0.42, gn, ha="center", va="top", fontsize=10, transform=ax.get_xaxis_transform())
            if qty == "d_peak_pct":
                ax.axhline(0, color="k", lw=0.6)
                ax.set_title(f"PV x{pvs:g}", fontsize=9)
            if c == 0:
                ax.set_ylabel("Peak vs no battery (%),\ncost-minimising" if qty == "d_peak_pct" else "Saving (A$ thousand per year),\nrule-based")
    h = [plt.Rectangle((0, 0), 1, 1, color=C["B"]), plt.Rectangle((0, 0), 1, 1, color=C["C"])]
    axes[0, 0].legend(h, ["Individual", "Community"], frameon=False, fontsize=9)
    h = [plt.Rectangle((0, 0), 1, 1, color=C[k]) for k in ("B",)] + [plt.Rectangle((0, 0), 1, 1, color=C["sky"]), plt.Rectangle((0, 0), 1, 1, color=C["C"])]
    axes[1, 0].legend(h, ["Individual", "Community, total", "Community, storage part"], frameon=False, fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.75)
    save(fig, "fig8_sensitivity")


def fig_crate():
    """Peak change against rated power (C-rate), ToU tariff: the rebound scales with the power of the block."""
    b = pd.read_csv(sim.ROOT / "results" / "v2_tables" / "batt.csv")
    ca = pd.read_csv(sim.ROOT / "results" / "v2_tables" / "controllers_all.csv")
    order, x = ["c_rate0.25", "base", "c_rate1"], np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharey=True)
    for ax, pvs in zip(axes, cfg.PV_SCALES):
        series = [("B_lp", C["B"], "o", "-", "Individual, cost-min. (solver default)"), ("C_lp", C["C"], "s", "-", "Community, cost-min. (solver default)"),
                  ("C_lp_minpeak", C["green"], "^", "--", "Community, peak-minimising tie-break"), ("C_rule", C["A"], "x", ":", "Community, self-consumption rule")]
        for scn, col, mkr, ls, lab in series:
            g = b[(b.pv == pvs) & (b.quantity == "d_peak_pct") & (b.scenario == scn)].set_index("batt")
            if scn == "C_lp_minpeak":
                base = ca[(ca.pv == pvs) & (ca.quantity == "d_peak_pct") & (ca.who == "C") & (ca.ctrl == "lp_minpeak")].iloc[0]
                g = pd.concat([g, pd.DataFrame([base[["mean", "lo", "hi"]]], index=["base"])])
            g = g.reindex(order).dropna(subset=["mean"])
            if len(g) == 0:
                continue
            xi = [order.index(i) for i in g.index]
            ax.errorbar(xi, g["mean"], yerr=errbars(g), color=col, marker=mkr, ls=ls, lw=1.3, ms=5, capsize=2, label=lab)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(["0.25C", "0.5C (base)", "1C"])
        ax.set_xlabel("Rated power / capacity")
        ax.set_title(f"PV x{pvs:g}", fontsize=9)
    axes[0].set_ylabel("Peak import vs no battery (%)")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, "fig10_crate")


def fig_tail():
    """Annual maximum against the 99th percentile of import: what single-maximum remedies do and do not control."""
    t = pd.read_csv(sim.ROOT / "results" / "v2_tables" / "rv_controllers.csv")
    tl = pd.read_csv(sim.ROOT / "results" / "v2_tables" / "tails.csv")
    pc = pd.read_csv(sim.ROOT / "results" / "v2_tables" / "peak_change.csv")
    items = [("rule", "Self-consumption\nrule"), ("lp", "Cost-min.\nsolver default"), ("lp_minpeak", "Cost-min.\npeak-minimising"),
             ("lp_cap", "Cost-min.\nimport cap"), ("spread", "Spread\ntop-up"), ("mpc_react_persist", "Day-ahead"),
             ("mpc_react_persist_capP100", "Day-ahead, cap at\nlast year's peak"), ("mpc_react_persist_dcap", "Day-ahead, cap at\nyesterday's max."),
             ("DEM24", "Cost-min., 24 h\ndemand charge")]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9), sharey=True)
    for ax, pvs in zip(axes, cfg.PV_SCALES):
        for j, (q, col, lab) in enumerate((("d_peak_pct", C["C"], "Annual maximum"), ("d_p99_pct", C["sky"], "99th percentile"))):
            rows = []
            for c, _ in items:
                if c == "DEM24":
                    src = pc if q == "d_peak_pct" else tl[tl.quantity == "d_p99_pct"]
                    g = src[(src.pv == pvs) & (src.tariff == "tou_demand24") & (src.scenario == "C_lp") & (src.n == 25) & (src.cap == 5.0)]
                else:
                    g = t[(t.pv == pvs) & (t.who == "C") & (t.ctrl == c) & (t.quantity == q)]
                rows.append(g.iloc[0][["mean", "lo", "hi"]])
            g = pd.DataFrame(rows).astype(float)
            ax.bar(np.arange(len(items)) + (j - 0.5) * 0.4, g["mean"], 0.38, yerr=errbars(g), color=col, label=lab, error_kw=dict(lw=0.8, capsize=1.5))
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(items)))
        ax.set_xticklabels([i[1] for i in items], rotation=60, ha="right", fontsize=7)
        ax.set_title(f"({'AB'[list(cfg.PV_SCALES).index(pvs)]}) PV x{pvs:g}", fontsize=9)
    axes[0].set_ylabel("Change in group import\nvs no battery (%)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, "fig5_tail")


def fig_fair():
    f = pd.read_csv(sim.ROOT / "results" / "fairness_summary.csv")
    rules = [("equal", "Equal split", "#bdbdbd"), ("prop_consumption", "Proportional to consumption", "#E69F00"),
             ("prop_bill", "Proportional to bill", "#D55E00"), ("shapley", "Shapley value", "#0072B2")]
    cases = [(t, pv) for t in ("flat", "tou", "tou_demand") for pv in cfg.PV_SCALES]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for j, (rk, rl, col) in enumerate(rules):
        v = [100 * f[(f.rule == rk) & (f.tariff == t) & (f.pv_scale == pv)].share_better_off.iloc[0] for t, pv in cases]
        ax.bar(np.arange(len(cases)) + (j - 1.5) * 0.2, v, 0.2, color=col, label=rl)
    nl = "\n"
    short = {"flat": "Flat", "tou": "ToU", "tou_demand": "ToU +" + nl + "demand"}
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([short[t] + nl + "PV x" + format(pv, "g") for t, pv in cases], fontsize=7.5)
    ax.set_ylabel("Homes better off than with\ntheir own battery (%)")
    ax.set_ylim(0, 108)
    ax.legend(frameon=False, fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    save(fig, "fig9_fairness")


if __name__ == "__main__":
    which = sys.argv[1:] or ["day", "grid", "ctrl", "mech", "remedies", "decomp", "econ", "sens", "fair", "crate", "tail"]
    main = load("main_") if any(w in which for w in ("grid", "remedies", "decomp", "econ")) else None
    ctrl = None   # controller figures read results/v2_tables/controllers_all.csv (written by 33_derived.py)
    for w in which:
        {"day": fig_day, "grid": lambda: fig_grid(main), "ctrl": lambda: fig_ctrl(ctrl), "mech": lambda: fig_mech(ctrl),
         "remedies": lambda: fig_remedies(main, ctrl), "decomp": lambda: fig_decomp(main), "econ": lambda: fig_econ(main),
         "sens": fig_sens, "fair": fig_fair, "crate": fig_crate, "tail": fig_tail}[w]()
