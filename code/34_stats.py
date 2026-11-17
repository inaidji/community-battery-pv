"""Paired significance tests for the headline contrasts (PLAN 3.6).

Unit of analysis: a neighbourhood group. Groups of different partitions share homes, so tests are run
inside each non-overlapping partition (11 groups for n=25) with the exact two-sided Wilcoxon signed-rank
test; we report the LARGEST p-value over the partitions (conservative) and apply the Holm correction
across all contrasts of the family. Contrasts that hold by construction
(a capped or peak-minimised annual maximum against the uncapped one) are not tested. With 11 pairs the smallest attainable two-sided p is 2/2^11 = 0.00098.
Also reported: a percentile bootstrap interval (10 000 resamples of groups within a partition, averaged
over partitions) for the mean paired difference.
Output: results/v2_tables/stats_tests.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import sim
from analysis_lib import load, load_controllers, wide

OUT = sim.ROOT / "results" / "v2_tables"
RNG = np.random.default_rng(20261115)


def wilcoxon_p(x):
    """Two-sided Wilcoxon signed-rank p-value. Exact distribution when there are no zero differences and no ties
    (scipy's exact mode requires this); all-zero differences give p = 1; otherwise zeros are dropped (Wilcoxon's
    rule) and the normal approximation is used. The same rule is used for both datasets."""
    x = np.asarray(x, float)
    if np.allclose(x, 0):
        return 1.0
    nz = x[x != 0]
    exact = len(nz) == len(x) and len(np.unique(np.abs(nz))) == len(nz) and len(x) <= 25
    return float(wilcoxon(x, alternative="two-sided", mode="exact" if exact else "approx").pvalue)


def contrast(df, metric, a, b, label, family, pct=False):
    """Test a vs b (paired by group). pct: difference of percentage changes relative to scenario A."""
    w = wide(df, metric)
    if pct:
        va, vb = 100 * (w[a] / w["A"] - 1), (100 * (w[b] / w["A"] - 1) if b != "A" else 0 * w["A"])
    else:
        va, vb = w[a], w[b]
    d = (va - vb).to_numpy()
    ps, los, his, ms = [], [], [], []
    for k in np.unique(w["part"]):
        x = d[w["part"].to_numpy() == k]
        x = x[~np.isnan(x)]
        ps.append(wilcoxon_p(x))
        bs = RNG.choice(x, size=(10000, len(x)), replace=True).mean(axis=1)
        los.append(np.percentile(bs, 2.5)); his.append(np.percentile(bs, 97.5)); ms.append(x.mean())
    return dict(family=family, contrast=label, mean_diff=np.mean(ms), boot_lo=np.mean(los), boot_hi=np.mean(his),
                p_max=max(ps), p_min=min(ps), partitions=len(ps), groups=int(len(d) / len(ps)))


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


if __name__ == "__main__":
    main = load("main_")
    m = main[(main.n == 25) & (main.cap == 5.0)]
    RV = load("rv_ctrl_")
    rows = []
    for pv in (1.0, 3.0):
        tou = m[(m.tariff == "tou") & (m.pv == pv)]
        d24 = m[(m.tariff == "tou_demand24") & (m.pv == pv)]
        fam = f"pv{pv:g}"
        rv = RV[RV.pv == pv]
        P, Q, B = "peak_import_kw", "p99_import_kw", "bill_aud"
        rows += [
            contrast(tou, P, "C_lp", "A", "Cpk|annual maximum: community, solver default, vs no battery [pp]", fam, pct=True),
            contrast(tou, P, "B_lp", "A", "Bpk|annual maximum: individual, solver default, vs no battery [pp]", fam, pct=True),
            contrast(tou, P, "C_lp", "B_lp", "CvB|annual maximum: community vs individual, solver default [pp]", fam, pct=True),
            contrast(tou, Q, "C_lp", "A", "Cp99|99th percentile: community, solver default, vs no battery [pp]", fam, pct=True),
            contrast(rv, Q, "C_lp_minpeak", "A", "minp99|99th percentile: community, peak-minimising, vs no battery [pp]", fam, pct=True),
            contrast(rv, Q, "C_lp_cap", "A", "capp99|99th percentile: community, cost-min. with import cap, vs no battery [pp]", fam, pct=True),
            contrast(rv, Q, "C_spread", "A", "sprp99|99th percentile: community, spread top-up, vs no battery [pp]", fam, pct=True),
            contrast(rv, P, "C_spread", "C_lp", "sprVdef|annual maximum: spread top-up vs solver default (community) [pp]", fam, pct=True),
            contrast(rv, Q, "C_mpc_react_persist", "A", "dap99|99th percentile: community, day-ahead (persistence), vs no battery [pp]", fam, pct=True),
            contrast(rv, Q, "C_mpc_react_persist_capP100", "A", "dacapp99|99th percentile: day-ahead, cap at last year's peak, vs no battery [pp]", fam, pct=True),
            contrast(rv, Q, "C_mpc_react_persist_dcap", "A", "dcapp99|99th percentile: day-ahead, daily cap, vs no battery [pp]", fam, pct=True),
            contrast(rv, B, "C_mpc_react_persist_capP100", "C_mpc_react_persist", "capBill|bill: day-ahead, cap at last year's peak vs no cap [A$]", fam),
            contrast(rv, B, "C_mpc_react_persist_dcap", "C_mpc_react_persist", "dcapBill|bill: day-ahead, daily cap vs no cap [A$]", fam),
            contrast(d24, P, "C_lp", "A", "dem24|annual maximum: community vs no battery, 24 h demand charge [pp]", fam, pct=True),
            contrast(d24, Q, "C_lp", "A", "dem24p99|99th percentile: community vs no battery, 24 h demand charge [pp]", fam, pct=True),
            contrast(tou, "reverse_flow_h", "C_lp", "B_lp", "rev|reverse-flow hours: community vs individual [h]", fam),
            contrast(tou, B, "C_lp", "B_lp", "billCvB|bill: community vs individual, solver default [A$]", fam),
            contrast(tou, B, "Bagg_lp", "B_lp", "billing|bill: billing effect (B_agg vs B) [A$]", fam),
            contrast(tou, B, "C_lp", "Bagg_lp", "pooling|bill: pooling effect (C vs B_agg) [A$]", fam),
        ]
    out = pd.DataFrame(rows)
    out["id"] = out.contrast.str.split("|").str[0]
    out["contrast"] = out.contrast.str.split("|").str[1]
    out["p_holm"] = np.nan
    for f, g in out.groupby("family"):
        out.loc[g.index, "p_holm"] = holm(g["p_max"].to_numpy())
    out.to_csv(OUT / "stats_tests.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 80)
    print(out.round(4).to_string(index=False))
