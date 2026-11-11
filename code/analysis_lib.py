"""Helpers for the v2 analysis: loading, paired within-group differences and confidence intervals.

Confidence intervals: groups inside a partition share no home, so they are (approximately) independent;
groups from different partitions overlap and are NOT. For every partition we compute the mean over its
groups and the standard error (sd / sqrt(groups)); the reported interval is the mean over partitions of the
point estimate +/- t(0.975, G-1) x the average standard error of one partition. It is therefore the interval
one would obtain from a single non-overlapping partition of the homes. Percentage changes are means of
per-group ratios (a paired, group-level statistic), not ratios of means.
"""
import numpy as np
import pandas as pd
from scipy import stats

import sim

R = sim.ROOT / "results" / "v2"
KEYS = ["year", "loadvar", "pv", "tariff", "tkw", "cap", "n", "nopv_every"]


def load(prefix):
    fs = sorted(R.glob(f"{prefix}*.csv"))
    if not fs:
        raise FileNotFoundError(prefix)
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df["tkw"] = df["tkw"].fillna("base")
    return df


def wide(df, metric):
    keys = [k for k in KEYS if k in df.columns]
    w = df.pivot_table(index=keys + ["part", "group"], columns="scenario", values=metric, aggfunc="first")
    return w.reset_index()


def est(v, part):
    """Mean over partitions and a single-partition 95% interval. Returns (mean, lo, hi, groups, partitions)."""
    v = np.asarray(v, float)
    part = np.asarray(part)
    ms, ses, Gs = [], [], []
    for k in np.unique(part):
        x = v[part == k]
        x = x[~np.isnan(x)]
        if len(x) < 2:
            continue
        ms.append(x.mean())
        ses.append(x.std(ddof=1) / np.sqrt(len(x)))
        Gs.append(len(x))
    if not ms:
        return np.nan, np.nan, np.nan, 0, 0
    G = float(np.mean(Gs))
    m = float(np.mean(ms))
    h = stats.t.ppf(0.975, max(G - 1, 1)) * float(np.mean(ses))
    return m, m - h, m + h, int(round(G)), len(ms)


def paired(df, metric, fn, by=None):
    """fn(wide_df) -> Series of per-group values; returns mean/CI per configuration."""
    w = wide(df, metric)
    w["_v"] = fn(w)
    by = by or [k for k in KEYS if k in w.columns]
    rows = []
    for key, g in w.groupby(by, dropna=False):
        m, lo, hi, G, K = est(g["_v"].to_numpy(), g["part"].to_numpy())
        key = key if isinstance(key, tuple) else (key,)
        rows.append(dict(zip(by, key), mean=m, lo=lo, hi=hi, groups=G, partitions=K))
    return pd.DataFrame(rows)


def pct_change(a, b):
    """100 * (a / b - 1) as a function on wide columns."""
    return lambda w: 100.0 * (w[a] / w[b] - 1.0)


def diff(a, b):
    return lambda w: w[a] - w[b]


def fmt_ci(r, digits=1, sign=False):
    f = f"{{:+.{digits}f}}" if sign else f"{{:.{digits}f}}"
    return f"{f.format(r['mean'])} [{f.format(r['lo'])}, {f.format(r['hi'])}]"
