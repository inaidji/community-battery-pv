"""Coverage check for the interval of analysis_lib.est (the paper's Eq. for the 95 % interval).

Question: does a t interval on the groups of ONE non-overlapping partition cover the population mean of a
skewed group-level ratio (peak change) at the nominal rate when there are only 11 groups?
Design: the 299 Ausgrid homes are the population. The statistic is the peak change of scenario B (individual
batteries, cost-minimising or self-consumption rule) relative to A for a random group of 25 homes, computed from
the cached per-home flows. 'Truth' = mean over 20,000 random groups. Then 2,000 times: draw a random partition
of the homes into 11 disjoint groups of 25, build the interval from it as est() does for one partition, and
record whether it covers the truth. Also reported: coverage of a percentile bootstrap over the 11 groups.
Random groups are not postcode-ordered blocks, so this checks the estimator, not the spatial design.
Writes results/v2_tables/coverage.csv."""
import numpy as np
import pandas as pd
from scipy import stats

import exp
import sim

rng = np.random.default_rng(20261116)
sp = exp.spec(pv=1.0, tname="tou", cap=5.0)
d = exp.get_data(sp)
netA = d["load"] - d["pv"]
N, n, G = len(netA), 25, 11
rows = []
for ctrl in ("lp", "rule", "jit"):
    fl = np.load(exp.cache_path(sp, ctrl))["flow"].astype(np.float64)

    def stat(idx):
        return 100.0 * (np.maximum(fl[idx].sum(0), 0).max() / np.maximum(netA[idx].sum(0), 0).max() - 1.0)

    truth = np.mean([stat(rng.choice(N, n, replace=False)) for _ in range(20000)])
    cov_t = cov_b = 0
    width = []
    R = 2000
    for _ in range(R):
        perm = rng.permutation(N)[:G * n].reshape(G, n)
        v = np.array([stat(g) for g in perm])
        h = stats.t.ppf(0.975, G - 1) * v.std(ddof=1) / np.sqrt(G)
        cov_t += (v.mean() - h <= truth <= v.mean() + h)
        width.append(2 * h)
        bm = v[rng.integers(0, G, (1000, G))].mean(1)
        lo, hi = np.percentile(bm, [2.5, 97.5])
        cov_b += (lo <= truth <= hi)
    rows.append(dict(controller=ctrl, truth_pct=truth, coverage_t_pct=100 * cov_t / R, coverage_bootstrap_pct=100 * cov_b / R,
                     mean_width_pp=float(np.mean(width)), replications=R, groups=G, n=n))
out = pd.DataFrame(rows)
out.to_csv(sim.ROOT / "results" / "v2_tables" / "coverage.csv", index=False)
print(out.round(2).to_string(index=False))
