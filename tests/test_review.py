"""Tests added after the independent review: capped operational controllers, the interval estimator,
revenue-neutral tariffs and the peak-maximising member of the cost-optimal set."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import analysis_lib  # noqa: E402
import exp  # noqa: E402
import sim  # noqa: E402

SP = exp.spec(pv=1.0, tname="tou", cap=5.0)


def _group(k=4):
    d = exp.get_data(SP)
    idx = sim.make_partitions(d, 25, 4, seed=1025)[0][0][:k]
    return d, d["load"][idx], d["pv"][idx]


def test_daily_cap_respected():
    """With the daily cap the grid top-up never lifts import above yesterday's no-battery maximum
    (import can exceed it only where the load itself does)."""
    d, load, pv = _group()
    tar = exp.get_tariff(SP, d)
    net = (load - pv).sum(0)
    f, b, _ = sim.group_flow(load, pv, 5.0 * len(load), "mpc_react_persist_dcap", tar)
    T = len(net) // 48
    for day in range(2, T):
        cap = np.maximum(net[(day - 1) * 48:day * 48], 0).max()
        sl = slice(day * 48, (day + 1) * 48)
        over = (f[sl] > cap + 1e-6) & (b[sl] > 1e-9)
        assert not over.any(), day


def test_external_cap_respected():
    d, load, pv = _group()
    tar = exp.get_tariff(SP, d)
    cap = 0.8 * np.maximum((load - pv).sum(0), 0).max()
    f, b, _ = sim.group_flow(load, pv, 5.0 * len(load), "mpc_react_persist_capP100", tar, ext_cap=cap)
    assert not ((f > cap + 1e-6) & (b > 1e-9)).any()


def test_maxpeak_and_minpeak_same_cost():
    d, load, pv = _group(3)
    tar = exp.get_tariff(SP, d)
    bills = [sim.bill(sim.group_flow(load, pv, 15.0, c, tar)[0], tar) for c in ("lp", "lp_minpeak", "lp_maxpeak")]
    assert max(bills) - min(bills) < 0.05 + 1e-6 * abs(bills[0])
    pk = [np.maximum(sim.group_flow(load, pv, 15.0, c, tar)[0], 0).max() for c in ("lp_minpeak", "lp_maxpeak")]
    assert pk[0] <= pk[1] + 1e-9


def test_revenue_neutral_tariff():
    d = exp.get_data(SP)
    net = d["load"] - d["pv"]
    t0 = exp.get_tariff(SP, d)
    for name in ("tou_demand_rn", "tou_demand24_rn"):
        t = exp.get_tariff(exp.spec(pv=1.0, tname=name, cap=5.0), d)
        a = sum(sim.bill(x, t0) for x in net)
        b = sum(sim.bill(x, t) for x in net)
        assert abs(a - b) < 1e-6 * a


def test_estimator_single_partition():
    """est() with one partition equals the textbook t interval on that partition."""
    from scipy import stats
    rng = np.random.default_rng(0)
    v = rng.normal(3.0, 2.0, 11)
    m, lo, hi, G, K = analysis_lib.est(v, np.zeros(11))
    h = stats.t.ppf(0.975, 10) * v.std(ddof=1) / np.sqrt(11)
    assert abs(m - v.mean()) < 1e-12 and abs((hi - lo) / 2 - h) < 1e-12 and (G, K) == (11, 1)
