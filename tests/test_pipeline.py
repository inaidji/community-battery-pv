"""Further checks: billing, partitions, tie-break variants, import cap, controllers, Shapley efficiency,
and a small end-to-end smoke run. Run from the project root:  python -m pytest -q tests"""
import sys
from math import factorial
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import config as cfg
import sim

D = sim.load_headline()
T = sim.make_tariff("tou", D["dates"])
ETA = cfg.BATT_ETA_ONE_WAY


def net(i):
    return D["load"][i] - D["pv"][i]


def test_data_sanity():
    assert D["load"].shape == (299, 365 * 48) and D["pv"].shape == D["load"].shape
    assert not np.isnan(D["load"]).any() and not np.isnan(D["pv"]).any()
    assert (D["load"] >= 0).all() and (D["pv"] >= 0).all()
    assert len(set(D["ids"])) == 299


def test_tariff_window_and_prices():
    p = T.price.reshape(365, 48)
    dates = np.array(D["dates"], dtype="datetime64[D]")
    jan = int(np.flatnonzero(dates == np.datetime64("2013-01-15"))[0])
    apr = int(np.flatnonzero(dates == np.datetime64("2013-04-15"))[0])
    assert p[jan, 30] == cfg.TOU_PEAK_C and p[jan, 41] == cfg.TOU_PEAK_C        # 15:00 and 20:30
    assert p[jan, 29] == cfg.TOU_OFFPEAK_C and p[jan, 42] == cfg.TOU_OFFPEAK_C  # 14:30 and 21:00
    assert (p[apr] == cfg.TOU_OFFPEAK_C).all()                                  # no peak in April


def test_bill_against_hand_calculation():
    flow = np.zeros(365 * 48)
    flow[0] = 2.0          # 2 kWh import, 1 Jul 00:00, off-peak
    flow[1] = -1.0         # 1 kWh export
    expect = (2.0 * cfg.TOU_OFFPEAK_C - 1.0 * cfg.FIT_C_PER_KWH) / 100
    assert sim.bill(flow, T) == pytest.approx(expect)
    td = sim.make_tariff("tou_demand24", D["dates"])
    assert sim.bill(flow, td) == pytest.approx(expect + cfg.DEMAND_AUD_PER_KW_DAY * 31 * (2.0 / 0.5))


def test_partitions_do_not_overlap():
    for n in (10, 25, 50):
        for part in sim.make_partitions(D, n, 3, seed=1000 + n):
            homes = np.concatenate(part)
            assert len(homes) == len(set(homes)) and all(len(g) == n for g in part)


def test_energy_balance_and_soc_all_lp_variants():
    x, pv = net(5), D["pv"][5]
    for ctrl in ("lp", "lp_random", "lp_cap"):
        b, k = sim.dispatch(x, pv, 5.0, ctrl, T, seed=5)
        c, d = np.maximum(b, 0), np.maximum(-b, 0)
        soc = 2.5 + np.cumsum(ETA * c - d / ETA)
        assert soc.min() > -1e-5 and soc.max() < 5.0 + 1e-5 and abs(soc[-1] - 2.5) < 1e-4, ctrl
        assert np.abs(b).max() <= 0.5 * 5.0 * 0.5 + 1e-6 and (k == 0).all()


def test_tie_break_variants_cost_the_same_and_cap_holds():
    idx = np.arange(10)
    X, P = (D["load"][idx] - D["pv"][idx]).sum(0), D["pv"][idx].sum(0)
    bills, peaks = {}, {}
    for ctrl in ("lp", "lp_cap", "lp_minpeak"):
        b, k = sim.dispatch(X, P, 50.0, ctrl, T, n_homes=10)
        f = X + b + k
        bills[ctrl], peaks[ctrl] = sim.bill(f, T), np.maximum(f, 0).max()
    assert bills["lp_minpeak"] == pytest.approx(bills["lp"], rel=1e-5)
    assert bills["lp_cap"] == pytest.approx(bills["lp"], rel=1e-4)
    assert peaks["lp_cap"] <= np.maximum(X, 0).max() + 1e-6
    assert peaks["lp_minpeak"] <= peaks["lp"] + 1e-6


def test_export_limit_respected():
    te = sim.make_tariff("tou_exportlimit", D["dates"])
    x, pv = D["load"][7] - 3 * D["pv"][7], 3 * D["pv"][7]
    for ctrl in ("rule", "lp"):
        f, b, k = sim.home_flow(D["load"][7], pv, 5.0, ctrl, te)
        assert (-f).max() <= cfg.EXPORT_LIMIT_KW_PER_HOME * 0.5 + 1e-6 and (k >= -1e-9).all() and (k <= pv + 1e-9).all()


def test_persistence_controller_uses_no_same_day_information():
    """Changing the load of day d must not change the schedule of day d (only later days), except through
    reactive execution, which may only use the current half-hour."""
    x = net(11).copy()
    b1 = sim.mpc_dispatch(x, 5.0, T, "persist")
    x2 = x.copy()
    x2[48 * 10 + 20: 48 * 11] += 0.3                 # change the afternoon of day 10
    b2 = sim.mpc_dispatch(x2, 5.0, T, "persist")
    assert np.allclose(b1[: 48 * 11], b2[: 48 * 11])  # plan-execution: day 10 and earlier unchanged
    r1 = sim.mpc_dispatch(x, 5.0, T, "persist", execute="react")
    r2 = sim.mpc_dispatch(x2, 5.0, T, "persist", execute="react")
    assert np.allclose(r1[: 48 * 10 + 20], r2[: 48 * 10 + 20])   # nothing before the change is affected


def test_capped_rules_never_exceed_no_battery_peak():
    idx = np.arange(25)
    X, P = (D["load"][idx] - D["pv"][idx]).sum(0), D["pv"][idx].sum(0)
    for ctrl in ("jit_cap", "spread_cap", "mpc_react_persist_cap"):
        b, k = sim.dispatch(X, P, 125.0, ctrl, T, n_homes=25)
        assert np.maximum(X + b + k, 0).max() <= np.maximum(X, 0).max() + 1e-6, ctrl


def test_shapley_is_efficient_on_a_small_game():
    n = 4
    idx = [0, 1, 2, 3]
    load, pv = D["load"][idx], D["pv"][idx]
    billA = np.array([sim.bill(load[i] - pv[i], T) for i in range(n)])
    v = np.zeros(1 << n)
    for mask in range(1, 1 << n):
        mem = [i for i in range(n) if mask >> i & 1]
        f, _, _ = sim.group_flow(load[mem], pv[mem], 5.0 * len(mem), "rule", T)
        v[mask] = billA[mem].sum() - sim.bill(f, T)
    sh = np.zeros(n)
    for i in range(n):
        for mask in range(1 << n):
            if mask >> i & 1:
                continue
            s = bin(mask).count("1")
            sh[i] += factorial(s) * factorial(n - s - 1) / factorial(n) * (v[mask | 1 << i] - v[mask])
    assert sh.sum() == pytest.approx(v[-1])


def test_smoke_three_homes_all_scenarios():
    idx = np.array([0, 1, 2])
    load, pv = D["load"][idx], D["pv"][idx]
    flows = [sim.home_flow(load[i], pv[i], 5.0, "rule", T)[0] for i in range(3)]
    billA = sum(sim.bill(load[i] - pv[i], T) for i in range(3))
    billB = sum(sim.bill(f, T) for f in flows)
    billBagg = sim.bill(np.sum(flows, axis=0), T)
    fC, _, _ = sim.group_flow(load, pv, 15.0, "rule", T)
    billAagg = sim.bill((load - pv).sum(0), T)
    assert billAagg <= billA + 1e-9 and billBagg <= billB + 1e-9      # netting never raises an energy-only bill
    assert sim.bill(fC, T) < billAagg and billB < billA
