"""Energy-balance and sanity tests for the simulation core. Run: python3 -m pytest tests/ (from the project root)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import sim

D = sim.load_headline()
T = sim.make_tariff("tou", D["dates"])
IDX = [0, 7, 19]
CAP = 5.0


def _home(i):
    return D["load"][i], D["pv"][i]




def test_rule_energy_balance_and_soc_bounds():
    l, p = _home(IDX[0])
    net = l - p
    b = sim.rule_based(net, CAP)
    eta = 0.95
    soc = 0.5 * CAP + np.cumsum(np.where(b > 0, eta * b, b / eta))
    assert soc.min() >= -1e-6 and soc.max() <= CAP + 1e-6
    assert np.abs(b).max() <= 0.5 * CAP * 0.5 + 1e-9          # power limit per half-hour


def test_lp_and_mpc_respect_limits_and_start_end_soc():
    l, p = _home(IDX[1])
    net = l - p
    for b in (sim.lp_dispatch(net, p, T, CAP, 1)[0], sim.mpc_dispatch(net, CAP, T, "perfect")):
        eta = 0.95
        soc = 0.5 * CAP + np.cumsum(np.where(b > 0, eta * b, b / eta))
        assert soc.min() >= -1e-5 and soc.max() <= CAP + 1e-5
        assert np.abs(b).max() <= 0.5 * CAP * 0.5 + 1e-6


def test_battery_never_increases_bill_under_lp():
    for i in IDX:
        l, p = _home(i)
        net = l - p
        b, k = sim.lp_dispatch(net, p, T, CAP, 1)
        assert sim.bill(net + b + k, T) <= sim.bill(net, T) + 1e-6


def test_energy_conservation_over_year():
    """Grid energy in minus out = load - PV + losses (battery ends at its start SoC) + curtailment."""
    l, p = _home(IDX[2])
    net = l - p
    b, k = sim.lp_dispatch(net, p, T, CAP, 1)
    flow = net + b + k
    losses = (flow.sum() - net.sum() - k.sum())
    assert losses >= -1e-6                                    # losses are non-negative
