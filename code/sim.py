"""Core library: data loading, tariffs, battery dispatch (rule-based and perfect-foresight LP),
billing and grid metrics. Energy in kWh per half hour, prices in c/kWh, costs in A$."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, identity, hstack, vstack, diags

import config as cfg

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "extracted"
CACHE = ROOT / "data" / "processed" / "headline_2012_13.npz"
H = cfg.DT_H


# ---------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------
YEARS = {  # key -> (csv file, first day, number of days)
    "2012-13": ("2012-2013 Solar home electricity data v2.csv", "2012-07-01", 365),
    "2011-12": ("2011-2012 Solar home electricity data v2.csv", "2011-07-01", 366),   # leap year
    "2010-11": ("2010-2011 Solar home electricity data.csv", "2010-07-01", 365),
}


def load_headline(year="2012-13", include_cl=True):
    """Dict with ids, postcode, kwp, load (N,T), pv (N,T), dates. kWh per half hour.

    load = GC + CL (CL is zero for homes without controlled load). Only customers with a full
    year of GC and GG are kept. Default year 2012-13 is the headline year.
    """
    fname, first_day, n_days = YEARS[year]
    cache = CACHE if year == "2012-13" else CACHE.with_name(f"headline_{year.replace('-', '_')}.npz")
    if not include_cl:
        cache = cache.with_name(cache.stem + "_gconly.npz")
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}

    df = pd.read_csv(RAW / fname, skiprows=1, keep_default_na=False, na_values=[""], low_memory=False)
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=True)
    slots = [c for c in df.columns if ":" in c]
    dates = pd.date_range(first_day, periods=n_days, freq="D")
    meta = df.groupby("Customer")[["Generator Capacity", "Postcode"]].first()

    def block(cat):
        sub = df[df["Consumption Category"] == cat].set_index(["Customer", "date"])[slots]
        sub = sub.apply(pd.to_numeric, errors="coerce")
        out = {}
        for cust, g in sub.groupby(level=0):
            out[cust] = g.droplevel(0).reindex(dates).to_numpy()  # (days, 48)
        return out

    gc, gg, cl = block("GC"), block("GG"), block("CL")
    ids, load, pv = [], [], []
    for cust in sorted(gc):
        a, b = gc[cust], gg.get(cust)
        if b is None or np.isnan(a).any() or np.isnan(b).any():
            continue
        c = cl.get(cust)
        c = np.zeros_like(a) if (c is None or not include_cl) else np.nan_to_num(c, nan=0.0)
        ids.append(cust)
        load.append((a + c).reshape(-1))
        pv.append(b.reshape(-1))
    ids = np.array(ids)
    out = dict(ids=ids, postcode=meta.loc[ids, "Postcode"].to_numpy(),
               kwp=meta.loc[ids, "Generator Capacity"].to_numpy(),
               load=np.array(load), pv=np.array(pv), dates=dates.to_numpy())
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def make_groups(data, n, n_groups, seed=0):
    """Neighbourhood groups: contiguous blocks of n homes in postcode order, random start."""
    rng = np.random.default_rng(seed)
    order = np.lexsort((data["ids"], data["postcode"]))
    starts = rng.choice(len(order) - n + 1, size=n_groups, replace=False)
    return [order[s:s + n] for s in starts]


def make_partitions(data, n, n_partitions, seed=0):
    """Non-overlapping neighbourhood groups: each partition cuts the postcode-ordered homes into
    consecutive blocks of n homes after a random offset (the remainder is dropped). Returns a list of
    partitions, each a list of index arrays. Groups inside a partition share no home."""
    rng = np.random.default_rng(seed)
    order = np.lexsort((data["ids"], data["postcode"]))
    offsets = rng.choice(n, size=n_partitions, replace=False)
    out = []
    for o in offsets:
        k = (len(order) - o) // n
        out.append([order[o + i * n:o + (i + 1) * n] for i in range(k)])
    return out


# ---------------------------------------------------------------------------------
# Tariffs
# ---------------------------------------------------------------------------------
@dataclass
class Tariff:
    name: str
    price: np.ndarray            # (T,) c/kWh buy price
    fit: float                   # c/kWh export credit
    peak_mask: np.ndarray        # (T,) bool, demand-charge window
    month: np.ndarray            # (T,) 0..11 calendar month index
    days_in_month: np.ndarray    # (12,)
    demand_c_per_kw_day: float   # 0 = no demand charge
    export_limit_kw_per_home: float | None


def make_tariff(name, dates, fit=None, demand_aud=None, export_limit=None, peak_c=None, offpeak_c=None):
    dates = pd.DatetimeIndex(dates)
    n_days = len(dates)
    start_h = np.arange(cfg.SLOTS) * H
    month_day = dates.month.to_numpy()
    month = np.repeat(month_day - 1, cfg.SLOTS)
    days_in_month = np.array([(month_day == m + 1).sum() for m in range(12)], dtype=float)

    in_window = (start_h >= cfg.TOU_PEAK_START_H) & (start_h < cfg.TOU_PEAK_END_H)
    peak_months = np.isin(month_day, cfg.TOU_PEAK_MONTHS)
    tou_peak = (peak_months[:, None] & in_window[None, :]).reshape(-1)
    window_all = np.tile(in_window, n_days)

    if name == "flat":
        price = np.full(n_days * cfg.SLOTS, cfg.FLAT_C_PER_KWH)
    else:
        pk, op = (cfg.TOU_PEAK_C if peak_c is None else peak_c), (cfg.TOU_OFFPEAK_C if offpeak_c is None else offpeak_c)
        price = np.where(tou_peak, pk, op)
    demand = 0.0
    if name in ("tou_demand", "tou_demand24"):
        demand = 100.0 * (cfg.DEMAND_AUD_PER_KW_DAY if demand_aud is None else demand_aud)
    if name == "tou_demand24":
        window_all = np.ones_like(window_all, dtype=bool)         # demand window = every half-hour
    limit = None
    if name == "tou_exportlimit":
        limit = cfg.EXPORT_LIMIT_KW_PER_HOME if export_limit is None else export_limit
    return Tariff(name, price, cfg.FIT_C_PER_KWH if fit is None else fit,
                  window_all, month, days_in_month, demand, limit)


# ---------------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------------
def _batt(cap):
    return cap, cfg.BATT_C_RATE * cap * H, cfg.BATT_ETA_ONE_WAY


def rule_based(net, cap):
    """Self-consumption: charge from surplus, discharge to cover deficit.
    net = load - pv (kWh/slot). Returns b (grid-side, +charge / -delivered)."""
    cap, pmax, eta = _batt(cap)
    b = np.zeros(len(net))
    soc = 0.5 * cap
    for t, n in enumerate(net):
        if n < 0:
            c = min(-n, pmax, (cap - soc) / eta)
            soc += eta * c
            b[t] = c
        else:
            out = min(n, pmax, soc * eta)
            soc -= out / eta
            b[t] = -out
    return b


def curtail(flow, limit_kw):
    """Curtail PV so exports never exceed limit_kw. Returns (flow_after, curtailed_kwh)."""
    if limit_kw is None:
        return flow, np.zeros_like(flow)
    k = np.maximum(-flow - limit_kw * H, 0.0)
    return flow + k, k


def _build_lp(net, pv, tariff, cap, n_homes, price=None, peak_cap_kwh=None, charge_frac=1.0):
    """Assemble the year-long dispatch LP. Variables x = [c, d, s, g, e, k (each T), m (12 monthly peaks)]."""
    cap, pmax, eta = _batt(cap)
    T = len(net)
    I = identity(T, format="csr")
    Z = coo_matrix((T, T)).tocsr()
    Zm = coo_matrix((T, 12)).tocsr()
    D = identity(T, format="csr") - diags([np.ones(T - 1)], [-1], shape=(T, T), format="csr")
    rows = [
        hstack([-eta * I, I / eta, D, Z, Z, Z, Zm], format="csr"),    # SoC dynamics
        hstack([-I, I, Z, I, -I, -I, Zm], format="csr"),              # balance
        coo_matrix(([1.0], ([0], [3 * T - 1])), shape=(1, 6 * T + 12)).tocsr(),  # s_T
    ]
    A_eq = vstack(rows, format="csr")
    b_eq = np.concatenate([[0.5 * cap], np.zeros(T - 1), net, [0.5 * cap]])

    A_ub = b_ub = None
    if tariff.demand_c_per_kw_day > 0:
        idx = np.flatnonzero(tariff.peak_mask)
        r = np.arange(len(idx))
        g_part = coo_matrix((np.ones(len(idx)), (r, idx)), shape=(len(idx), T))
        m_part = coo_matrix((np.full(len(idx), -H), (r, tariff.month[idx])), shape=(len(idx), 12))
        A_ub = hstack([coo_matrix((len(idx), 3 * T)), g_part, coo_matrix((len(idx), 2 * T)), m_part],
                      format="csr")
        b_ub = np.zeros(len(idx))

    p = tariff.price if price is None else price
    cost = np.concatenate([np.zeros(3 * T), p, -np.full(T, tariff.fit), np.zeros(T),
                           tariff.demand_c_per_kw_day * tariff.days_in_month])
    lim = tariff.export_limit_kw_per_home
    ub_e = np.inf if lim is None else lim * n_homes * H
    ub_k = np.zeros(T) if lim is None else np.maximum(pv, 0.0)
    ub_g = None if peak_cap_kwh is None else peak_cap_kwh
    bounds = ([(0, pmax * charge_frac)] * T + [(0, pmax)] * T + [(cfg.BATT_SOC_MIN * cap, cap)] * T
              + [(0, ub_g)] * T + [(0, ub_e)] * T + list(zip(np.zeros(T), ub_k)) + [(0, None)] * 12)
    return cost, A_ub, b_ub, A_eq, b_eq, bounds, T


def lp_dispatch(net, pv, tariff, cap, n_homes=1, tiebreak=None, seed=0, peak_cap_kwh=None,
                charge_frac=1.0):
    """Perfect-foresight cost-minimising dispatch over the year (cyclic SoC).

    Balance: g - e - c + d - k = net. Returns (b, k) with b = c - d (grid side).

    The cost-minimising schedule is usually not unique (equal off-peak prices), so the solution depends
    on tie-breaking. tiebreak: None = solver vertex (what HiGHS returns); 'random' = tiny random
    perturbation of the prices (each battery picks its own cheapest slots); 'minpeak' = among schedules
    whose cost is within 1e-6 of the optimum, the one with the smallest peak import. peak_cap_kwh caps
    import per half-hour; charge_frac limits the charge power to a fraction of the rated power.
    """
    T = len(net)
    price = None
    if tiebreak == "random":
        rng = np.random.default_rng(seed)
        price = tariff.price * (1.0 + rng.uniform(-0.01, 0.01, T))
    cost, A_ub, b_ub, A_eq, b_eq, bounds, T = _build_lp(net, pv, tariff, cap, n_homes, price,
                                                        peak_cap_kwh, charge_frac)
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    x = res.x
    if tiebreak == "minpeak":
        true_cost = res.fun
        nvar = len(cost)
        # stage 2: minimise z >= g_t subject to cost <= optimum (+ tolerance)
        r = np.arange(T)
        gz = hstack([coo_matrix((T, 3 * T)), identity(T), coo_matrix((T, 2 * T)), coo_matrix((T, 12)),
                     coo_matrix((-np.ones(T), (r, np.zeros(T, dtype=int))), shape=(T, 1))], format="csr")
        cost_row = coo_matrix(np.concatenate([cost, [0.0]]).reshape(1, -1))
        A_ub2 = vstack([gz, cost_row] if A_ub is None else
                       [hstack([A_ub, coo_matrix((A_ub.shape[0], 1))]), gz, cost_row], format="csr")
        b_ub2 = np.concatenate([[] if b_ub is None else b_ub, np.zeros(T),
                                [true_cost + 1e-6 * abs(true_cost) + 1e-6]])
        A_eq2 = hstack([A_eq, coo_matrix((A_eq.shape[0], 1))], format="csr")
        c2 = np.zeros(nvar + 1)
        c2[-1] = 1.0
        res2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq, bounds=bounds + [(0, None)],
                       method="highs")
        if res2.status == 0:
            x = res2.x[:nvar]
    return x[:T] - x[T:2 * T], x[5 * T:6 * T]


CONTROLLERS = ("rule", "lp", "lp_random", "lp_minpeak", "lp_cap", "jit", "spread", "mpc_perfect", "mpc_persist")


def heuristic(net, cap, tariff, style):
    """Price-aware rules without optimisation. Self-consumption always (charge from surplus PV, discharge
    to load); in addition, on days with a peak-price window the battery is topped up from the grid before
    the window opens. style 'jit': charge at full power as late as possible so it is full when the window
    opens (what the cost-optimal LP vertex does); 'spread': from 10:00 charge at an even rate so it is full
    when the window opens."""
    cap_, pmax, eta = _batt(cap)
    spd = cfg.SLOTS
    T = len(net)
    price = tariff.price.reshape(T // spd, spd)
    b = np.zeros(T)
    soc = 0.5 * cap_
    for day in range(T // spd):
        pr = price[day]
        hi = np.flatnonzero(pr > pr.min() + 1e-9)
        t_pk = hi[0] if len(hi) else None
        for s_ in range(spd):
            t = day * spd + s_
            n = net[t]
            ch = dis = 0.0
            if n < 0:
                ch = min(-n, pmax, (cap_ - soc) / eta)
            else:
                dis = min(n, pmax, soc * eta)
            if t_pk is not None and s_ < t_pk:
                left = t_pk - s_                                    # slots left before the window, this one included
                need = (cap_ - (soc + eta * ch - dis / eta)) / eta  # grid-side energy still needed
                room = pmax - ch
                if style == "jit":
                    if need > pmax * (left - 1) + 1e-9:
                        ch += min(room, need)
                elif style == "spread" and s_ >= cfg.SPREAD_START_SLOT:
                    ch += min(room, need / left)
            soc += eta * ch - dis / eta
            b[t] = ch - dis
    return b


def mpc_dispatch(net, cap, tariff, forecast="perfect", wear_c=0.5, end_frac=0.5):
    """Rolling-horizon day-ahead controller. Each day a 48-slot LP is solved on a forecast of that day's
    net load (forecast='perfect' = the actual net load; 'persist' = yesterday's actual net load) with the
    known prices and the end-of-day state of charge at least half full; the planned schedule is then
    executed against the actual load. A small throughput cost (wear_c, c/kWh) stops the optimiser from making
    cost-neutral charge/discharge cycles that turn into exports when the forecast is wrong. Demand charges
    and export limits are not represented."""
    cap_, pmax, eta = _batt(cap)
    n = cfg.SLOTS
    T = len(net)
    I = np.eye(n)
    Lw = np.tril(np.ones((n, n)))
    A_eq = np.hstack([-I, I, I, -I])                                  # g - e - c + d = f
    A_ub = np.vstack([np.hstack([eta * Lw, -Lw / eta, np.zeros((n, 2 * n))]),
                      np.hstack([-eta * Lw, Lw / eta, np.zeros((n, 2 * n))]),
                      np.hstack([-eta * np.ones((1, n)), np.ones((1, n)) / eta, np.zeros((1, 2 * n))])])
    bounds = [(0, pmax)] * n + [(0, pmax)] * n + [(0, None)] * n + [(0, None)] * n
    b = np.zeros(T)
    soc = 0.5 * cap_
    for day in range(T // n):
        sl = slice(day * n, (day + 1) * n)
        f = net[sl] if (forecast == "perfect" or day == 0) else net[(day - 1) * n:day * n]
        cost = np.concatenate([np.full(2 * n, wear_c), tariff.price[sl], np.full(n, -tariff.fit)])
        b_ub = np.concatenate([np.full(n, cap_ - soc), np.full(n, soc), [soc - end_frac * cap_]])
        res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=f, bounds=bounds, method="highs")
        if res.status != 0:
            raise RuntimeError(f"MPC LP failed on day {day}: {res.message}")
        c, d = res.x[:n], res.x[n:2 * n]
        b[sl] = c - d
        soc += float((eta * c - d / eta).sum())
    return b


def dispatch(net, pv, cap, ctrl, tariff, n_homes=1, seed=0):
    """Battery schedule b (grid side, + charge) and curtailed PV k for one battery at one billing point."""
    zero = np.zeros_like(net)
    if ctrl == "rule":
        return rule_based(net, cap), zero
    if ctrl == "lp":
        return lp_dispatch(net, pv, tariff, cap, n_homes)
    if ctrl == "lp_random":
        return lp_dispatch(net, pv, tariff, cap, n_homes, tiebreak="random", seed=seed)
    if ctrl == "lp_minpeak":
        return lp_dispatch(net, pv, tariff, cap, n_homes, tiebreak="minpeak")
    if ctrl == "lp_cap":      # import never above the no-battery peak at this billing point
        return lp_dispatch(net, pv, tariff, cap, n_homes, peak_cap_kwh=float(np.maximum(net, 0).max()))
    if ctrl in ("jit", "spread"):
        return heuristic(net, cap, tariff, ctrl), zero
    if ctrl in ("mpc_perfect", "mpc_persist"):
        return mpc_dispatch(net, cap, tariff, "perfect" if ctrl == "mpc_perfect" else "persist"), zero
    raise ValueError(ctrl)


LP_CTRLS = ("lp", "lp_random", "lp_minpeak", "lp_cap")      # these handle curtailment inside the LP


def _to_flow(net, b, k, ctrl, tariff, limit_kw):
    if ctrl in LP_CTRLS:
        return net + b + k, k          # curtailing PV raises net demand: g - e = net + c - d + k
    flow, k = curtail(net + b, limit_kw)
    return flow, k


def home_flow(load, pv, cap, mode, tariff, seed=0):
    """Grid exchange of one home (T,) with its own battery (cap=0 -> none). + = import. mode = controller."""
    net = load - pv
    if cap == 0:
        flow, k = curtail(net, tariff.export_limit_kw_per_home)
        return flow, np.zeros_like(net), k
    b, k = dispatch(net, pv, cap, mode, tariff, n_homes=1, seed=seed)
    flow, k = _to_flow(net, b, k, mode, tariff, tariff.export_limit_kw_per_home)
    return flow, b, k


def group_flow(load, pv, cap_total, mode, tariff, seed=0):
    """One community battery on the aggregate. The export limit scales with group size (n x per-home)."""
    n = load.shape[0]
    net = (load - pv).sum(axis=0)
    lim = tariff.export_limit_kw_per_home
    lim_group = None if lim is None else lim * n
    if cap_total == 0:
        flow, k = curtail(net, lim_group)
        return flow, np.zeros_like(net), k
    b, k = dispatch(net, pv.sum(axis=0), cap_total, mode, tariff, n_homes=n, seed=seed)
    flow, k = _to_flow(net, b, k, mode, tariff, lim_group)
    return flow, b, k


# ---------------------------------------------------------------------------------
# Billing and metrics
# ---------------------------------------------------------------------------------
def bill(flow, tariff):
    """Annual A$ bill at one billing point. flow (T,) kWh, + = import."""
    imp = np.maximum(flow, 0.0)
    exp = np.maximum(-flow, 0.0)
    energy = (imp * tariff.price - exp * tariff.fit).sum() / 100.0
    demand = 0.0
    if tariff.demand_c_per_kw_day > 0:
        for m in range(12):
            sel = tariff.peak_mask & (tariff.month == m)
            if sel.any():
                demand += tariff.demand_c_per_kw_day * tariff.days_in_month[m] * imp[sel].max() / H / 100.0
    return energy + demand


def grid_metrics(flow_agg, pv_total, load_total, curtailed_kwh=0.0):
    """Metrics at the shared transformer, from the aggregate flow (T,)."""
    imp = np.maximum(flow_agg, 0.0)
    exp = np.maximum(-flow_agg, 0.0)
    return dict(
        import_kwh=imp.sum(), export_kwh=exp.sum(), curtailed_kwh=curtailed_kwh,
        peak_import_kw=imp.max() / H, peak_export_kw=exp.max() / H,
        p99_import_kw=np.percentile(imp, 99) / H,
        reverse_flow_h=(exp > 1e-9).sum() * H,
        self_consumption=(pv_total - exp.sum() - curtailed_kwh) / pv_total,
        self_sufficiency=1 - imp.sum() / load_total,
    )


def cycles_per_year(b, cap):
    """Equivalent full cycles/year from grid-side charge energy."""
    return np.maximum(b, 0).sum() * cfg.BATT_ETA_ONE_WAY / cap if cap > 0 else 0.0
