"""Experiment pipeline (v2): cached per-home dispatch (stage 1) and group evaluation (stage 2).

Scenarios per group:
  A       no battery, homes billed individually
  A_agg   no battery, group billed as one connection
  B_<c>   individual batteries (controller c), homes billed individually
  Bagg_<c> the same individual batteries, group billed as one connection (separates ownership from billing)
  C_<c>   one community battery (controller c), group billed as one connection
Groups come from non-overlapping partitions (sim.make_partitions), so groups inside a partition share no home.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
import sim

H = cfg.DT_H
CACHE_DIR = sim.ROOT / "data" / "processed" / "cache_v2"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = sim.ROOT / "results" / "v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------------
# specs
# ---------------------------------------------------------------------------------
def spec(year="2012-13", loadvar="gc+cl", pv=1.0, tname="tou", tkw=None, cap=5.0):
    return dict(year=year, loadvar=loadvar, pv=float(pv), tname=tname, tkw=dict(tkw or {}), cap=float(cap))


def _kwstr(tkw):
    return "-".join(f"{k}{v:g}" for k, v in sorted(tkw.items())) or "base"


def spec_label(sp):
    return f"{sp['year']}_{sp['loadvar']}_pv{sp['pv']:g}_{sp['tname']}_{_kwstr(sp['tkw'])}_cap{sp['cap']:g}"


def cache_path(sp, ctrl):
    return CACHE_DIR / f"{spec_label(sp)}_{ctrl}.npz"


def get_data(sp):
    return sim.load_headline(sp["year"], include_cl=(sp["loadvar"] == "gc+cl"))


def get_tariff(sp, data):
    return sim.make_tariff(sp["tname"], data["dates"], **sp["tkw"])


# ---------------------------------------------------------------------------------
# stage 1: per-home dispatch for every home
# ---------------------------------------------------------------------------------
def stage1(sp, ctrl):
    f = cache_path(sp, ctrl)
    if f.exists():
        return str(f)
    d = get_data(sp)
    tariff = get_tariff(sp, d)
    N, T = d["load"].shape
    flow = np.zeros((N, T), dtype=np.float32)
    cyc, curt, bill = np.zeros(N), np.zeros(N), np.zeros(N)
    for i in range(N):
        fl, b, k = sim.home_flow(d["load"][i], d["pv"][i] * sp["pv"], sp["cap"], ctrl, tariff, seed=i)
        flow[i] = fl
        cyc[i] = sim.cycles_per_year(b, sp["cap"])
        curt[i] = k.sum()
        bill[i] = sim.bill(fl, tariff)
    np.savez(f, flow=flow, cyc=cyc, curt=curt, bill=bill)
    return str(f)


# ---------------------------------------------------------------------------------
# stage 2: groups
# ---------------------------------------------------------------------------------
def _charge_stats(charge_series, n, p_home_kw):
    """charge_series: (T,) kWh per slot of grid-side charging summed over batteries (simultaneous)."""
    pk = charge_series.max() / H
    return dict(chg_peak_kw=pk, chg_coincidence=pk / (n * p_home_kw), chg_energy_kwh=charge_series.sum())


def eval_groups(sp, groups, ctrls_B, ctrls_C, nopv_every=0):
    """groups: list of (part, gi, idx). Returns a DataFrame with one row per group x scenario."""
    d = get_data(sp)
    tariff = get_tariff(sp, d)
    caches = {c: np.load(cache_path(sp, c)) for c in ctrls_B}
    caches0 = None
    if nopv_every:
        sp0 = dict(sp, pv=0.0)
        caches0 = {c: np.load(cache_path(sp0, c)) for c in ctrls_B}
    cap = sp["cap"]
    p_home = cfg.BATT_C_RATE * cap
    rows = []
    for part, gi, idx in groups:
        n = len(idx)
        load = d["load"][idx]
        scale = np.full(n, sp["pv"])
        nopv = np.zeros(n, dtype=bool)
        if nopv_every:
            nopv = (np.arange(n) % nopv_every == 0)
            scale[nopv] = 0.0
        pv = d["pv"][idx] * scale[:, None]
        pv_tot, load_tot = pv.sum(), load.sum()
        net_i = load - pv
        cap_tot = cap * n
        lim = tariff.export_limit_kw_per_home

        def add(scn, flow_agg, bill_aud, cyc, curt, extra=None):
            m = sim.grid_metrics(flow_agg, pv_tot, load_tot, curt)
            imp = np.maximum(flow_agg, 0)
            m["peak_import_hour"] = (np.argmax(imp) % cfg.SLOTS) * H
            m.update(part=part, group=gi, n=n, scenario=scn, bill_aud=bill_aud, cycles=cyc)
            if extra:
                m.update(extra)
            rows.append(m)

        fa = [sim.curtail(net_i[i], lim) for i in range(n)]
        flow_A = np.sum([x[0] for x in fa], axis=0)
        add("A", flow_A, sum(sim.bill(x[0], tariff) for x in fa), 0.0, sum(x[1].sum() for x in fa))
        fAg, _, kAg = sim.group_flow(load, pv, 0.0, "rule", tariff)
        add("A_agg", fAg, sim.bill(fAg, tariff), 0.0, kAg.sum())

        for c in ctrls_B:
            Z = caches[c]
            if nopv_every:
                fl = np.array([(caches0[c] if nopv[j] else Z)["flow"][idx[j]] for j in range(n)], dtype=np.float64)
                cyc_i = np.array([(caches0[c] if nopv[j] else Z)["cyc"][idx[j]] for j in range(n)])
                curt_i = np.array([(caches0[c] if nopv[j] else Z)["curt"][idx[j]] for j in range(n)])
                bill_i = np.array([(caches0[c] if nopv[j] else Z)["bill"][idx[j]] for j in range(n)])
            else:
                fl = Z["flow"][idx].astype(np.float64)
                cyc_i, curt_i, bill_i = Z["cyc"][idx], Z["curt"][idx], Z["bill"][idx]
            flow_B = fl.sum(axis=0)
            extra = None
            if lim is None:                                       # battery series = flow - net (no curtailment)
                b_i = fl - net_i
                extra = _charge_stats(np.maximum(b_i, 0).sum(axis=0), n, p_home)
            add(f"B_{c}", flow_B, bill_i.sum(), cyc_i.mean(), curt_i.sum(), extra)
            add(f"Bagg_{c}", flow_B, sim.bill(flow_B, tariff), cyc_i.mean(), curt_i.sum(), extra)

        for c in ctrls_C:
            fC, bC, kC = sim.group_flow(load, pv, cap_tot, c, tariff, seed=gi)
            extra = _charge_stats(np.maximum(bC, 0), 1, cap_tot * cfg.BATT_C_RATE) if lim is None else None
            add(f"C_{c}", fC, sim.bill(fC, tariff), sim.cycles_per_year(bC, cap_tot), kC.sum(), extra)
    df = pd.DataFrame(rows)
    for k in ("year", "loadvar", "pv", "tname", "cap"):
        df[k if k != "tname" else "tariff"] = sp[k]
    df["tkw"] = _kwstr(sp["tkw"])
    df["nopv_every"] = nopv_every
    return df


def stage2(sp, n, partitions_seed, n_part, ctrls_B, ctrls_C, nopv_every=0, tag=""):
    """Evaluate all groups of n homes from n_part non-overlapping partitions; cached as CSV."""
    f = OUT_DIR / f"{tag}{spec_label(sp)}_n{n}_K{n_part}_{'-'.join(ctrls_B)}_{'-'.join(ctrls_C)}_mix{nopv_every}.csv"
    if f.exists():
        return str(f)
    d = get_data(sp)
    parts = sim.make_partitions(d, n, n_part, seed=partitions_seed)
    groups = [(k, gi, idx) for k, part in enumerate(parts) for gi, idx in enumerate(part)]
    df = eval_groups(sp, groups, ctrls_B, ctrls_C, nopv_every)
    df.to_csv(f, index=False)
    return str(f)
