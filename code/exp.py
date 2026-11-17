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

import hashlib
import json
import time
from pathlib import Path

import config as cfg
import sim

H = cfg.DT_H


def code_sha():
    """Hash of the simulation code (sim.py + exp.py); stored next to every stage-2 result so that results
    produced by an older version of the code can be detected (tools/check_provenance.py)."""
    h = hashlib.sha256()
    for name in ("sim.py", "exp.py"):
        h.update((Path(__file__).resolve().parent / name).read_bytes())
    return h.hexdigest()[:12]
CACHE_DIR = sim.ROOT / "data" / "processed" / "cache_v2"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = sim.ROOT / "results" / "v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------------
# specs
# ---------------------------------------------------------------------------------
def spec(year="2012-13", loadvar="gc+cl", pv=1.0, tname="tou", tkw=None, cap=5.0, batt=None):
    """batt: optional battery overrides {'c_rate': .., 'eta': ..} (default: values in config.py)."""
    return dict(year=year, loadvar=loadvar, pv=float(pv), tname=tname, tkw=dict(tkw or {}), cap=float(cap),
                batt=dict(batt or {}))


def use_battery(sp):
    sim.set_battery(**sp.get("batt", {}))


def _kwstr(tkw):
    return "-".join(f"{k}{v:g}" for k, v in sorted(tkw.items())) or "base"


def spec_label(sp):
    lab = f"{sp['year']}_{sp['loadvar']}_pv{sp['pv']:g}_{sp['tname']}_{_kwstr(sp['tkw'])}_cap{sp['cap']:g}"
    if sp.get("batt"):
        lab += "_batt-" + _kwstr(sp["batt"])
    return lab


def cache_path(sp, ctrl):
    return CACHE_DIR / f"{spec_label(sp)}_{ctrl}.npz"


def get_data(sp):
    return sim.load_headline(sp["year"], include_cl=(sp["loadvar"] == "gc+cl"))


PREV_YEAR = {"2012-13": "2011-12", "2011-12": "2010-11"}


def prev_year_data(sp):
    return sim.load_headline(PREV_YEAR[sp["year"]], include_cl=(sp["loadvar"] == "gc+cl"))


def coordinated_individual(sp, groups):
    """Scenario B_coord: individual batteries, individual bills, coordinated charging. Starting from each home's own
    cost-minimising schedule (stage-1 cache, controller 'lp'), homes are visited once in order; each re-chooses,
    among the schedules that cost it no more than its own optimum, the one that minimises the group's maximum
    import given the other homes' current schedules (sim.lp_coord_home). One pass, so a heuristic and not the
    joint optimum. Returns rows like eval_groups for scenarios A, B_lp and Bcoord_lp."""
    use_battery(sp)
    d = get_data(sp)
    tariff = get_tariff(sp, d)
    Z = np.load(cache_path(sp, "lp"))
    rows = []
    for part, gi, idx in groups:
        load, pv = d["load"][idx], d["pv"][idx] * sp["pv"]
        net_i = load - pv
        fl = Z["flow"][idx].astype(np.float64)
        J = Z["bill"][idx] * 100.0
        fA = net_i.sum(axis=0)
        pkA = np.maximum(fA, 0).max()
        new = fl.copy()
        for j in range(len(idx)):
            others = new.sum(axis=0) - new[j]
            new[j] = sim.lp_coord_home(net_i[j], pv[j], tariff, sp["cap"], others, J[j])
        for scn, F, bills in (("A", fA, sum(sim.bill(x, tariff) for x in net_i)), ("B_lp", fl.sum(axis=0), Z["bill"][idx].sum()),
                              ("Bcoord_lp", new.sum(axis=0), sum(sim.bill(x, tariff) for x in new))):
            m = sim.grid_metrics(F, pv.sum(), load.sum())
            imp = np.maximum(F, 0)
            m.update(part=part, group=gi, n=len(idx), scenario=scn, bill_aud=bills,
                     h_ge90=float((imp >= 0.9 * pkA).sum() * H), peak_import_hour=(np.argmax(imp) % cfg.SLOTS) * H)
            rows.append(m)
    df = pd.DataFrame(rows)
    for k in ("year", "loadvar", "pv", "tname", "cap"):
        df[k if k != "tname" else "tariff"] = sp[k]
    df["tkw"], df["batt"], df["nopv_every"] = _kwstr(sp["tkw"]), _kwstr(sp.get("batt", {})), 0
    return df


def get_tariff(sp, data):
    """Tariff of a spec. Names ending in '_rn' are revenue-neutral demand-charge variants: all energy prices are
    scaled by one factor so that the homes of the dataset, billed individually and without batteries (scenario A),
    pay in total exactly what they pay under the plain ToU tariff."""
    t = sim.make_tariff(sp["tname"], data["dates"], **sp["tkw"])
    if not sp["tname"].endswith("_rn"):
        return t
    net = data["load"] - data["pv"] * sp["pv"]
    imp = np.maximum(net, 0.0)
    energy = float((imp * t.price).sum())                                     # cents
    zero = sim.Tariff(t.name, np.zeros_like(t.price), 0.0, t.peak_mask, t.month, t.days_in_month,
                      t.demand_c_per_kw_day, t.export_limit_kw_per_home)
    demand = 100.0 * sum(sim.bill(x, zero) for x in net)                      # cents
    return sim.make_tariff(sp["tname"], data["dates"], scale=1.0 - demand / energy, **sp["tkw"])


# ---------------------------------------------------------------------------------
# stage 1: per-home dispatch for every home
# ---------------------------------------------------------------------------------
def stage1(sp, ctrl):
    f = cache_path(sp, ctrl)
    if f.exists():
        return str(f)
    use_battery(sp)
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
    use_battery(sp)
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

        ref = {}

        def add(scn, flow_agg, bill_aud, cyc, curt, extra=None):
            m = sim.grid_metrics(flow_agg, pv_tot, load_tot, curt)
            imp = np.maximum(flow_agg, 0)
            m["peak_import_hour"] = (np.argmax(imp) % cfg.SLOTS) * H
            m["peak_import_day"] = int(np.argmax(imp) // cfg.SLOTS)
            if scn == "A":
                ref["pk"] = imp.max()
            m["h_ge90"] = float((imp >= 0.9 * ref["pk"]).sum() * H)       # hours at or above 90 % of the no-battery peak
            m["h_ge999"] = float((imp >= 0.999 * ref["pk"]).sum() * H)    # hours at (or above) the no-battery peak
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

        ext_cap = None
        if any("capP" in c for c in ctrls_C):       # previous year's no-battery peak of the same homes (kWh per half-hour)
            dp = prev_year_data(sp)
            pos = {h: k for k, h in enumerate(dp["ids"])}
            jj = [pos[h] for h in d["ids"][idx]]
            ext_cap = float(np.maximum((dp["load"][jj] - dp["pv"][jj] * sp["pv"]).sum(axis=0), 0).max())
        for c in ctrls_C:
            fC, bC, kC = sim.group_flow(load, pv, cap_tot, c, tariff, seed=gi, ext_cap=ext_cap)
            extra = _charge_stats(np.maximum(bC, 0), 1, cap_tot * cfg.BATT_C_RATE) if lim is None else None
            add(f"C_{c}", fC, sim.bill(fC, tariff), sim.cycles_per_year(bC, cap_tot), kC.sum(), extra)
    df = pd.DataFrame(rows)
    for k in ("year", "loadvar", "pv", "tname", "cap"):
        df[k if k != "tname" else "tariff"] = sp[k]
    df["tkw"] = _kwstr(sp["tkw"])
    df["batt"] = _kwstr(sp.get("batt", {}))
    df["nopv_every"] = nopv_every
    return df


def stage2(sp, n, partitions_seed, n_part, ctrls_B, ctrls_C, nopv_every=0, tag=""):
    """Evaluate all groups of n homes from n_part non-overlapping partitions; cached as CSV."""
    cpart = f"{'-'.join(ctrls_B)}_{'-'.join(ctrls_C)}"
    if len(cpart) > 120:     # file-name limit: long controller lists are replaced by a hash (the lists are in .meta.json)
        cpart = "h" + hashlib.md5(cpart.encode()).hexdigest()[:10]
    f = OUT_DIR / f"{tag}{spec_label(sp)}_n{n}_K{n_part}_{cpart}_mix{nopv_every}.csv"
    if f.exists():
        return str(f)
    d = get_data(sp)
    parts = sim.make_partitions(d, n, n_part, seed=partitions_seed)
    groups = [(k, gi, idx) for k, part in enumerate(parts) for gi, idx in enumerate(part)]
    t0 = time.time()
    df = eval_groups(sp, groups, ctrls_B, ctrls_C, nopv_every)
    df.to_csv(f, index=False)
    f.with_suffix(".meta.json").write_text(json.dumps(dict(
        spec=sp, n=n, partitions_seed=partitions_seed, n_partitions=n_part, controllers_B=ctrls_B, controllers_C=ctrls_C,
        code_sha=code_sha(), config_sha=cfg.CONFIG_SHA, duration_s=round(time.time() - t0, 1),
        written=time.strftime("%Y-%m-%dT%H:%M:%S")), default=str))
    return str(f)
