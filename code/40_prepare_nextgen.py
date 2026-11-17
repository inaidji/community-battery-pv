"""Preprocess the second dataset: NextGen energy storage trial, ACT, Australia (Sturmberg & Shaw, Zenodo
10.5281/zenodo.14885589, CC BY 4.0). 100 households, calendar year 2018, 5-minute load and solar power.

Raw files (data/raw_nextgen/*.csv) are read only; checksums are verified against tools/nextgen_md5.txt.
Output: data/processed/nextgen_2018.npz with the same keys as sim.load_headline (kWh per half-hour, local
clock time Australia/Sydney, which is also the ACT's time zone).

Steps
  1. index (Unix seconds, UTC) -> local wall-clock date and half-hour slot.
  2. half-hour energy = mean power in the slot x 0.5 h (mean, not sum, so missing 5-minute samples do not bias it).
  3. Daylight saving: the hour that does not exist on the spring-forward day is filled by linear interpolation
     between its neighbours; the hour that occurs twice on the fall-back day is averaged.
  4. solar power is stored negative in the raw files; PV output = -solar, clipped at 0.
  5. the homes' own batteries are ignored: 'load power' is the household consumption and 'solar power' the PV
     output, both measured separately from the battery, so the net load without a battery is load - PV.
  6. Missing days. 99 of the 100 homes have 1-10 whole days without measurements (median 6 days, 1.6 % of
     the year; the days differ between homes). A missing day is replaced by the same home's nearest
     measured day of the same weekday (7 days earlier, else 7 later, else 14 earlier, else 14 later), for load
     and PV together. The number of imputed days per home is stored ('imputed_days') and logged.
  7. homes still incomplete after step 6, or with a non-positive annual load or PV, are dropped (logged).

Two outputs are written:
  nextgen_2018_strict.npz  MAIN version, NO imputation of days (steps 6 is skipped). Days on which at least
                           OUTAGE_SHARE of the homes have no data are dropped for all homes; homes with more than
                           MAX_MISSING_DAYS further whole days missing are excluded; the calendar is then restricted to
                           the days on which every remaining home has complete measurements. All quantities are
                           therefore measured; totals refer to the retained days, not to a full year.
  nextgen_2018.npz         sensitivity version with step 6 (same-weekday imputation), all homes, full year.
The dataset has no postcode. Groups are therefore blocks of a fixed random order of the homes (seed below),
stored in the 'postcode' field so that sim.make_partitions works unchanged.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_nextgen"
OUT = ROOT / "data" / "processed" / "nextgen_2018.npz"
ORDER_SEED = 2018
MAX_MISSING_DAYS = 4      # strict version: a home may lack at most this many days outside common outages
OUTAGE_SHARE = 0.10       # a day without data for at least this share of homes is a common outage (dropped for all)


def main():
    sums = dict(l.split()[::-1] for l in (ROOT / "tools" / "nextgen_md5.txt").read_text().splitlines() if l.strip())
    days = pd.date_range("2018-01-01", periods=365, freq="D")
    ids, load, pv, kwp, battcap, dropped, imputed = [], [], [], [], [], [], []
    raw_day, all_ids, all_kwp, all_batt = [], [], [], []
    for name in sorted(sums, key=lambda s: int(s.split("-")[0])):
        f = RAW / name
        if hashlib.md5(f.read_bytes()).hexdigest() != sums[name]:
            sys.exit(f"checksum mismatch: {name}")
        d = pd.read_csv(f)
        d.columns = ["ts", "load", "solar", "batt", "soc", "solcap", "battcap", "battpk"]
        t = pd.to_datetime(d["ts"], unit="s", utc=True).dt.tz_convert("Australia/Sydney")
        key = [t.dt.normalize().dt.tz_localize(None), t.dt.hour * 2 + t.dt.minute // 30]
        g = d.assign(pvp=(-d["solar"]).clip(lower=0)).groupby(key)[["load", "pvp"]].mean()
        full = pd.MultiIndex.from_product([days, range(48)])
        g = g.reindex(full)
        arr = g.to_numpy().reshape(365 * 48, 2)
        arr = pd.DataFrame(arr).interpolate(limit=2, limit_area="inside").to_numpy()   # only the DST gap (2 slots)
        day = arr.reshape(365, 48, 2)
        raw_day.append(day.copy())
        miss = np.flatnonzero(np.isnan(day).any(axis=(1, 2)))
        okday = ~np.isnan(day).any(axis=(1, 2))
        n_imp = 0
        for dd in miss:
            for off in (-7, 7, -14, 14):
                if 0 <= dd + off < 365 and okday[dd + off]:
                    day[dd] = day[dd + off]
                    n_imp += 1
                    break
        arr = day.reshape(365 * 48, 2)
        hid = name.split("id_")[1].split(".")[0]
        all_ids.append(int(hid)); all_kwp.append(float(d["solcap"].iloc[0])); all_batt.append(float(d["battcap"].iloc[0]))
        if np.isnan(arr).any() or arr[:, 0].sum() <= 0 or arr[:, 1].sum() <= 0:
            dropped.append((hid, int(np.isnan(arr).sum())))
            continue
        ids.append(int(hid)); load.append(arr[:, 0] * 0.5); pv.append(arr[:, 1] * 0.5)
        kwp.append(float(d["solcap"].iloc[0])); battcap.append(float(d["battcap"].iloc[0])); imputed.append(n_imp)
    ids = np.array(ids)
    order = np.random.default_rng(ORDER_SEED).permutation(len(ids))
    pseudo = np.empty(len(ids), dtype=int)
    pseudo[order] = np.arange(len(ids))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, ids=ids, postcode=pseudo, kwp=np.array(kwp), battcap=np.array(battcap), imputed_days=np.array(imputed),
                        load=np.array(load), pv=np.array(pv), dates=days.to_numpy())
    L, P = np.array(load), np.array(pv)
    print(f"kept {len(ids)} homes, dropped {len(dropped)}: {dropped}")
    print(f"mean annual load {L.sum(1).mean():.0f} kWh, PV {P.sum(1).mean():.0f} kWh, PV size {np.mean(kwp):.2f} kWp, "
          f"own battery {np.mean(battcap):.1f} kWh; imputed days per home: median {np.median(imputed):.0f}, "
          f"max {np.max(imputed)}, total {np.sum(imputed)} of {365 * len(ids)} home-days")

    # ---- strict version: no imputed day ----
    R = np.array(raw_day)                                   # (homes, 365, 48, 2), NaN where not measured
    missing = np.isnan(R).any(axis=(2, 3))                  # (homes, 365)
    outage = missing.mean(0) >= OUTAGE_SHARE                # days lost for many homes at once (2 days in 2018)
    keep_h = missing[:, ~outage].sum(1) <= MAX_MISSING_DAYS
    keep_h &= (np.nansum(R[..., 0], axis=(1, 2)) > 0) & (np.nansum(R[..., 1], axis=(1, 2)) > 0)
    keep_d = ~missing[keep_h].any(0)
    S = R[keep_h][:, keep_d]                                # (homes, days, 48, 2)
    assert not np.isnan(S).any()
    sid = np.array(all_ids)[keep_h]
    order = np.random.default_rng(ORDER_SEED).permutation(len(sid))
    pseudo = np.empty(len(sid), dtype=int)
    pseudo[order] = np.arange(len(sid))
    Ls, Ps = S[..., 0].reshape(len(sid), -1) * 0.5, S[..., 1].reshape(len(sid), -1) * 0.5
    np.savez_compressed(OUT.with_name("nextgen_2018_strict.npz"), ids=sid, postcode=pseudo, kwp=np.array(all_kwp)[keep_h],
                        battcap=np.array(all_batt)[keep_h], imputed_days=np.zeros(len(sid), dtype=int),
                        load=Ls, pv=Ps, dates=days[keep_d].to_numpy())
    bym = pd.Series(keep_d, index=days).groupby(days.month).sum().tolist()
    f = 365.0 / keep_d.sum()
    print(f"STRICT: kept {len(sid)} homes (<= {MAX_MISSING_DAYS} missing days outside {int(outage.sum())} common outage days) and {keep_d.sum()} days complete for all of them; "
          f"days per month {bym}; load {Ls.sum(1).mean():.0f} kWh, PV {Ps.sum(1).mean():.0f} kWh over the retained days "
          f"(x{f:.3f} = {Ls.sum(1).mean()*f:.0f} / {Ps.sum(1).mean()*f:.0f} kWh per year); PV size {np.array(all_kwp)[keep_h].mean():.2f} kWp")


if __name__ == "__main__":
    main()
