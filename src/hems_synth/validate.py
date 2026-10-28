"""Compare the loader output with the generator truth and write ``results/validation.json`` and
``results/tariffs.json``. Every number in the README comes from these files.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import TZ
from .loader import _counter_to_cumulative, _despike, read_entity, read_esb_hdf
from .marts import duckdb_sums, load_export
from .profiles import load_profiles
from .simulate import STEP_S, read_truth
from .tariffs import PLANS, bill, brute_force_unit_eur, rate_vector

ACCEPT_FLOWS = ("import", "export", "pv", "load")
MIN_TRUTH_KWH = 2.0


def _local_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_convert(TZ).normalize().tz_localize(None)


def _dst_days(truth_hh: pd.DataFrame) -> list[str]:
    counts = truth_hh.groupby(pd.DatetimeIndex(truth_hh["ts_local"]).normalize()).size()
    return [str(d.date()) for d, c in counts.items() if c != 48]


def _daily_errors(e30: pd.DataFrame, truth_daily: pd.DataFrame) -> pd.DataFrame:
    loader_daily = e30.groupby(_local_days(e30.index)).sum(min_count=1)
    rows = []
    for day, trow in truth_daily.iterrows():
        if day not in loader_daily.index:
            continue
        lrow = loader_daily.loc[day]
        for flow in ("import", "export", "pv", "load", "batt_charge", "batt_discharge", "ev"):
            if flow not in lrow:
                continue
            t = float(trow[f"{flow}_kwh"])
            v = float(lrow[flow])
            rows.append({"date": str(day.date()), "flow": flow, "truth_kwh": t, "loader_kwh": v, "abs_kwh": abs(v - t), "pct": abs(v - t) / t * 100.0 if t >= MIN_TRUTH_KWH else np.nan, "pct_any": abs(v - t) / t * 100.0 if t > 0.0 else np.nan})
    return pd.DataFrame(rows)


def _sofar_counters(folder: Path, artefacts: dict, truth_daily: pd.DataFrame) -> dict:
    out: dict = {}
    spikes = artefacts.get("counter_spikes", [])
    removed = 0
    for entity in ("today_energy_import", "today_energy_export"):
        raw = read_entity(folder, entity)
        clean = _despike(raw)
        for sp in spikes:
            if sp["entity"].endswith(entity):
                t = pd.Timestamp(sp["time_utc"])
                if t in raw.index and t not in clean.index:
                    removed += 1
        # resets: first point of each local day whose value is below half the last value of the previous day
        local = raw.index.tz_convert(TZ)
        days = local.normalize()
        resets_total = 0
        resets_kept = 0
        prev_last = None
        prev_day = None
        for day in days.unique():
            part = raw[days == day]
            if prev_last is not None and prev_last > 0.05 and (day - prev_day) == pd.Timedelta(days=1):
                first_t = part.index[0]
                if float(part.iloc[0]) < 0.5 * prev_last:
                    resets_total += 1
                    if first_t in clean.index:
                        resets_kept += 1
            prev_last = float(part.iloc[-1])
            prev_day = day
        out[entity] = {"points": int(len(raw)), "resets": resets_total, "resets_kept": resets_kept}
    out["spikes_injected"] = len(spikes)
    out["spikes_removed"] = removed
    out["resets"] = sum(v["resets"] for v in out.values() if isinstance(v, dict))
    out["resets_kept"] = sum(v["resets_kept"] for v in out.values() if isinstance(v, dict))
    # lifetime counter recovery straight from the raw series (with injected negative glitches), per local day
    raw = read_entity(folder, "total_energy_import")
    cum = _counter_to_cumulative(raw)
    days = _local_days(cum.index)
    first_point_day = days[0]
    deltas = cum.groupby(days).last().diff()
    errs = []
    for day, delta in deltas.items():
        if day == first_point_day or day not in truth_daily.index or pd.isna(delta):
            continue
        prev_day = day - pd.Timedelta(days=1)
        if prev_day not in truth_daily.index:
            continue
        # the day's energy is the counter at the end of the day minus the counter at the end of the previous day
        errs.append({"date": str(day.date()), "abs_kwh": abs(float(delta) - float(truth_daily.loc[day, "import_kwh"]))})
    out["lifetime_glitches_injected"] = len(artefacts.get("lifetime_glitches", []))
    out["lifetime_days_checked"] = len(errs)
    out["lifetime_max_abs_error_kwh"] = round(max((e["abs_kwh"] for e in errs), default=0.0), 4)
    return out


def _gaps(site: dict, artefacts: dict, truth_days: pd.DatetimeIndex) -> dict:
    minute = site["minute"]
    alive = minute["alive"].to_numpy()
    idx = minute.index
    dead = ~alive
    starts = np.nonzero(dead & ~np.r_[False, dead[:-1]])[0]
    ends = np.nonzero(dead & ~np.r_[dead[1:], False])[0]
    runs = []
    day_set = set(truth_days.to_numpy())
    for s, e in zip(starts, ends):
        day = idx[s].normalize().tz_localize(None).to_numpy()
        if day in day_set:
            runs.append((idx[s], idx[e], int(e - s + 1)))
    injected = artefacts.get("silent_gaps", [])
    found = 0
    for g in injected:
        g0 = pd.Timestamp(g["start_utc"]).tz_convert(TZ)
        g1 = pd.Timestamp(g["end_utc"]).tz_convert(TZ)
        for r0, r1, _ in runs:
            if abs((r0 - g0).total_seconds()) <= 120 and abs((r1 - g1).total_seconds()) <= 120:
                found += 1
                break
    held_in_gap = 0
    for g in injected:
        g0 = pd.Timestamp(g["start_utc"]).tz_convert(TZ).ceil("min") + pd.Timedelta(minutes=1)
        g1 = pd.Timestamp(g["end_utc"]).tz_convert(TZ).floor("min") - pd.Timedelta(minutes=1)
        inside = minute.loc[g0:g1]
        if len(inside):
            cols = [c for c in ("grid_power", "load_power", "pv_power") if c in inside]
            held_in_gap += int(inside[cols].notna().any(axis=1).sum())
    return {"injected": len(injected), "found": found, "nan_runs_in_truth_days": len(runs), "held_minutes_inside_gaps": held_in_gap}


def _sigen_counters(site: dict, artefacts: dict, export_prefixes: dict | None) -> dict:
    raw = site["raw_counters"]["daily_pv_energy"]
    clean = site["clean_counters"]["daily_pv_energy"]
    injected = artefacts.get("sigen_pv_dips", []) + artefacts.get("sigen_pv_spikes", [])
    removed = 0
    for a in injected:
        t = pd.Timestamp(a["time_utc"]).tz_convert(TZ)
        if t in raw.index:
            c, r, tv = float(clean[t]), float(raw[t]), float(a["true_value"])
            if abs(c - r) > 0.1 and abs(c - tv) < abs(r - tv):
                removed += 1
    battery_unchanged = all(site["raw_counters"][k].equals(site["clean_counters"][k]) for k in ("daily_charge_energy", "daily_discharge_energy") if k in site["raw_counters"])
    prefixes_ok = export_prefixes is None or (site["prefixes"]["station"] == export_prefixes.get("station") and site["prefixes"]["inverter"] == export_prefixes.get("inverter"))
    return {
        "dips_injected": len(artefacts.get("sigen_pv_dips", [])),
        "spikes_injected": len(artefacts.get("sigen_pv_spikes", [])),
        "artefacts_removed": removed,
        "battery_counters_unchanged": "yes" if battery_unchanged else "no",
        "prefixes_from_files": "yes" if prefixes_ok else "no",
        "prefixes": site["prefixes"],
    }


def _esb_checks(folder: Path, mprn: str, truth_hh: pd.DataFrame) -> dict:
    path = folder / f"HDF_calckWh_{mprn}.csv"
    raw = pd.read_csv(path, dtype=str)
    ends = raw["Read Date and End Time"]
    dst_end_rows = int(((ends.str.startswith("25-10-2026")) & (ends.str.endswith(" 01:00") | ends.str.endswith(" 01:30"))).sum())
    local = read_esb_hdf(path, clock="local")
    th = truth_hh.copy()
    th.index = th.index.tz_convert(TZ)
    j = local.join(th[["import_kwh", "export_kwh"]], how="inner")
    max_diff = float(max((j["import"] - j["import_kwh"]).abs().max(), (j["export"] - j["export_kwh"]).abs().max()))
    day_counts = local.groupby(local.index.normalize()).size()
    rows_25_oct = int(day_counts.get(pd.Timestamp("2026-10-25", tz=TZ), 0))
    no_duplicate_labels = bool(local.index.is_unique)
    kw_path = folder / f"HDF_kW_{mprn}.csv"
    kw = read_esb_hdf(kw_path, clock="local")
    jk = kw.join(th[["import_kwh"]], how="inner")
    kw_max_diff = float((jk["import"] / 2.0 - jk["import_kwh"]).abs().max())
    return {
        "kwh_rows_joined": int(len(j)),
        "kwh_max_abs_diff_per_interval": round(max_diff, 5),
        "kw_file_max_abs_diff_kwh": round(kw_max_diff, 5),
        "dst_end_rows_with_end_time_in_01_hour_both_registers": dst_end_rows,
        "dst_end_rows_per_register": dst_end_rows // 2,
        "rows_25_oct_after_local_read": rows_25_oct,
        "ambiguous_hour_dropped": "yes" if (dst_end_rows == 8 and rows_25_oct == 46 and no_duplicate_labels) else ("no" if dst_end_rows > 0 else "not in run"),
    }


def _tariff_checks(truth: pd.DataFrame, truth_hh: pd.DataFrame, tariff_cost: pd.DataFrame, site_id: str) -> dict:
    imp_w = truth["import_w"]
    hh = truth_hh.copy()
    hh.index = hh.index.tz_convert(TZ)
    months = hh.index.strftime("%Y-%m")
    months_5s = imp_w.index.tz_convert(TZ).strftime("%Y-%m")
    max_diff = 0.0
    checks = 0
    for month in sorted(set(months)):
        m = months == month
        m5 = months_5s == month
        days = len(set(hh.index[m].normalize()))
        for key, plan in PLANS.items():
            vec = bill(hh["import_kwh"][m], hh["export_kwh"][m], plan, days)["unit_eur"]
            brute = brute_force_unit_eur(imp_w[m5], plan, STEP_S)
            max_diff = max(max_diff, abs(vec - brute))
            checks += 1
    rows_per_month = tariff_cost[tariff_cost["site_id"] == site_id].groupby("month").size()
    return {
        "plans": len(PLANS),
        "rate_vector_lengths_all_48": "yes" if all(len(rate_vector(p)) == 48 for p in PLANS.values()) else "no",
        "brute_force_checks": checks,
        "brute_force_max_abs_diff_eur": round(max_diff, 6),
        "tariff_cost_rows_per_month": {str(k): int(v) for k, v in rows_per_month.items()},
        "eleven_rows_every_month": "yes" if len(rows_per_month) and (rows_per_month == len(PLANS)).all() else "no",
    }


def validate(data_dir: Path, marts_dir: Path, loaded: dict | None = None, config: Path | None = None) -> dict:
    """Validate every site with truth under ``data_dir`` and marts under ``marts_dir``."""
    data_dir, marts_dir = Path(data_dir), Path(marts_dir)
    profiles = load_profiles(config) if config else load_profiles()
    tariff_cost = pd.read_parquet(marts_dir / "tariff_cost.parquet")
    energy_30 = pd.read_parquet(marts_dir / "energy_30min.parquet")
    result: dict = {"sites": {}, "marts": {}, "tariffs": {}}
    for truth_folder in sorted((data_dir / "truth").iterdir()):
        sid = truth_folder.name
        if sid not in profiles or not (truth_folder / "truth_30min.parquet").exists():
            continue
        prof = profiles[sid]
        tr = read_truth(truth_folder)
        export_folder = data_dir / "exports" / sid
        site = (loaded or {}).get(sid) or load_export(export_folder, prof.inverter_family)
        artefacts = tr["artefacts"]
        truth_hh = tr["half_hour"]
        truth_daily = tr["daily"]
        truth_days = pd.DatetimeIndex(truth_hh["ts_local"]).normalize().unique()
        e30 = site["energy30"]
        e30 = e30[np.isin(_local_days(e30.index).to_numpy(), truth_days.to_numpy())]
        dst_days = _dst_days(truth_hh)
        gap_days = sorted({str(pd.Timestamp(g["start_utc"]).tz_convert(TZ).date()) for g in artefacts.get("silent_gaps", [])})
        errors = _daily_errors(e30, truth_daily)
        acc = errors[errors["flow"].isin(ACCEPT_FLOWS)]
        ordinary = acc[~acc["date"].isin(dst_days)]
        no_gap = ordinary[~ordinary["date"].isin(gap_days)]
        hh_mae = {}
        th = truth_hh.copy()
        th.index = th.index.tz_convert(TZ)
        for flow in ("import", "export", "pv", "load"):
            j = e30[[flow]].join(th[[f"{flow}_kwh"]], how="inner")
            hh_mae[flow] = round(float((j[flow] - j[f"{flow}_kwh"]).abs().mean()), 5)
        counts = e30.groupby(_local_days(e30.index)).size()
        mart_rows = energy_30[(energy_30["site_id"] == sid)]
        mart_counts = {src: int(n) for src, n in mart_rows.groupby("source").size().items()}
        expected_rows = int(truth_hh.groupby(pd.DatetimeIndex(truth_hh["ts_local"]).normalize()).size().sum())
        site_result = {
            "family": prof.inverter_family,
            "days": int(len(truth_days)),
            "dst_days": dst_days,
            "gap_days": gap_days,
            "daily_pct_floor_kwh": MIN_TRUTH_KWH,
            "daily_max_pct_ordinary_days": round(float(ordinary["pct"].max()), 4) if ordinary["pct"].notna().any() else None,
            "daily_max_pct_ordinary_days_without_gaps": round(float(no_gap["pct"].max()), 4) if no_gap["pct"].notna().any() else None,
            "daily_max_pct_ordinary_days_any_size": round(float(ordinary["pct_any"].max()), 4) if ordinary["pct_any"].notna().any() else None,
            "daily_max_abs_kwh_ordinary_days": round(float(ordinary["abs_kwh"].max()), 4),
            "daily_max_pct_by_flow": {f: round(float(ordinary[ordinary["flow"] == f]["pct"].max()), 4) if ordinary[ordinary["flow"] == f]["pct"].notna().any() else None for f in ACCEPT_FLOWS},
            "daily_max_abs_kwh_by_flow": {f: round(float(ordinary[ordinary["flow"] == f]["abs_kwh"].max()), 4) for f in ACCEPT_FLOWS},
            "daily_errors_within_0_5_pct": "yes" if ordinary["pct"].dropna().le(0.5).all() else "no",
            "half_hour_mae_kwh": hh_mae,
            "half_hours_2026_03_29": int(counts.get(pd.Timestamp("2026-03-29"), 0)),
            "half_hours_2026_10_25": int(counts.get(pd.Timestamp("2026-10-25"), 0)),
            "energy_30min_rows": mart_counts,
            "energy_30min_expected_rows_per_source": expected_rows,
            "energy_30min_row_count_ok": "yes" if all(v == expected_rows for v in mart_counts.values()) and len(mart_counts) == 2 else "no",
            "read_entity_utc_unique": None,
        }
        if prof.inverter_family == "sofar":
            ok = True
            for name in ("grid_power", "load_power", "pv_power", "power"):
                s = read_entity(export_folder, name)
                ok &= str(s.index.tz) == "UTC" and s.index.is_unique and s.dtype.kind == "f"
            site_result["read_entity_utc_unique"] = "yes" if ok else "no"
            site_result["counters"] = _sofar_counters(export_folder, artefacts, truth_daily)
            site_result["gaps"] = _gaps(site, artefacts, truth_days)
        else:
            meta_prefixes = tr["meta"].get("export_prefixes")
            site_result["sigen"] = _sigen_counters(site, artefacts, meta_prefixes)
            site_result["gaps"] = {"injected": 0, "found": 0, "nan_runs_in_truth_days": 0, "held_minutes_inside_gaps": 0}
        site_result["esb"] = _esb_checks(export_folder, prof.mprn, truth_hh)
        site_result["tariffs"] = _tariff_checks(tr["truth"], truth_hh, tariff_cost, sid)
        result["sites"][sid] = site_result

    # marts: parquet and duckdb both present, duckdb sums equal parquet sums
    db = marts_dir / "hems.duckdb"
    cols = ["import_kwh", "export_kwh", "pv_kwh", "load_kwh"]
    parquet_sums = {c: float(energy_30[c].sum()) for c in cols}
    duck = duckdb_sums(db, "energy_30min", cols) if db.exists() else {}
    sums_equal = bool(duck) and all(abs(parquet_sums[c] - duck[c]) < 1e-6 for c in cols)
    result["marts"] = {
        "parquet_present": "yes" if all((marts_dir / f"{t}.parquet").exists() for t in ("sites", "energy_30min", "daily", "tariff_cost", "tariffs", "esb_30min", "truth_vs_export")) else "no",
        "duckdb_present": "yes" if db.exists() else "no",
        "duckdb_sums_equal_parquet": "yes" if sums_equal else "no",
        "parquet_sums": {k: round(v, 4) for k, v in parquet_sums.items()},
        "duckdb_sums": {k: round(v, 4) for k, v in duck.items()},
    }
    # tariffs summary: cheapest plan per site and month from the mart
    cheapest = {}
    for (sid, month), part in tariff_cost.groupby(["site_id", "month"]):
        best = part.sort_values("net_eur").iloc[0]
        cheapest[f"{sid}|{month}"] = {"plan_key": best["plan_key"], "net_eur": round(float(best["net_eur"]), 2), "days": int(best["days"])}
    per_site = {}
    for sid, part in tariff_cost.groupby("site_id"):
        tot = part.groupby("plan_key")[["net_eur", "days"]].sum().sort_values("net_eur")
        best_key = tot.index[0]
        per_30 = float(tot.loc[best_key, "net_eur"]) / float(tot.loc[best_key, "days"]) * 30.0
        per_site[sid] = {"plan_key": best_key, "net_eur_per_30_days": round(per_30, 2), "days": int(tot.loc[best_key, "days"])}
    result["tariffs"] = {"plans_costed": len(PLANS), "cheapest_per_site_month": cheapest, "cheapest_per_site": per_site}
    return result


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
