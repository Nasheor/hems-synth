"""Clean marts in Parquet and DuckDB.

Tables under ``data/marts/`` (one Parquet file each) and in ``data/marts/hems.duckdb``:

* ``sites``: profile columns, no names.
* ``energy_30min``: ``site_id, ts_utc, ts_local, source`` (``truth`` or ``export``) and kWh per flow
  plus ``soc_pct_end``. ``ts_local`` is naive Irish wall-clock time, ``ts_utc`` is UTC.
* ``daily``: the same flows per local day plus ``self_consumption_pct`` and ``ev_sessions``.
* ``tariff_cost``: per site, month and plan, computed from the truth rows.
* ``tariffs``: the plans table.
* ``esb_30min``: the HDF-style file read with ``clock="local"``.
* ``truth_vs_export``: per site and day, kWh per flow from truth and from the loader, difference and percentage.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from . import TZ
from .loader import load_site, load_site_sigen, read_esb_hdf
from .profiles import SiteProfile, load_profiles
from .simulate import FLOWS, read_truth
from .tariffs import PLANS, bill, plans_table

TABLES = ("sites", "energy_30min", "daily", "tariff_cost", "tariffs", "esb_30min", "truth_vs_export")
FLOW_COLS = [f"{f}_kwh" for f in FLOWS]


def load_export(site_folder: Path, family: str) -> dict:
    return load_site_sigen(site_folder) if family == "sigenergy" else load_site(site_folder)


def _export_30min(site: dict, site_id: str, truth_days: pd.DatetimeIndex) -> pd.DataFrame:
    e = site["energy30"]
    local_day = e.index.normalize().tz_localize(None)
    e = e[np.isin(local_day.to_numpy(), truth_days.to_numpy())]
    out = pd.DataFrame(
        {
            "site_id": site_id,
            "ts_utc": e.index.tz_convert("UTC"),
            "ts_local": e.index.tz_localize(None),
            "source": "export",
            "import_kwh": e["import"].to_numpy(),
            "export_kwh": e["export"].to_numpy(),
            "pv_kwh": e["pv"].to_numpy(),
            "load_kwh": e["load"].to_numpy(),
            "batt_charge_kwh": e["batt_charge"].to_numpy(),
            "batt_discharge_kwh": e["batt_discharge"].to_numpy(),
            "ev_kwh": e["ev"].to_numpy(),
            "soc_pct_end": e["soc_pct_end"].to_numpy(),
        }
    )
    return out.reset_index(drop=True)


def _truth_30min(hh: pd.DataFrame, site_id: str) -> pd.DataFrame:
    out = pd.DataFrame({"site_id": site_id, "ts_utc": hh.index.tz_convert("UTC"), "ts_local": hh["ts_local"].to_numpy(), "source": "truth"})
    for c in FLOW_COLS + ["soc_pct_end"]:
        out[c] = hh[c].to_numpy()
    return out.reset_index(drop=True)


def _daily(e30: pd.DataFrame, ev_sessions_by_day: dict[str, int] | None) -> pd.DataFrame:
    local_day = pd.DatetimeIndex(e30["ts_local"]).normalize()
    g = e30.groupby([e30["site_id"], local_day.rename("date_local"), e30["source"]])[FLOW_COLS].sum(min_count=1).reset_index()
    with np.errstate(divide="ignore", invalid="ignore"):
        g["self_consumption_pct"] = np.where(g["pv_kwh"] > 0, (g["pv_kwh"] - g["export_kwh"]) / g["pv_kwh"] * 100.0, np.nan)
    if ev_sessions_by_day:
        g["ev_sessions"] = [ev_sessions_by_day.get(f"{s}|{d.date()}|{src}", 0) for s, d, src in zip(g["site_id"], g["date_local"], g["source"])]
    else:
        g["ev_sessions"] = 0
    return g


def _tariff_cost(truth_30: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for site_id, part in truth_30.groupby("site_id"):
        idx = pd.DatetimeIndex(part["ts_utc"]).tz_convert(TZ)
        imp = pd.Series(part["import_kwh"].to_numpy(), index=idx)
        exp = pd.Series(part["export_kwh"].to_numpy(), index=idx)
        months = idx.strftime("%Y-%m")
        for month in sorted(set(months)):
            m = months == month
            days = len(set(idx[m].normalize()))
            for key, plan in PLANS.items():
                b = bill(imp[m], exp[m], plan, days)
                rows.append({"site_id": site_id, "month": month, "plan_key": key, "days": days, **{k: v for k, v in b.items() if k != "days"}})
    cols = ["site_id", "month", "plan_key", "days", "kwh_import", "kwh_export", "kwh_day", "kwh_night", "kwh_peak", "kwh_ev", "unit_eur", "standing_eur", "pso_eur", "export_credit_eur", "net_eur"]
    return pd.DataFrame(rows, columns=cols)


def _truth_vs_export(daily: pd.DataFrame) -> pd.DataFrame:
    t = daily[daily["source"] == "truth"].set_index(["site_id", "date_local"])
    e = daily[daily["source"] == "export"].set_index(["site_id", "date_local"])
    rows = []
    for key in t.index:
        if key not in e.index:
            continue
        for flow in FLOWS:
            tv = float(t.loc[key, f"{flow}_kwh"])
            ev = float(e.loc[key, f"{flow}_kwh"])
            diff = ev - tv
            rows.append({"site_id": key[0], "date_local": key[1], "flow": flow, "truth_kwh": round(tv, 4), "export_kwh": round(ev, 4), "diff_kwh": round(diff, 4), "diff_pct": round(diff / tv * 100.0, 4) if tv > 0 else np.nan})
    return pd.DataFrame(rows, columns=["site_id", "date_local", "flow", "truth_kwh", "export_kwh", "diff_kwh", "diff_pct"])


def _esb(site_folder: Path, profile: SiteProfile) -> pd.DataFrame:
    path = site_folder / f"HDF_calckWh_{profile.mprn}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["site_id", "ts_local", "ts_utc", "import_kwh", "export_kwh"])
    df = read_esb_hdf(path, clock="local")
    return pd.DataFrame(
        {
            "site_id": profile.site_id,
            "ts_local": df.index.tz_localize(None),
            "ts_utc": df.index.tz_convert("UTC"),
            "import_kwh": df["import"].to_numpy(),
            "export_kwh": df["export"].to_numpy(),
        }
    ).reset_index(drop=True)


def build_marts(data_dir: Path, out_dir: Path, site_ids: list[str] | None = None, config: Path | None = None) -> dict:
    """Read truth and exports under ``data_dir`` and write every mart to ``out_dir``. Returns row counts."""
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profiles = load_profiles(config) if config else load_profiles()
    truth_root = data_dir / "truth"
    found = sorted(p.name for p in truth_root.iterdir() if (p / "truth_30min.parquet").exists()) if truth_root.exists() else []
    sites = [s for s in (site_ids or found) if s in found]
    frames_30: list[pd.DataFrame] = []
    esb_frames: list[pd.DataFrame] = []
    site_rows = []
    ev_by_day: dict[str, int] = {}
    loaded: dict[str, dict] = {}
    for sid in sites:
        prof = profiles[sid]
        tr = read_truth(truth_root / sid)
        hh = tr["half_hour"]
        frames_30.append(_truth_30min(hh, sid))
        truth_days = pd.DatetimeIndex(hh["ts_local"]).normalize().unique()
        for s in tr["ev_sessions"]:
            key = f"{sid}|{s['start_local_date']}|truth"
            ev_by_day[key] = ev_by_day.get(key, 0) + 1
        export_folder = data_dir / "exports" / sid
        if export_folder.exists():
            site = load_export(export_folder, prof.inverter_family)
            loaded[sid] = site
            frames_30.append(_export_30min(site, sid, truth_days))
            sessions = site.get("ev_sessions")
            if sessions is not None and len(sessions):
                for _, r in sessions.iterrows():
                    key = f"{sid}|{pd.Timestamp(r['start']).date()}|export"
                    ev_by_day[key] = ev_by_day.get(key, 0) + 1
            esb_frames.append(_esb(export_folder, prof))
        row = prof.as_dict()
        site_rows.append(row)

    energy_30 = pd.concat(frames_30, ignore_index=True)
    energy_30["ts_utc"] = pd.DatetimeIndex(energy_30["ts_utc"]).tz_convert("UTC")
    energy_30 = energy_30.sort_values(["site_id", "source", "ts_utc"]).reset_index(drop=True)
    daily = _daily(energy_30, ev_by_day).sort_values(["site_id", "source", "date_local"]).reset_index(drop=True)
    tariff_cost = _tariff_cost(energy_30[energy_30["source"] == "truth"])
    tariffs = plans_table()
    esb = pd.concat(esb_frames, ignore_index=True) if esb_frames else pd.DataFrame(columns=["site_id", "ts_local", "ts_utc", "import_kwh", "export_kwh"])
    tve = _truth_vs_export(daily)
    sites_df = pd.DataFrame(site_rows)

    tables = {
        "sites": sites_df,
        "energy_30min": energy_30,
        "daily": daily,
        "tariff_cost": tariff_cost,
        "tariffs": tariffs,
        "esb_30min": esb,
        "truth_vs_export": tve,
    }
    for name, df in tables.items():
        df.to_parquet(out_dir / f"{name}.parquet", engine="pyarrow", index=False)
    db_path = out_dir / "hems.duckdb"
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    try:
        con.execute("SET TimeZone = 'UTC'")
        for name in TABLES:
            path = str((out_dir / f"{name}.parquet").resolve()).replace("\\", "/")
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM read_parquet('{path}')")
    finally:
        con.close()
    counts = {name: int(len(df)) for name, df in tables.items()}
    return {"sites": sites, "rows": counts, "parquet_dir": str(out_dir), "duckdb": str(db_path), "loaded": loaded}


def duckdb_sums(db_path: Path, table: str, columns: list[str]) -> dict[str, float]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        cols = ", ".join(f"SUM({c}) AS {c}" for c in columns)
        row = con.execute(f"SELECT {cols} FROM {table}").fetchone()
    finally:
        con.close()
    return {c: float(v) if v is not None else float("nan") for c, v in zip(columns, row)}
