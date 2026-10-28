"""Reference loader: the pilot's loading and cleaning logic re-implemented from the docstrings of
smartcoremtu/ems (``hems_data``, ``sigen_data``, ``reference_data`` and the EV rule of ``metrics``).

Names and signatures are kept so that a later unit can port the identical logic. No code was copied:
every rule below is written from its one-paragraph description.

Sofar and Home Assistant export (``hems_data``): one gzipped CSV per entity, columns
``_time, field, unit, value``, UTC timestamps. Home Assistant writes a point only when a value
changes, so a long run without points means "unchanged", not "no data". A run of ``ALIVE_WINDOW_MIN``
minutes with no point from any entity is a real gap and becomes NaN. Entities with fewer than
``MIN_POINTS`` points are unused inputs.

Sigenergy export (``sigen_data``): same file shape, entities named after the station, polled about
every five minutes, power in kW, battery power positive when charging, grid power positive when
exporting. Station-level power under-reads small flows, so energy comes from counters and the energy
balance. The PV day counter is cleaned with a centred median of five.

ESB Networks HDF (``reference_data``): one row per register per half-hour, ``Read Date and End Time``
is the end of the interval in Irish clock time, labelled by interval start on output.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import TZ
from .export_common import read_entity_csv

PREFIX = "inverter_"
ALIVE_WINDOW_MIN = 15
MIN_POINTS = 1000
POWER = ["grid_power", "load_power", "pv_power", "power", "pv1_power", "pv2_power", "power_losses"]
ELECTRICAL = ["l1_voltage", "l1_current", "frequency", "pv1_voltage", "pv1_current"]
THERMAL = ["temperature", "module_temperature", "radiator_temperature", "room_temperature"]
DAILY_COUNTERS = ["today_energy_import", "today_energy_export"]
LIFETIME_COUNTERS = ["total_energy_import", "total_energy_export"]
STATES = ["device_state", "device_fault"]
SPIKE_JUMP_KWH = 0.2
SPIKE_CONTINUE_KWH = 0.05

EV_THRESHOLD_W = 5000.0
EV_MERGE_MIN = 10
EV_BASE_W = 350.0
IMPORT_LIMIT_W = 12000.0

FINE_PER_MIN = 12


# ----------------------------------------------------------------------------------------------
# Sofar and Home Assistant export
# ----------------------------------------------------------------------------------------------


def read_entity(folder: Path, name: str) -> pd.Series:
    """Numeric series for one entity, UTC index, duplicates dropped."""
    return read_entity_csv(Path(folder) / f"{PREFIX}{name}.csv.gz", "value")


def read_states(folder: Path, name: str) -> pd.Series:
    """Text state series (device_state, device_fault)."""
    return read_entity_csv(Path(folder) / f"{PREFIX}{name}.csv.gz", "state")


def _minute_mean(s: pd.Series, grid: pd.DatetimeIndex) -> pd.Series:
    """Time-weighted one-minute mean of a write-on-change series (hold last value, five-second steps)."""
    if len(s) == 0:
        return pd.Series(np.nan, index=grid)
    fine = pd.date_range(grid[0], grid[-1] + pd.Timedelta(seconds=55), freq="5s")
    held = s.reindex(fine, method="ffill").to_numpy(dtype=float)
    vals = held.reshape(len(grid), FINE_PER_MIN).mean(axis=1)
    return pd.Series(vals, index=grid)


def _despike(s: pd.Series) -> pd.Series:
    """Drop single bad reads of a counter: a point that jumps away from its neighbours while the next
    point carries on from the previous one. A midnight reset is not a spike because the point after
    it stays low."""
    if len(s) < 3:
        return s
    prev = s.shift(1)
    nxt = s.shift(-1)
    jump = (s - prev).abs()
    cont = (nxt - prev).abs()
    spike = ((jump > SPIKE_JUMP_KWH) & (cont <= SPIKE_CONTINUE_KWH)).fillna(False)
    return s[~spike.to_numpy()]


def _counter_to_cumulative(s: pd.Series) -> pd.Series:
    """Turn a daily-resetting counter into a monotonic cumulative series.

    A drop to below half the previous value is the midnight reset (the new value is the energy since
    the reset). Any other negative step is a glitch and counts as zero: the counter is then read
    against the last accepted value, so a single bad read adds nothing on either side."""
    v = s.to_numpy(dtype=float).tolist()
    n = len(v)
    out = [0.0] * n
    if n == 0:
        return pd.Series(out, index=s.index, dtype=float)
    total = 0.0
    ref = v[0]
    for i in range(1, n):
        x = v[i]
        if x < 0.5 * ref:
            total += x
            ref = x
        elif x < ref:
            pass
        else:
            total += x - ref
            ref = x
        out[i] = total
    return pd.Series(out, index=s.index, dtype=float)


def _union_times(series: list[pd.Series]) -> pd.DatetimeIndex:
    """Sorted unique union of the UTC timestamps of several series."""
    parts = [s.index.as_unit("ns").asi8 for s in series if len(s)]
    ints = np.unique(np.concatenate(parts))
    return pd.DatetimeIndex(ints.view("datetime64[ns]")).tz_localize("UTC")


def _dead_minutes(all_times: pd.DatetimeIndex, grid: pd.DatetimeIndex) -> np.ndarray:
    """True for minute slots inside a run of ``ALIVE_WINDOW_MIN`` minutes or more without any point."""
    dead = np.zeros(len(grid), dtype=bool)
    if len(all_times) < 2:
        return dead
    t = all_times.sort_values()
    diffs = np.diff(t.asi8)
    limit = ALIVE_WINDOW_MIN * 60 * 1_000_000_000
    for k in np.nonzero(diffs >= limit)[0]:
        a = t[k].floor("min") + pd.Timedelta(minutes=1)
        b = t[k + 1].floor("min")
        i0 = int(grid.searchsorted(a))
        i1 = int(grid.searchsorted(b))
        dead[i0:i1] = True
    return dead


def _cumulative_at(cum: pd.Series, bounds: pd.DatetimeIndex) -> pd.Series:
    """Value of a cumulative series at each bound: the last point at or before it, the first value before that."""
    return cum.reindex(bounds, method="ffill").bfill()


def _interval_from_cumulative(cum: pd.Series, grid: pd.DatetimeIndex, freq: str) -> pd.Series:
    """kWh per interval of ``freq`` from a cumulative series: the difference between its values at the
    interval bounds, read from the points themselves so that no partial minute is lost."""
    bounds = pd.date_range(grid[0].floor(freq), grid[-1].floor(freq) + pd.Timedelta(freq), freq=freq)
    at = _cumulative_at(cum, bounds)
    energy = at.diff().shift(-1).iloc[:-1]
    energy.index = bounds[:-1]
    return energy


def ev_sessions(load_w: pd.Series) -> pd.DataFrame:
    """EV sessions from minute house load: EV detected above 5,000 W, minutes closer than 10 minutes
    merge into one session, 350 W base load subtracted for the EV-only energy, load capped at the
    12 kW import limit."""
    load = load_w.clip(upper=IMPORT_LIMIT_W)
    on = (load > EV_THRESHOLD_W).fillna(False).to_numpy()
    idx = load.index
    rows = []
    starts = np.nonzero(on & ~np.r_[False, on[:-1]])[0]
    ends = np.nonzero(on & ~np.r_[on[1:], False])[0]
    merged: list[list[int]] = []
    for s, e in zip(starts, ends):
        if merged and (s - merged[-1][1]) < EV_MERGE_MIN:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    for s, e in merged:
        seg = load.iloc[s : e + 1]
        kwh = float(((seg - EV_BASE_W).clip(lower=0.0)).sum() / 60.0 / 1000.0)
        rows.append({"start": idx[s], "end": idx[e], "minutes": int(e - s + 1), "kwh": round(kwh, 3)})
    return pd.DataFrame(rows, columns=["start", "end", "minutes", "kwh"])


def _ev_minute_w(load_w: pd.Series, sessions: pd.DataFrame) -> pd.Series:
    ev = pd.Series(0.0, index=load_w.index)
    for _, r in sessions.iterrows():
        seg = load_w.loc[r["start"] : r["end"]]
        ev.loc[r["start"] : r["end"]] = (seg - EV_BASE_W).clip(lower=0.0)
    return ev


def load_site(folder: Path, cache: Path | None = None) -> dict:
    """Return {'minute': 1-min frame, 'energy15' and 'energy30': 15- and 30-min kWh frames,
    'totals': counter deltas per local day, 'states': dict, 'inventory': per-entity point counts}.
    All indexes are Europe/Dublin.

    Energy columns: ``pv`` and ``load`` are integrated minute power, ``import`` and ``export`` come
    from the day counters (``import_power`` and ``export_power`` are the integrated grid power kept
    for comparison), ``batt_charge`` and ``batt_discharge`` from the difference between PV power and
    inverter output power, ``ev`` from the EV session rule."""
    folder = Path(folder)
    files = sorted(folder.glob(f"{PREFIX}*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"no {PREFIX}*.csv.gz files in {folder}")
    inventory: dict[str, dict] = {}
    numeric: dict[str, pd.Series] = {}
    states: dict[str, pd.Series] = {}
    for f in files:
        name = f.name[len(PREFIX) : -len(".csv.gz")]
        if name in STATES:
            s = read_states(folder, name)
            states[name] = s
        else:
            s = read_entity(folder, name)
            numeric[name] = s
        inventory[f"{PREFIX}{name}"] = {"points": int(len(s)), "used": bool(len(s) >= MIN_POINTS)}

    all_times = _union_times(list(numeric.values()) + list(states.values()))
    grid = pd.date_range(all_times[0].floor("min"), all_times[-1].floor("min"), freq="min").as_unit("ns")
    dead = _dead_minutes(all_times, grid)

    minute = pd.DataFrame(index=grid)
    for name in POWER + ELECTRICAL + THERMAL:
        s = numeric.get(name)
        if s is None or len(s) < MIN_POINTS:
            minute[name] = np.nan
            continue
        col = _minute_mean(s, grid)
        col[dead] = np.nan
        minute[name] = col
    minute["alive"] = ~dead

    cumulative: dict[str, pd.Series] = {}
    for name in DAILY_COUNTERS + LIFETIME_COUNTERS:
        s = numeric.get(name)
        if s is None or len(s) == 0:
            continue
        cumulative[name] = _counter_to_cumulative(_despike(s))

    load_minute = minute["load_power"]
    sessions = ev_sessions(load_minute) if load_minute.notna().any() else pd.DataFrame(columns=["start", "end", "minutes", "kwh"])
    ev_w = _ev_minute_w(load_minute, sessions) if len(sessions) else pd.Series(0.0, index=grid)

    def unused_as_zero(name: str) -> pd.Series:
        col = minute[name]
        if col.isna().all():
            return pd.Series(0.0, index=grid).where(~dead)
        return col

    pv_min = unused_as_zero("pv_power")
    power_min = unused_as_zero("power")

    def energy(freq: str) -> pd.DataFrame:
        hours = pd.Timedelta(freq) / pd.Timedelta(hours=1)
        base = pd.DataFrame(
            {
                "pv": pv_min,
                "load": minute["load_power"],
                "import_power": (-minute["grid_power"]).clip(lower=0.0),
                "export_power": minute["grid_power"].clip(lower=0.0),
                "batt_charge": (pv_min - power_min).clip(lower=0.0),
                "batt_discharge": (power_min - pv_min).clip(lower=0.0),
                "ev": ev_w,
            }
        )
        e = base.resample(freq).mean() * hours / 1000.0
        for col, name in (("import", "today_energy_import"), ("export", "today_energy_export")):
            if name in cumulative:
                e[col] = _interval_from_cumulative(cumulative[name], grid, freq).reindex(e.index)
            else:
                e[col] = e[f"{col}_power"]
        e["soc_pct_end"] = np.nan
        cols = ["pv", "load", "import", "export", "batt_charge", "batt_discharge", "ev", "import_power", "export_power", "soc_pct_end"]
        return e[cols]

    energy15 = energy("15min")
    energy30 = energy("30min")

    local_days = grid.tz_convert(TZ).normalize().unique()
    day_bounds = local_days.append(pd.DatetimeIndex([local_days[-1] + pd.DateOffset(days=1)])).tz_convert("UTC")
    totals = pd.DataFrame(index=pd.Index(local_days.tz_localize(None), name="date_local"))
    for name in DAILY_COUNTERS + LIFETIME_COUNTERS:
        if name not in cumulative:
            continue
        at = _cumulative_at(cumulative[name], day_bounds)
        totals[name.replace("energy_", "") + "_kwh"] = at.diff().shift(-1).iloc[:-1].to_numpy()

    def localise(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.index = out.index.tz_convert(TZ)
        return out

    return {
        "minute": localise(minute),
        "energy15": localise(energy15),
        "energy30": localise(energy30),
        "totals": totals,
        "states": {k: v.tz_convert(TZ) for k, v in states.items()},
        "inventory": inventory,
        "cumulative": {k: v.tz_convert(TZ) for k, v in cumulative.items()},
        "ev_sessions": sessions,
    }


# ----------------------------------------------------------------------------------------------
# Sigenergy export
# ----------------------------------------------------------------------------------------------


def prefixes(folder: Path) -> tuple[str, str]:
    """(station, inverter) entity prefixes read from the export files, e.g. ('<station>_', '<station>_inverter_<id>_')."""
    folder = Path(folder)
    for f in sorted(folder.glob("*_inverter_*.csv.gz")):
        head, _, tail = f.name.partition("_inverter_")
        inverter_id = tail.split("_")[0]
        return f"{head}_", f"{head}_inverter_{inverter_id}_"
    raise FileNotFoundError(f"no *_inverter_* entity file in {folder}")


def _frient_prefix(folder: Path) -> str | None:
    for f in sorted(Path(folder).glob("*summation_delivered.csv.gz")):
        return f.name[: -len("summation_delivered.csv.gz")]
    return None


def entities(folder: Path) -> dict:
    """Signal name to entity, per group. kW entities are converted to W on load."""
    folder = Path(folder)
    station, inverter = prefixes(folder)
    frient = _frient_prefix(folder)
    out: dict[str, dict[str, str]] = {"station": {}, "inverter": {}, "frient": {}}
    for f in sorted(folder.glob("*.csv.gz")):
        entity = f.name[: -len(".csv.gz")]
        if entity.startswith(inverter):
            out["inverter"][entity[len(inverter) :]] = entity
        elif entity.startswith(station):
            out["station"][entity[len(station) :]] = entity
        elif frient and entity.startswith(frient):
            out["frient"][entity[len(frient) :]] = entity
    return out


def read_sigen_entity(folder: Path, entity: str, field: str = "value") -> pd.Series:
    """One entity of a Sigenergy export by its full entity name."""
    return read_entity_csv(Path(folder) / f"{entity}.csv.gz", field)


def _clean_daily_counter(s: pd.Series) -> pd.Series:
    """Centred rolling median over five readings. Removes one-reading dips to about one hundredth of the
    value and two-reading spikes (yesterday's value twice, then zero). On a rising counter the median is
    the reading itself. Used on the PV counter only."""
    return s.rolling(5, center=True, min_periods=1).median()


def _held(s: pd.Series, grid: pd.DatetimeIndex) -> pd.Series:
    """Hold-last-value at one minute (the polls are five minutes apart, so a finer grid adds nothing)."""
    return s.reindex(grid, method="ffill")


def load_site_sigen(folder: Path, cache: Path | None = None) -> dict:
    """{'minute', 'energy15', 'energy30', 'lifetime', 'inventory'}, indexes Europe/Dublin.

    energy15 columns (kWh): pv, batt_charge, batt_discharge (Sigenergy counters, the PV counter cleaned
    with the median of five), import (frient meter reader), export (station export counter, with
    ``export_power`` the integrated grid power kept for comparison), load (energy balance), plus
    import_sigen and load_sensor (the Sigenergy power readings, kept for comparison)."""
    folder = Path(folder)
    ent = entities(folder)
    inventory: dict[str, dict] = {}
    raw: dict[str, pd.Series] = {}
    for group in ent.values():
        for entity in group.values():
            s = read_sigen_entity(folder, entity)
            raw[entity] = s
            inventory[entity] = {"points": int(len(s)), "used": bool(len(s) > 1)}
    all_times = _union_times(list(raw.values()))
    grid = pd.date_range(all_times[0].floor("min"), all_times[-1].floor("min"), freq="min").as_unit("ns")
    dead = _dead_minutes(all_times, grid)

    st, inv, fr = ent["station"], ent["inverter"], ent["frient"]

    def held_w(entity: str | None, kw: bool = True) -> pd.Series:
        if entity is None or len(raw[entity]) <= 1:
            return pd.Series(np.nan, index=grid)
        h = _held(raw[entity], grid)
        if kw:
            h = h * 1000.0
        h[dead] = np.nan
        return h

    minute = pd.DataFrame(index=grid)
    minute["pv_power"] = held_w(st.get("pv_power"))
    minute["load_power"] = held_w(st.get("load_power"))
    minute["battery_power"] = held_w(st.get("battery_power"))
    minute["grid_power"] = held_w(st.get("grid_power"))
    minute["battery_soc"] = held_w(st.get("battery_soc"), kw=False)
    minute["inverter_pv_power"] = held_w(inv.get("pv_power"))
    minute["inverter_active_power"] = held_w(inv.get("active_power"))
    minute["inverter_battery_power"] = held_w(inv.get("battery_power"))
    minute["frient_demand"] = held_w(fr.get("instantaneous_demand"), kw=False)
    minute["alive"] = ~dead

    raw_counters = {k: raw[st[k]] for k in ("daily_pv_energy", "daily_charge_energy", "daily_discharge_energy", "daily_import_energy", "daily_export_energy") if k in st}
    clean_counters = dict(raw_counters)
    if "daily_pv_energy" in clean_counters:
        clean_counters["daily_pv_energy"] = _clean_daily_counter(raw_counters["daily_pv_energy"])
    cumulative = {k: _counter_to_cumulative(v) for k, v in clean_counters.items()}
    lifetime = raw[fr["summation_delivered"]] if "summation_delivered" in fr else None

    def energy(freq: str) -> pd.DataFrame:
        hours = pd.Timedelta(freq) / pd.Timedelta(hours=1)
        e = pd.DataFrame(index=pd.date_range(grid[0].floor(freq), grid[-1].floor(freq), freq=freq))
        e["pv"] = _interval_from_cumulative(cumulative["daily_pv_energy"], grid, freq).reindex(e.index)
        e["batt_charge"] = _interval_from_cumulative(cumulative["daily_charge_energy"], grid, freq).reindex(e.index)
        e["batt_discharge"] = _interval_from_cumulative(cumulative["daily_discharge_energy"], grid, freq).reindex(e.index)
        if lifetime is not None:
            e["import"] = _interval_from_cumulative(lifetime, grid, freq).reindex(e.index)
        else:
            e["import"] = _interval_from_cumulative(cumulative["daily_import_energy"], grid, freq).reindex(e.index)
        e["export"] = _interval_from_cumulative(cumulative["daily_export_energy"], grid, freq).reindex(e.index)
        e["load"] = e["pv"] + e["import"] + e["batt_discharge"] - e["export"] - e["batt_charge"]
        power = pd.DataFrame(
            {
                "export_power": minute["grid_power"].clip(lower=0.0),
                "import_sigen": (-minute["grid_power"]).clip(lower=0.0),
                "load_sensor": minute["load_power"],
            }
        ).resample(freq).mean() * hours / 1000.0
        e = e.join(power)
        e["ev"] = 0.0
        e["soc_pct_end"] = minute["battery_soc"].resample(freq).last()
        return e[["pv", "load", "import", "export", "batt_charge", "batt_discharge", "ev", "export_power", "import_sigen", "load_sensor", "soc_pct_end"]]

    energy15 = energy("15min")
    energy30 = energy("30min")

    def localise(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.index = out.index.tz_convert(TZ)
        return out

    return {
        "minute": localise(minute),
        "energy15": localise(energy15),
        "energy30": localise(energy30),
        "lifetime": lifetime.tz_convert(TZ) if lifetime is not None else None,
        "inventory": inventory,
        "prefixes": {"station": prefixes(folder)[0], "inverter": prefixes(folder)[1], "frient": _frient_prefix(folder)},
        "raw_counters": {k: v.tz_convert(TZ) for k, v in raw_counters.items()},
        "clean_counters": {k: v.tz_convert(TZ) for k, v in clean_counters.items()},
    }


# ----------------------------------------------------------------------------------------------
# ESB Networks HDF
# ----------------------------------------------------------------------------------------------


def read_esb_hdf(path: Path, clock: str = "local") -> pd.DataFrame:
    """Half-hourly import and export (kWh, or kW for the kW file), labelled by interval START, Europe/Dublin.

    clock: "utc" = file times are UTC all year. "local" = Irish clock time, the ambiguous hour at the
    end of summer time is dropped because the file carries no marker for it."""
    if clock not in ("utc", "local"):
        raise ValueError("clock must be 'utc' or 'local'")
    df = pd.read_csv(path, dtype=str)
    df["Read Value"] = pd.to_numeric(df["Read Value"], errors="coerce")
    raw = df["Read Date and End Time"].str.strip()
    fmt = "%d/%m/%Y %H:%M" if raw.str.contains("/").any() else "%d-%m-%Y %H:%M"
    end_naive = pd.to_datetime(raw, format=fmt)
    if clock == "utc":
        end = end_naive.dt.tz_localize("UTC")
    else:
        end = end_naive.dt.tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
    df = df.assign(_end=end).dropna(subset=["_end"])
    df["_start"] = (df["_end"] - pd.Timedelta(minutes=30)).dt.tz_convert(TZ)
    df["_register"] = np.where(df["Read Type"].str.contains("Export"), "export", "import")
    unit = "kWh" if df["Read Type"].str.contains(r"\(kWh\)").any() else "kW"
    out = df.pivot_table(index="_start", columns="_register", values="Read Value", aggfunc="first").sort_index()
    out.index.name = "ts_local"
    out.columns.name = None
    for col in ("import", "export"):
        if col not in out.columns:
            out[col] = np.nan
    out = out[["import", "export"]]
    out.attrs["unit"] = unit
    out.attrs["clock"] = clock
    return out
