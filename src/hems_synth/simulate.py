"""Five-second truth for one site: load, PV, battery, EV and grid flows.

Everything comes from ``numpy.random.default_rng`` seeded by the run seed and the profile seed, so a
run is reproducible. The clear-sky model is geometric (latitude, day of year, clock time), the cloud
factor is an AR(1) process at five-minute steps, and the battery follows a self-consumption dispatch.
No weather data, meter file or household record is read.

Sign conventions in the truth frame are explicit column names: ``import_w`` and ``export_w`` are both
non-negative, as are ``batt_charge_w`` and ``batt_discharge_w``. Device sign conventions are applied
by the exporters, not here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import TZ
from .profiles import SiteProfile

STEP_S = 5
STEPS_PER_MIN = 60 // STEP_S
FLOWS = ("import", "export", "pv", "load", "batt_charge", "batt_discharge", "ev")

Window = tuple[str, int]


@dataclass
class SimResult:
    profile: SiteProfile
    truth: pd.DataFrame
    half_hour: pd.DataFrame
    daily: pd.DataFrame
    ev_sessions: list[dict] = field(default_factory=list)
    windows: list[Window] = field(default_factory=list)
    seed: int = 0


def local_grid(windows: list[Window], step_s: int = STEP_S) -> pd.DatetimeIndex:
    """UTC index at ``step_s`` seconds covering whole local days for each (start date, days) window."""
    parts = []
    for start, days in windows:
        s = pd.Timestamp(start, tz=TZ)
        e = s + pd.DateOffset(days=int(days))
        parts.append(pd.date_range(s.tz_convert("UTC"), e.tz_convert("UTC"), freq=f"{step_s}s", inclusive="left").as_unit("ns"))
    idx = parts[0]
    for p in parts[1:]:
        idx = idx.append(p)
    if not idx.is_monotonic_increasing or idx.has_duplicates:
        raise ValueError("windows must be in order and must not overlap")
    return idx


def _local_hours(idx: pd.DatetimeIndex) -> np.ndarray:
    loc = idx.tz_convert(TZ)
    return loc.hour.to_numpy() + loc.minute.to_numpy() / 60.0 + loc.second.to_numpy() / 3600.0


def _gauss(h: np.ndarray, centre: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((h - centre) / width) ** 2)


def _daily_shape(h: np.ndarray, weekend: np.ndarray) -> np.ndarray:
    """Relative load shape over the local day, separate shapes for weekdays and weekends."""
    week = 1.0 + 0.8 * _gauss(h, 7.5, 0.9) + 0.15 * _gauss(h, 13.0, 2.0) + 1.5 * _gauss(h, 18.5, 1.6) - 0.35 * _gauss(h, 3.0, 2.5)
    wend = 1.0 + 0.9 * _gauss(h, 9.5, 1.3) + 0.5 * _gauss(h, 13.5, 2.0) + 1.3 * _gauss(h, 18.5, 1.8) - 0.35 * _gauss(h, 3.0, 2.5)
    return np.where(weekend, wend, week)


def _solar_elevation(idx: pd.DatetimeIndex, latitude: float, longitude: float) -> np.ndarray:
    """Sine of the solar elevation from a plain geometric model (declination, equation of time, hour angle)."""
    doy = idx.dayofyear.to_numpy().astype(float)
    utc_h = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0 + idx.second.to_numpy() / 3600.0
    b = 2 * np.pi * (doy - 81) / 365.0
    eot_min = 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)
    solar_h = utc_h + longitude / 15.0 + eot_min / 60.0
    hour_angle = np.deg2rad(15.0 * (solar_h - 12.0))
    decl = np.deg2rad(23.45 * np.sin(2 * np.pi * (284 + doy) / 365.0))
    lat = np.deg2rad(latitude)
    sin_el = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    return np.clip(sin_el, 0.0, 1.0)


def _clear_sky_ghi(sin_el: np.ndarray) -> np.ndarray:
    """Clear-sky global irradiance in W/m2 (Haurwitz form)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ghi = 1098.0 * sin_el * np.exp(-0.059 / np.where(sin_el > 0, sin_el, 1.0))
    return np.where(sin_el > 0.0, ghi, 0.0)


def _cloud_factor(idx: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """AR(1) cloud transmission at five-minute nodes with a daily mean, interpolated to the grid."""
    n = len(idx)
    nodes = np.arange(0, n, 60)
    node_days = idx[nodes].tz_convert(TZ).normalize()
    uniq_days = node_days.unique()
    doy = uniq_days.dayofyear.to_numpy().astype(float)
    season_mean = 0.58 + 0.12 * np.cos(2 * np.pi * (doy - 196) / 365.0)
    day_mean = np.clip(rng.normal(season_mean, 0.22), 0.08, 0.98)
    mean_by_day = dict(zip(uniq_days, day_mean))
    mu = np.array([mean_by_day[d] for d in node_days])
    rho, sigma = 0.9, 0.07
    eps = rng.normal(0.0, sigma, size=len(nodes))
    c = np.empty(len(nodes))
    c[0] = mu[0]
    for i in range(1, len(nodes)):
        c[i] = mu[i] + rho * (c[i - 1] - mu[i]) + eps[i]
    c = np.clip(c, 0.03, 1.0)
    return np.interp(np.arange(n), nodes, c)


def _appliance_events(idx: pd.DatetimeIndex, local_h: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random kettle, oven, washing machine, dishwasher and fridge cycles added to the base shape (W)."""
    n = len(idx)
    add = np.zeros(n)
    loc = idx.tz_convert(TZ)
    _, day_starts = np.unique(loc.normalize().to_numpy(), return_index=True)
    day_ends = np.r_[day_starts[1:], n]

    def pulse(start: int, minutes: float, watts: float, end_cap: int) -> None:
        stop = min(end_cap, start + int(minutes * STEPS_PER_MIN))
        if stop > start:
            add[start:stop] += watts

    def at_hour(d0: int, hours: np.ndarray, h: float) -> int:
        return d0 + int(np.searchsorted(hours, h))

    for d0, d1 in zip(day_starts, day_ends):
        hours = local_h[d0:d1]
        for _ in range(rng.poisson(5)):
            h = rng.choice([7.2, 8.0, 10.5, 13.0, 16.0, 18.0, 20.5, 21.5]) + rng.normal(0, 0.4)
            pulse(at_hour(d0, hours, h % 24), rng.uniform(2.5, 3.5), 2000.0, d1)
        if rng.random() < 0.5:
            pulse(at_hour(d0, hours, rng.uniform(17.0, 19.5)), rng.uniform(45, 75), 1400.0, d1)
        if rng.random() < 0.6:
            s = at_hour(d0, hours, rng.uniform(8.0, 20.0))
            pulse(s, 15, 2000.0, d1)
            pulse(s + 15 * STEPS_PER_MIN, 75, 300.0, d1)
        if rng.random() < 0.5:
            s = at_hour(d0, hours, rng.uniform(20.0, 22.0))
            pulse(s, 20, 1800.0, d1)
            pulse(s + 20 * STEPS_PER_MIN, 55, 100.0, d1)
    period = 40 * STEPS_PER_MIN
    phase = rng.integers(0, period)
    fridge = ((np.arange(n) + phase) % period) < int(0.4 * period)
    add += np.where(fridge, 100.0, 0.0)
    return add


def _ev_profile(idx: pd.DatetimeIndex, local_h: np.ndarray, profile: SiteProfile, rng: np.random.Generator) -> tuple[np.ndarray, list[dict]]:
    """EV charging power (W) at the charger rating until a nightly target is met, on some nights."""
    n = len(idx)
    ev = np.zeros(n)
    sessions: list[dict] = []
    if not profile.ev_charger_kw:
        return ev, sessions
    p_w = profile.ev_charger_kw * 1000.0
    loc = idx.tz_convert(TZ)
    _, day_starts = np.unique(loc.normalize().to_numpy(), return_index=True)
    p_session = min(1.0, profile.ev_nights_per_week / 7.0)
    for k, d0 in enumerate(day_starts):
        if rng.random() >= p_session:
            continue
        hours = local_h[d0 : day_starts[k + 1] if k + 1 < len(day_starts) else n]
        start_h = rng.uniform(22.0, 23.9)
        s = d0 + int(np.searchsorted(hours, start_h))
        target_kwh = rng.uniform(profile.ev_daily_kwh_min, profile.ev_daily_kwh_max)
        steps = int(round(target_kwh / p_w * 3600.0 / STEP_S * 1000.0))
        e = min(n, s + steps)
        if e <= s:
            continue
        ev[s:e] += p_w
        sessions.append(
            {
                "start_utc": idx[s].isoformat(),
                "end_utc": idx[e - 1].isoformat(),
                "start_local_date": str(loc[s].date()),
                "kwh": round((e - s) * p_w * STEP_S / 3600.0 / 1000.0, 3),
            }
        )
    return ev, sessions


def _battery(pv: np.ndarray, load: np.ndarray, profile: SiteProfile) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Self-consumption dispatch: charge from surplus, discharge to cover deficit, SoC within 5 % to 100 %."""
    n = len(pv)
    charge = np.zeros(n)
    discharge = np.zeros(n)
    soc_pct = np.full(n, np.nan)
    cap_wh = profile.battery_kwh * 1000.0
    if cap_wh <= 0:
        return charge, discharge, soc_pct
    p_max = profile.battery_power_kw * 1000.0
    dt_h = STEP_S / 3600.0
    eta = 0.95
    soc = 0.5 * cap_wh
    soc_min = 0.05 * cap_wh
    surplus = (pv - load).tolist()
    c_out = charge.tolist()
    d_out = discharge.tolist()
    s_out = soc_pct.tolist()
    for i in range(n):
        s = surplus[i]
        if s > 0.0:
            c = s if s < p_max else p_max
            room = (cap_wh - soc) / (eta * dt_h)
            if c > room:
                c = room
            soc += c * eta * dt_h
            c_out[i] = c
        elif s < 0.0:
            d = -s if -s < p_max else p_max
            avail = (soc - soc_min) * eta / dt_h
            if d > avail:
                d = avail if avail > 0.0 else 0.0
            soc -= d / eta * dt_h
            d_out[i] = d
        s_out[i] = soc / cap_wh * 100.0
    return np.asarray(c_out), np.asarray(d_out), np.asarray(s_out)


def simulate(profile: SiteProfile, windows: list[Window], seed: int = 42) -> SimResult:
    """Run the five-second model for the given windows and return the truth plus its aggregates."""
    rng = np.random.default_rng([int(seed), int(profile.seed)])
    idx = local_grid(windows)
    n = len(idx)
    loc = idx.tz_convert(TZ)
    local_h = _local_hours(idx)
    weekend = loc.dayofweek.to_numpy() >= 5
    doy = loc.dayofyear.to_numpy().astype(float)

    season = 1.0 + profile.winter_load_factor * (0.5 + 0.5 * np.cos(2 * np.pi * (doy - 15) / 365.0))
    occ = (profile.occupants / 3.0) ** 0.5
    base = profile.base_load_w * _daily_shape(local_h, weekend) * occ * season
    appliances = _appliance_events(idx, local_h, rng)
    ev, sessions = _ev_profile(idx, local_h, profile, rng)
    load = base + appliances + ev

    if profile.pv_kwp > 0:
        sin_el = _solar_elevation(idx, profile.latitude, profile.longitude)
        ghi = _clear_sky_ghi(sin_el)
        cloud = _cloud_factor(idx, rng)
        pv = np.minimum(profile.pv_kwp * 1000.0 * ghi / 1000.0 * 0.82 * cloud, profile.ac_limit_w)
        pv = np.where(pv < 5.0, 0.0, pv)
    else:
        pv = np.zeros(n)

    charge, discharge, soc = _battery(pv, load, profile)
    grid = load - pv + charge - discharge
    imp = np.maximum(grid, 0.0)
    exp = np.maximum(-grid, 0.0)

    truth = pd.DataFrame(
        {
            "load_w": load,
            "pv_w": pv,
            "batt_charge_w": charge,
            "batt_discharge_w": discharge,
            "import_w": imp,
            "export_w": exp,
            "ev_w": ev,
            "soc_pct": soc,
        },
        index=pd.DatetimeIndex(idx, name="ts_utc"),
    )
    half_hour = aggregate(truth, "30min")
    daily = aggregate_daily(truth, sessions)
    return SimResult(profile=profile, truth=truth, half_hour=half_hour, daily=daily, ev_sessions=sessions, windows=list(windows), seed=seed)


def _kwh_columns(frame: pd.DataFrame, factor: float) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for flow in FLOWS:
        out[f"{flow}_kwh"] = frame[f"{flow}_w"] * factor
    return out


def aggregate(truth: pd.DataFrame, freq: str) -> pd.DataFrame:
    """kWh per interval of ``freq`` (local boundaries, which coincide with UTC ones for whole half-hours)."""
    factor = STEP_S / 3600.0 / 1000.0
    kwh = _kwh_columns(truth, factor)
    g = kwh.groupby(truth.index.floor(freq)).sum()
    soc_end = truth["soc_pct"].groupby(truth.index.floor(freq)).last()
    g["soc_pct_end"] = soc_end
    g.index.name = "ts_utc"
    g.insert(0, "ts_local", g.index.tz_convert(TZ).tz_localize(None))
    return g


def aggregate_daily(truth: pd.DataFrame, sessions: list[dict] | None = None) -> pd.DataFrame:
    """kWh per local day plus self-consumption share and the number of EV sessions that started that day."""
    factor = STEP_S / 3600.0 / 1000.0
    kwh = _kwh_columns(truth, factor)
    days = truth.index.tz_convert(TZ).normalize().tz_localize(None)
    g = kwh.groupby(days).sum()
    g.index.name = "date_local"
    with np.errstate(divide="ignore", invalid="ignore"):
        g["self_consumption_pct"] = np.where(g["pv_kwh"] > 0, (g["pv_kwh"] - g["export_kwh"]) / g["pv_kwh"] * 100.0, np.nan)
    counts = pd.Series(0, index=g.index, dtype=int)
    for s in sessions or []:
        d = pd.Timestamp(s["start_local_date"])
        if d in counts.index:
            counts[d] += 1
    g["ev_sessions"] = counts
    return g


def write_truth(sim: SimResult, out_dir: Path) -> dict:
    """Write the truth frames and the EV sessions under ``out_dir`` and return the file list."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t5 = sim.truth.reset_index()
    t5.to_parquet(out_dir / "truth_5s.parquet", engine="pyarrow", index=False)
    hh = sim.half_hour.reset_index()
    hh.to_parquet(out_dir / "truth_30min.parquet", engine="pyarrow", index=False)
    dy = sim.daily.reset_index()
    dy.to_parquet(out_dir / "truth_daily.parquet", engine="pyarrow", index=False)
    (out_dir / "ev_sessions.json").write_text(json.dumps(sim.ev_sessions, indent=2), encoding="utf-8")
    meta = {
        "site_id": sim.profile.site_id,
        "seed": sim.seed,
        "windows": [[s, int(d)] for s, d in sim.windows],
        "steps": int(len(sim.truth)),
        "profile": sim.profile.as_dict(),
    }
    (out_dir / "profile.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def read_truth(folder: Path) -> dict:
    """Read back what ``write_truth`` wrote."""
    folder = Path(folder)
    t5 = pd.read_parquet(folder / "truth_5s.parquet").set_index("ts_utc")
    hh = pd.read_parquet(folder / "truth_30min.parquet").set_index("ts_utc")
    dy = pd.read_parquet(folder / "truth_daily.parquet").set_index("date_local")
    meta = json.loads((folder / "profile.json").read_text(encoding="utf-8"))
    sessions = json.loads((folder / "ev_sessions.json").read_text(encoding="utf-8"))
    artefacts_path = folder / "artefacts.json"
    artefacts = json.loads(artefacts_path.read_text(encoding="utf-8")) if artefacts_path.exists() else {}
    return {"truth": t5, "half_hour": hh, "daily": dy, "meta": meta, "ev_sessions": sessions, "artefacts": artefacts}
