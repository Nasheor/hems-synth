"""Sofar and Home Assistant style export: one ``inverter_<signal>.csv.gz`` per entity.

Shape (from the hems_data docstring): columns ``_time, field, unit, value``, UTC timestamps, a point
only when the value changes by more than a per-entity threshold (10 W for power, 0.1 V, 0.01 Hz,
0.01 kWh for counters). ``today_*`` counters reset at local midnight, ``total_*`` counters never
reset, ``device_state`` is a text state, and three placeholder entities carry fewer than 1,000
points, the way unused phase inputs do.

Optional artefacts, each logged to ``artefacts.json`` with its timestamp:
  * single-point counter spikes in the documented shape (a point jumps away, the next point carries
    on from the previous one),
  * one silent gap of 15 minutes or longer across every entity per month,
  * one negative single-point glitch on the lifetime import counter per month.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import TZ
from .export_common import (
    ArtefactLog,
    daily_counter,
    day_segments,
    lifetime_counter,
    month_segments,
    write_entity_csv,
    write_on_change,
    write_on_value_change,
)
from .simulate import STEP_S, SimResult

PREFIX = "inverter_"
THRESHOLDS = {"W": 10.0, "V": 0.1, "A": 0.01, "Hz": 0.01, "°C": 0.5}
DECIMALS = {"W": 1, "V": 1, "A": 2, "Hz": 2, "°C": 1, "kWh": 2}
STEPS_PER_MIN = 60 // STEP_S


def _signals(sim: SimResult, rng: np.random.Generator) -> dict[str, tuple[np.ndarray, str]]:
    """Sensor readings (truth plus sensor noise) for every entity, keyed by signal name."""
    t = sim.truth
    n = len(t)
    p = sim.profile
    pv = t["pv_w"].to_numpy()
    load = t["load_w"].to_numpy()
    imp = t["import_w"].to_numpy()
    exp = t["export_w"].to_numpy()
    ch = t["batt_charge_w"].to_numpy()
    dis = t["batt_discharge_w"].to_numpy()
    grid = exp - imp
    inv_power = pv - ch + dis
    losses = 0.02 * np.abs(inv_power) + np.where(pv > 0, 12.0, 0.0)
    hours = np.arange(n) * STEP_S / 3600.0
    doy = t.index.tz_convert(TZ).dayofyear.to_numpy().astype(float)
    local_h = t.index.tz_convert(TZ).hour.to_numpy() + t.index.tz_convert(TZ).minute.to_numpy() / 60.0

    def noisy(x: np.ndarray, sd: float) -> np.ndarray:
        return x + rng.normal(0.0, sd, n)

    voltage = 230.0 + 3.0 * np.sin(2 * np.pi * hours / 6.0) + noisy(np.zeros(n), 0.3)
    current = np.abs(inv_power) / voltage
    freq = 50.0 + noisy(np.zeros(n), 0.008)
    pv1 = 0.55 * pv
    pv2 = 0.45 * pv
    pv1_v = np.where(pv1 > 0, 330.0 + 0.01 * pv1 + noisy(np.zeros(n), 1.0), 0.0)
    pv1_i = np.where(pv1_v > 0, pv1 / np.where(pv1_v > 0, pv1_v, 1.0), 0.0)
    ambient = 10.0 + 6.0 * np.cos(2 * np.pi * (doy - 200) / 365.0) + 4.0 * np.sin(2 * np.pi * (local_h - 9) / 24.0)
    inverter_t = ambient + 15.0 + 0.004 * np.abs(inv_power)
    module_t = ambient + 0.02 * pv
    radiator_t = inverter_t + 3.0
    room_t = 20.0 + 1.5 * np.sin(2 * np.pi * (local_h - 10) / 24.0)
    sig: dict[str, tuple[np.ndarray, str]] = {
        "grid_power": (noisy(grid, 8.0), "W"),
        "load_power": (noisy(load, 8.0), "W"),
        "pv_power": (noisy(pv, 6.0) if p.pv_kwp > 0 else pv, "W"),
        "power": (noisy(inv_power, 8.0) if p.pv_kwp > 0 or p.battery_kwh > 0 else inv_power, "W"),
        "pv1_power": (noisy(pv1, 4.0) if p.pv_kwp > 0 else pv1, "W"),
        "pv2_power": (noisy(pv2, 4.0) if p.pv_kwp > 0 else pv2, "W"),
        "power_losses": (losses, "W"),
        "l1_voltage": (voltage, "V"),
        "l1_current": (current, "A"),
        "frequency": (freq, "Hz"),
        "pv1_voltage": (pv1_v, "V"),
        "pv1_current": (pv1_i, "A"),
        "temperature": (noisy(inverter_t, 0.2), "°C"),
        "module_temperature": (noisy(module_t, 0.2), "°C"),
        "radiator_temperature": (noisy(radiator_t, 0.2), "°C"),
        "room_temperature": (noisy(room_t, 0.1), "°C"),
    }
    return sig


def _counters(sim: SimResult) -> dict[str, np.ndarray]:
    t = sim.truth
    f = STEP_S / 3600.0 / 1000.0
    imp = t["import_w"].to_numpy() * f
    exp = t["export_w"].to_numpy() * f
    p = sim.profile
    return {
        "today_energy_import": daily_counter(imp, t.index, TZ),
        "today_energy_export": daily_counter(exp, t.index, TZ),
        "total_energy_import": lifetime_counter(imp, p.lifetime_import_kwh),
        "total_energy_export": lifetime_counter(exp, p.lifetime_export_kwh),
    }


def _pick_gaps(idx: pd.DatetimeIndex, rng: np.random.Generator) -> list[tuple[int, int]]:
    """One silent gap per local calendar month, 15 to 25 minutes, between 06:00 and 22:00 local."""
    local_h = idx.tz_convert(TZ).hour.to_numpy()
    gaps = []
    for s, e in month_segments(idx, TZ):
        ok = np.nonzero((local_h[s:e] >= 6) & (local_h[s:e] < 22))[0]
        if len(ok) == 0:
            continue
        minutes = int(rng.integers(15, 26))
        start = s + int(rng.choice(ok))
        end = min(e, start + minutes * STEPS_PER_MIN)
        gaps.append((start, end))
    return gaps


def _apply_gaps(keep: np.ndarray, gaps: list[tuple[int, int]]) -> np.ndarray:
    if not gaps:
        return keep
    mask = np.ones(len(keep), dtype=bool)
    for s, e in gaps:
        mask &= ~((keep >= s) & (keep < e))
    return keep[mask]


def _spike_positions(keep: np.ndarray, values: np.ndarray, idx: pd.DatetimeIndex, gaps: list[tuple[int, int]], count: int, rng: np.random.Generator) -> np.ndarray:
    """Written points that may carry a spike: counter above 0.5 kWh, next write within a minute, away from midnight and gaps."""
    if len(keep) < 3 or count <= 0:
        return np.zeros(0, dtype=np.int64)
    local = idx[keep].tz_convert(TZ)
    minute_of_day = local.hour.to_numpy() * 60 + local.minute.to_numpy()
    nxt = np.r_[keep[1:], keep[-1]]
    ok = (values > 0.5) & (minute_of_day > 10) & (minute_of_day < 24 * 60 - 10) & ((nxt - keep) <= STEPS_PER_MIN)
    ok[0] = False
    ok[-1] = False
    for s, e in gaps:
        ok &= ~((keep >= s - 30 * STEPS_PER_MIN) & (keep < e + 30 * STEPS_PER_MIN))
    cand = np.nonzero(ok)[0]
    if len(cand) == 0:
        return np.zeros(0, dtype=np.int64)
    pick = rng.choice(cand, size=min(count, len(cand)), replace=False)
    return np.sort(pick)


def export_sofar(sim: SimResult, out_folder: Path, seed: int, artefacts: bool = False) -> dict:
    """Write the Sofar style export for one site and return an inventory with the artefact log."""
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng([int(seed), int(sim.profile.seed), 7])
    idx = sim.truth.index
    n = len(idx)
    log = ArtefactLog()
    gaps = _pick_gaps(idx, rng) if artefacts else []
    for s, e in gaps:
        log.silent_gaps.append({"start_utc": idx[s].isoformat(), "end_utc": idx[e - 1].isoformat() if e - 1 < n else idx[-1].isoformat(), "minutes": round((e - s) * STEP_S / 60.0, 2)})

    inventory: dict[str, int] = {}
    points = 0

    for name, (vals, unit) in _signals(sim, rng).items():
        keep = write_on_change(vals, THRESHOLDS[unit])
        keep = _apply_gaps(keep, gaps)
        points += write_entity_csv(out_folder / f"{PREFIX}{name}.csv.gz", idx[keep], vals[keep], unit, DECIMALS[unit])
        inventory[f"{PREFIX}{name}"] = int(len(keep))

    days = len(day_segments(idx, TZ))
    for name, vals in _counters(sim).items():
        keep = write_on_value_change(vals)
        keep = _apply_gaps(keep, gaps)
        written = vals[keep].copy()
        if artefacts and name.startswith("today_"):
            count = int(rng.poisson(0.5 * days))
            for pos in _spike_positions(keep, written, idx, gaps, count, rng):
                true_v = float(written[pos])
                spike = round(true_v + float(rng.uniform(0.5, 3.0)), 2)
                written[pos] = spike
                log.counter_spikes.append({"entity": f"{PREFIX}{name}", "time_utc": idx[keep[pos]].isoformat(), "written_value": spike, "true_value": true_v})
        if artefacts and name == "total_energy_import":
            for s, e in month_segments(idx, TZ):
                in_month = np.nonzero((keep >= s) & (keep < e))[0]
                if len(in_month) < 3:
                    continue
                sub_keep = keep[in_month]
                sub_vals = written[in_month]
                pos = _spike_positions(sub_keep, sub_vals, idx, gaps, 1, rng)
                if len(pos) == 0:
                    continue
                j = int(in_month[pos[0]])
                true_v = float(written[j])
                glitch = round(true_v - float(rng.uniform(0.5, 5.0)), 2)
                written[j] = glitch
                log.lifetime_glitches.append({"entity": f"{PREFIX}{name}", "time_utc": idx[keep[j]].isoformat(), "written_value": glitch, "true_value": true_v})
        points += write_entity_csv(out_folder / f"{PREFIX}{name}.csv.gz", idx[keep], written, "kWh", 2)
        inventory[f"{PREFIX}{name}"] = int(len(keep))

    t = sim.truth
    running = (t["pv_w"].to_numpy() > 0) | (t["batt_charge_w"].to_numpy() > 0) | (t["batt_discharge_w"].to_numpy() > 0)
    state = np.where(running, "normal", "standby")
    keep = _apply_gaps(write_on_value_change(state), gaps)
    points += write_entity_csv(out_folder / f"{PREFIX}device_state.csv.gz", idx[keep], state[keep], "", field="state")
    inventory[f"{PREFIX}device_state"] = int(len(keep))

    day_starts = np.array([s for s, _ in day_segments(idx, TZ)])
    for name, unit in (("l2_voltage", "V"), ("l3_voltage", "V"), ("l2_current", "A")):
        points += write_entity_csv(out_folder / f"{PREFIX}{name}.csv.gz", idx[day_starts], np.zeros(len(day_starts)), unit, DECIMALS[unit])
        inventory[f"{PREFIX}{name}"] = int(len(day_starts))

    (out_folder / "artefacts.json").write_text(json.dumps(log.as_dict(), indent=2), encoding="utf-8")
    return {"family": "sofar", "entities": len(inventory), "points": int(points), "inventory": inventory, "artefacts": log.as_dict()}
