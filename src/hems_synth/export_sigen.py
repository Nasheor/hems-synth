"""Sigenergy style export: station and inverter entities with a synthetic station prefix.

Traits reproduced from the sigen_data docstring: station and inverter sensors polled about every five
minutes and written on change, power in kW, entity names that carry the station prefix (read from the
export by the loader, never from code), a frient meter-LED reader that gives grid import only
(``summation_delivered`` kWh and ``instantaneous_demand`` W), station-level ``load_power`` and
``battery_power`` that under-read small flows (a dead band below about 0.5 kW), battery power positive
when charging, grid power positive when exporting, and placeholder phase B and C, EV charger and heat
pump entities (655.35 V, 0 kW).

Two choices are this generator's own and are stated here because the docstring leaves them open: the
station day counters and the state of charge are written at every poll (the documented two-reading
spike after midnight can only be cleaned by a centred median of five when zero readings surround it),
and the station also exposes a ``daily_export_energy`` counter so that export has a counter source.

Optional artefacts on the PV day counter, logged with timestamps: one-reading dips to one hundredth
of the value and two-reading spikes (yesterday's final value twice, then zero).
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
    write_entity_csv,
    write_on_change,
    write_on_value_change,
)
from .simulate import STEP_S, SimResult

POLL_STEPS = 300 // STEP_S
FRIENT_STEPS = 60 // STEP_S
PLACEHOLDER_V = 655.35


def dead_band(kw: np.ndarray) -> np.ndarray:
    """Station-level reading of a small flow: nothing below 0.17 kW, under-read up to about 0.5 kW, true above."""
    a = np.abs(kw)
    read = np.minimum(a, 1.55 * np.maximum(0.0, a - 0.17))
    return np.sign(kw) * read


def _poll_positions(idx: pd.DatetimeIndex, rng: np.random.Generator, step: int) -> np.ndarray:
    """Poll every ``step`` samples from a random offset, plus one poll at the start of every window."""
    n = len(idx)
    k0 = int(rng.integers(0, step))
    gaps = np.diff(idx.asi8) > (STEP_S * 1_000_000_000 * 2)
    starts = np.r_[0, np.nonzero(gaps)[0] + 1]
    return np.union1d(np.arange(k0, n, step), starts)


def export_sigenergy(sim: SimResult, out_folder: Path, seed: int, artefacts: bool = False) -> dict:
    """Write the Sigenergy style export for one site and return an inventory with the artefact log."""
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    p = sim.profile
    if not (p.station_prefix and p.inverter_id):
        raise ValueError("a sigenergy export needs station_prefix and inverter_id in the profile")
    station = f"{p.station_prefix}_"
    inverter = f"{p.station_prefix}_inverter_{p.inverter_id}_"
    frient = f"{p.frient_prefix or 'frient_meter_reader'}_"
    rng = np.random.default_rng([int(seed), int(p.seed), 11])
    log = ArtefactLog()
    t = sim.truth
    idx = t.index
    f = STEP_S / 3600.0 / 1000.0

    pv = t["pv_w"].to_numpy()
    load = t["load_w"].to_numpy()
    imp = t["import_w"].to_numpy()
    exp = t["export_w"].to_numpy()
    ch = t["batt_charge_w"].to_numpy()
    dis = t["batt_discharge_w"].to_numpy()
    soc = t["soc_pct"].to_numpy()
    batt_kw = (ch - dis) / 1000.0
    grid_kw = (exp - imp) / 1000.0

    polls = _poll_positions(idx, rng, POLL_STEPS)
    pt = idx[polls]
    m = len(polls)

    def noisy(x: np.ndarray, sd: float) -> np.ndarray:
        return x + rng.normal(0.0, sd, len(x))

    inventory: dict[str, int] = {}
    points = 0

    def put(entity: str, times: pd.DatetimeIndex, values, unit: str, decimals: int = 2) -> None:
        nonlocal points
        points += write_entity_csv(out_folder / f"{entity}.csv.gz", times, values, unit, decimals)
        inventory[entity] = int(len(times))

    def changed(entity: str, values: np.ndarray, unit: str, threshold: float, decimals: int = 2) -> None:
        keep = write_on_change(values, threshold)
        put(entity, pt[keep], values[keep], unit, decimals)

    changed(f"{station}pv_power", np.round(noisy(pv[polls] / 1000.0, 0.02), 2), "kW", 0.005)
    changed(f"{station}load_power", np.round(noisy(dead_band(load[polls] / 1000.0), 0.02), 2), "kW", 0.005)
    changed(f"{station}battery_power", np.round(noisy(dead_band(batt_kw[polls]), 0.02), 2), "kW", 0.005)
    changed(f"{station}grid_power", np.round(noisy(grid_kw[polls], 0.02), 2), "kW", 0.005)
    soc_poll = np.round(np.nan_to_num(soc[polls], nan=0.0))
    keep = write_on_value_change(soc_poll)
    put(f"{station}battery_soc", pt[keep], soc_poll[keep], "%", 0)

    counters = {
        "daily_pv_energy": daily_counter(pv * f, idx, TZ)[polls],
        "daily_charge_energy": daily_counter(ch * f, idx, TZ)[polls],
        "daily_discharge_energy": daily_counter(dis * f, idx, TZ)[polls],
        "daily_import_energy": daily_counter(imp * f, idx, TZ)[polls],
        "daily_export_energy": daily_counter(exp * f, idx, TZ)[polls],
    }
    pv_true = counters["daily_pv_energy"].copy()
    if artefacts:
        pv_polls = counters["daily_pv_energy"]
        local = pt.tz_convert(TZ)
        local_h = local.hour.to_numpy() + local.minute.to_numpy() / 60.0
        day_segs = day_segments(pt, TZ)
        # one-reading dips to about one hundredth of the value on a rising daytime counter, about one per
        # day, at least three polls apart so that each sits alone in its median window
        cand = np.nonzero((pv_polls > 1.0) & (pv[polls] > 0.0))[0]
        cand = cand[(cand > 2) & (cand < m - 3)]
        n_dips = min(len(cand), int(rng.poisson(len(day_segs))))
        chosen: list[int] = []
        if n_dips > 0:
            for j in rng.permutation(cand):
                if all(abs(int(j) - c) >= 3 for c in chosen):
                    chosen.append(int(j))
                if len(chosen) >= n_dips:
                    break
            for j in sorted(chosen):
                true_v = float(pv_polls[j])
                dip = round(true_v / 100.0, 2)
                pv_polls[j] = dip
                log.sigen_pv_dips.append({"entity": f"{station}daily_pv_energy", "time_utc": pt[j].isoformat(), "written_value": dip, "true_value": true_v})
        # two-reading spikes after midnight: yesterday's final value twice, then zero, about one every three days
        for k, (s, e) in enumerate(day_segs):
            if k == 0 or rng.random() >= 1.0 / 3.0:
                continue
            yesterday_final = float(pv_true[s - 1])
            if yesterday_final <= 0.5:
                continue
            window = np.nonzero((local_h[s:e] >= 0.5) & (local_h[s:e] <= 4.0) & (pv_true[s:e] == 0.0))[0]
            if len(window) < 4:
                continue
            j = s + int(rng.choice(window[:-3]))
            for jj in (j, j + 1):
                pv_polls[jj] = yesterday_final
                log.sigen_pv_spikes.append({"entity": f"{station}daily_pv_energy", "time_utc": pt[jj].isoformat(), "written_value": yesterday_final, "true_value": float(pv_true[jj])})
    for name, vals in counters.items():
        put(f"{station}{name}", pt, vals, "kWh", 2)

    inv_pv = np.round(noisy(pv[polls] / 1000.0, 0.01), 2)
    inv_active = np.round(noisy((pv - ch + dis)[polls] / 1000.0, 0.01), 2)
    inv_batt = np.round(noisy(batt_kw[polls], 0.01), 2)
    changed(f"{inverter}pv_power", inv_pv, "kW", 0.005)
    changed(f"{inverter}active_power", inv_active, "kW", 0.005)
    changed(f"{inverter}battery_power", inv_batt, "kW", 0.005)
    hours = polls * STEP_S / 3600.0
    phase_v = np.round(231.0 + 2.5 * np.sin(2 * np.pi * hours / 5.0) + rng.normal(0.0, 0.3, m), 1)
    changed(f"{inverter}phase_a_voltage", phase_v, "V", 0.1, 1)
    changed(f"{inverter}phase_a_current", np.round(np.abs(inv_active) * 1000.0 / phase_v, 2), "A", 0.01)
    temp = np.round(28.0 + 3.0 * np.abs(inv_active) + rng.normal(0.0, 0.2, m), 1)
    changed(f"{inverter}temperature", temp, "°C", 0.5, 1)

    first = pt[:1]
    put(f"{inverter}phase_b_voltage", first, [PLACEHOLDER_V], "V", 2)
    put(f"{inverter}phase_c_voltage", first, [PLACEHOLDER_V], "V", 2)
    put(f"{station}ev_charger_power", first, [0.0], "kW", 2)
    put(f"{station}heat_pump_power", first, [0.0], "kW", 2)

    fpolls = _poll_positions(idx, rng, FRIENT_STEPS)
    ft = idx[fpolls]
    summation = lifetime_counter(imp * f, p.lifetime_import_kwh)[fpolls]
    keep = write_on_value_change(summation)
    put(f"{frient}summation_delivered", ft[keep], summation[keep], "kWh", 2)
    demand = np.round(noisy(imp[fpolls], 4.0), 0)
    keep = write_on_change(demand, 5.0)
    put(f"{frient}instantaneous_demand", ft[keep], demand[keep], "W", 0)

    (out_folder / "artefacts.json").write_text(json.dumps(log.as_dict(), indent=2), encoding="utf-8")
    return {"family": "sigenergy", "entities": len(inventory), "points": int(points), "inventory": inventory, "artefacts": log.as_dict(), "prefixes": {"station": station, "inverter": inverter, "frient": frient}}
