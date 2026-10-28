"""Shared pieces of the device-style exporters: write-on-change compression and the export CSV shape.

Every export file has the columns ``_time, field, unit, value``, UTC timestamps, one gzipped CSV per
entity. Rows with ``field == "value"`` are numeric, rows with ``field == "state"`` are text.
"""
from __future__ import annotations

import gzip
import io
from pathlib import Path

import numpy as np
import pandas as pd

HEADER = "_time,field,unit,value\n"


def write_on_change(values: np.ndarray, threshold: float) -> np.ndarray:
    """Indices a write-on-change recorder keeps: the first sample, then every sample that differs from the
    last written value by more than ``threshold``."""
    v = np.asarray(values, dtype=float).tolist()
    if not v:
        return np.zeros(0, dtype=np.int64)
    keep = [0]
    last = v[0]
    for i in range(1, len(v)):
        x = v[i]
        d = x - last
        if d > threshold or d < -threshold:
            keep.append(i)
            last = x
    return np.asarray(keep, dtype=np.int64)


def write_on_value_change(values: np.ndarray) -> np.ndarray:
    """Indices where an already quantised series changes value (counters, states, integer percentages)."""
    v = np.asarray(values)
    if len(v) == 0:
        return np.zeros(0, dtype=np.int64)
    ch = np.nonzero(v[1:] != v[:-1])[0] + 1
    return np.r_[np.int64(0), ch.astype(np.int64)]


def utc_strings(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy()


def write_entity_csv(path: Path, times: pd.DatetimeIndex, values, unit: str, decimals: int = 2, field: str = "value") -> int:
    """Write one entity file in the export shape. Returns the number of rows written.

    The gzip header carries a fixed mtime so that two runs with the same seed give the same bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = utc_strings(times)
    if field == "state":
        vals = [str(x) for x in values]
    else:
        vals = [f"{float(x):.{decimals}f}" for x in values]
    with gzip.GzipFile(path, "wb", mtime=0) as gz:
        with io.TextIOWrapper(gz, encoding="utf-8", newline="\n") as f:
            f.write(HEADER)
            f.write("".join(f"{t},{field},{unit},{v}\n" for t, v in zip(ts, vals)))
    return len(ts)


def read_entity_csv(path: Path, field: str = "value") -> pd.Series:
    """Rows of one field from an entity file, UTC index, duplicates dropped (last wins)."""
    df = pd.read_csv(path, compression="gzip", dtype={"_time": str, "field": str, "unit": str, "value": str})
    df = df[df["field"] == field]
    idx = pd.DatetimeIndex(pd.to_datetime(df["_time"].to_numpy(), utc=True, format="%Y-%m-%dT%H:%M:%SZ")).as_unit("ns")
    if field == "state":
        s = pd.Series(df["value"].to_numpy(dtype=object), index=idx)
    else:
        s = pd.Series(pd.to_numeric(df["value"].to_numpy(), errors="coerce"), index=idx, dtype=float)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.name = path.name.removesuffix(".csv.gz")
    return s


class ArtefactLog:
    """Every injected artefact with its timestamp, so the validation can check it was handled."""

    def __init__(self) -> None:
        self.counter_spikes: list[dict] = []
        self.lifetime_glitches: list[dict] = []
        self.silent_gaps: list[dict] = []
        self.sigen_pv_dips: list[dict] = []
        self.sigen_pv_spikes: list[dict] = []

    def as_dict(self) -> dict:
        return {
            "counter_spikes": self.counter_spikes,
            "lifetime_glitches": self.lifetime_glitches,
            "silent_gaps": self.silent_gaps,
            "sigen_pv_dips": self.sigen_pv_dips,
            "sigen_pv_spikes": self.sigen_pv_spikes,
        }


def month_segments(idx: pd.DatetimeIndex, tz: str) -> list[tuple[int, int]]:
    """(start, end) index positions of each calendar month (local) present in a time index."""
    months = idx.tz_convert(tz).strftime("%Y-%m").to_numpy()
    _, starts = np.unique(months, return_index=True)
    starts = np.sort(starts)
    ends = np.r_[starts[1:], len(idx)]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def day_segments(idx: pd.DatetimeIndex, tz: str) -> list[tuple[int, int]]:
    """(start, end) index positions of each local day present in a time index."""
    days = idx.tz_convert(tz).normalize().to_numpy()
    _, starts = np.unique(days, return_index=True)
    starts = np.sort(starts)
    ends = np.r_[starts[1:], len(idx)]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def daily_counter(kwh_per_step: np.ndarray, idx: pd.DatetimeIndex, tz: str, decimals: int = 2) -> np.ndarray:
    """A counter that accumulates within each local day and resets to zero at local midnight.

    A reading stamped ``t`` reports the energy accumulated up to ``t``, not including the step that
    starts at ``t``, the way a device reports what it has already measured. The reading at midnight is
    therefore exactly zero and the last step of a day is carried only by the lifetime counter."""
    cs = np.cumsum(kwh_per_step) - kwh_per_step
    out = np.empty_like(cs)
    for s, e in day_segments(idx, tz):
        out[s:e] = cs[s:e] - cs[s]
    return np.round(out, decimals)


def lifetime_counter(kwh_per_step: np.ndarray, start_kwh: float, decimals: int = 2) -> np.ndarray:
    """A never-resetting counter whose reading at ``t`` is the energy accumulated up to ``t``."""
    return np.round(start_kwh + np.cumsum(kwh_per_step) - kwh_per_step, decimals)
