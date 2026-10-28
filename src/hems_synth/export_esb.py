"""ESB Networks HDF style file: the half-hourly smart-meter download a household can fetch itself.

Columns ``MPRN, Meter Serial Number, Read Value, Read Type, Read Date and End Time``. One row per
register per half-hour, registers ``Active Import Interval (kW)`` and ``Active Export Interval (kW)``
in the kW file (average power over the half-hour) and ``Active Import Interval (kWh)`` and
``Active Export Interval (kWh)`` in the calculated kWh file. ``Read Date and End Time`` is the end
of the interval in Irish clock time, ``dd-mm-yyyy HH:MM``. The DST-end hour appears twice without a
marker and the DST-start hour is absent, because the clock itself skips and repeats. Rows are
written newest first, as a portal download is.

The MPRN and meter serial come from the site profile and are synthetic.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import TZ
from .simulate import SimResult

COLUMNS = ["MPRN", "Meter Serial Number", "Read Value", "Read Type", "Read Date and End Time"]
READ_TYPES = {
    "kW": ("Active Import Interval (kW)", "Active Export Interval (kW)"),
    "kWh": ("Active Import Interval (kWh)", "Active Export Interval (kWh)"),
}


def hdf_frame(half_hour: pd.DataFrame, mprn: str, serial: str, unit: str = "kW") -> pd.DataFrame:
    """Rows of the HDF file from a truth half-hour frame indexed by UTC interval start."""
    if unit not in READ_TYPES:
        raise ValueError("unit must be kW or kWh")
    imp_type, exp_type = READ_TYPES[unit]
    end_local = (half_hour.index + pd.Timedelta(minutes=30)).tz_convert(TZ).strftime("%d-%m-%Y %H:%M")
    scale = 2.0 if unit == "kW" else 1.0
    imp = pd.DataFrame(
        {
            "MPRN": mprn,
            "Meter Serial Number": serial,
            "Read Value": (half_hour["import_kwh"].to_numpy() * scale).round(3),
            "Read Type": imp_type,
            "Read Date and End Time": end_local,
        }
    )
    exp = imp.copy()
    exp["Read Value"] = (half_hour["export_kwh"].to_numpy() * scale).round(3)
    exp["Read Type"] = exp_type
    order = pd.Series(range(len(half_hour)))
    rows = pd.concat([imp.assign(_o=order), exp.assign(_o=order)]).sort_values(["_o", "Read Type"], ascending=[False, True])
    return rows.drop(columns="_o").reset_index(drop=True)[COLUMNS]


def export_esb(sim: SimResult, out_folder: Path) -> dict:
    """Write the kW file and the calculated kWh file for one site."""
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    p = sim.profile
    written = {}
    for unit, name in (("kW", "HDF_kW"), ("kWh", "HDF_calckWh")):
        frame = hdf_frame(sim.half_hour, p.mprn, p.meter_serial, unit)
        path = out_folder / f"{name}_{p.mprn}.csv"
        frame.to_csv(path, index=False, lineterminator="\n", float_format="%.3f")
        written[unit] = {"path": str(path), "rows": int(len(frame))}
    return written
