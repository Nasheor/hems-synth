"""Irish residential smart-meter plans and the half-hourly bill arithmetic.

``PLANS`` loads from ``data/tariffs/plans.json``: eleven plans re-researched from the suppliers' public
price pages on the date recorded in the file. Unit rates are cent/kWh including 9 % VAT, standing
charges EUR/year including VAT for an urban smart meter. Hours are local clock hours 0 to 24, a window
with start greater than end wraps past midnight, and the band priority is EV, then peak, then night,
then day. ``rate_vector`` gives cent/kWh for each of the 48 half-hours of a day.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import TZ

ROOT = Path(__file__).resolve().parents[2]
PLANS_PATH = ROOT / "data" / "tariffs" / "plans.json"
BANDS = ("ev", "peak", "night", "day")


def load_plans(path: Path | str = PLANS_PATH) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for plan in data["plans"]:
        plan.setdefault("windows", {})
        for band in ("night", "peak", "ev"):
            plan["windows"].setdefault(band, [])
    return data


_DATA = load_plans()
PLANS: dict[str, dict] = {p["key"]: p for p in _DATA["plans"]}
PSO_EUR_MONTH = float(_DATA["pso_levy"]["eur_month_inc_vat"])
DEFAULT_EXPORT_CENT = float(_DATA["default_export_cent"])
VAT_PCT = float(_DATA["vat"]["pct"])


def in_window(t: float, win) -> bool:
    """True when local clock hour ``t`` (0 to 24) falls in ``win`` = (start, end). start > end wraps midnight."""
    start, end = float(win[0]), float(win[1])
    if start == end:
        return False
    if start < end:
        return start <= t < end
    return t >= start or t < end


def band_at(plan: dict, t: float) -> str:
    """Band for local clock hour ``t`` by priority EV, peak, night, day."""
    for band in ("ev", "peak", "night"):
        if plan.get(band) is None:
            continue
        if any(in_window(t, w) for w in plan["windows"].get(band, [])):
            return band
    return "day"


def rate_vector(plan: dict) -> list[float]:
    """Cent/kWh for each of the 48 half-hours of a day."""
    return [float(plan[band_at(plan, h / 2.0)]) for h in range(48)]


def band_vector(plan: dict) -> list[str]:
    return [band_at(plan, h / 2.0) for h in range(48)]


def _slot(index: pd.DatetimeIndex) -> np.ndarray:
    loc = index.tz_convert(TZ) if index.tz is not None else index.tz_localize(TZ)
    return (loc.hour.to_numpy() * 2 + loc.minute.to_numpy() // 30).astype(int)


def export_rate_cent(plan: dict) -> float:
    rate = plan.get("export_cent")
    return float(rate) if rate is not None else DEFAULT_EXPORT_CENT


def bill(imp30: pd.Series, exp30: pd.Series, plan: dict, days: int) -> dict:
    """Cost of a half-hourly import and export pair on ``plan`` over ``days`` days.

    Returns kWh per band, the unit cost, the standing charge and PSO levy for the period, the export
    credit and ``net_eur`` = unit + standing + pso - export credit. Half-hours are assigned to bands by
    their local start time."""
    imp = imp30.fillna(0.0)
    exp = exp30.reindex(imp.index).fillna(0.0)
    slots = _slot(imp.index)
    rates = np.asarray(rate_vector(plan))
    bands = np.asarray(band_vector(plan))
    kwh = imp.to_numpy(dtype=float)
    out: dict[str, float] = {}
    for band in BANDS:
        out[f"kwh_{band}"] = float(kwh[bands[slots] == band].sum())
    unit_eur = float((kwh * rates[slots]).sum() / 100.0)
    standing_eur = float(plan["standing_eur_year"]) * days / 365.0
    pso_eur = PSO_EUR_MONTH * 12.0 * days / 365.0
    export_credit_eur = float(exp.sum()) * export_rate_cent(plan) / 100.0
    unit_eur, standing_eur, pso_eur, export_credit_eur = (round(x, 4) for x in (unit_eur, standing_eur, pso_eur, export_credit_eur))
    out.update(
        {
            "kwh_import": float(kwh.sum()),
            "kwh_export": float(exp.sum()),
            "unit_eur": unit_eur,
            "standing_eur": standing_eur,
            "pso_eur": pso_eur,
            "export_credit_eur": export_credit_eur,
            "net_eur": round(unit_eur + standing_eur + pso_eur - export_credit_eur, 4),
            "days": int(days),
        }
    )
    return out


def brute_force_unit_eur(import_w: pd.Series, plan: dict, step_s: float) -> float:
    """Unit cost from a fine-grained power series by assigning every sample to its band directly.
    Used only to check that the 48-slot vector gives the same answer."""
    loc = import_w.index.tz_convert(TZ)
    t = loc.hour.to_numpy() + loc.minute.to_numpy() / 60.0 + loc.second.to_numpy() / 3600.0
    rates = np.empty(len(t))
    cache: dict[float, float] = {}
    for i, h in enumerate(t):
        r = cache.get(h)
        if r is None:
            r = float(plan[band_at(plan, h)])
            cache[h] = r
        rates[i] = r
    kwh = import_w.to_numpy(dtype=float) * step_s / 3600.0 / 1000.0
    return float((kwh * rates).sum() / 100.0)


def plans_table() -> pd.DataFrame:
    rows = []
    for p in _DATA["plans"]:
        rows.append(
            {
                "plan_key": p["key"],
                "supplier": p["supplier"],
                "name": p["name"],
                "day_cent": p["day"],
                "night_cent": p.get("night"),
                "peak_cent": p.get("peak"),
                "ev_cent": p.get("ev"),
                "night_windows": json.dumps(p["windows"].get("night", [])),
                "peak_windows": json.dumps(p["windows"].get("peak", [])),
                "ev_windows": json.dumps(p["windows"].get("ev", [])),
                "standing_eur_year": p["standing_eur_year"],
                "export_cent": p.get("export_cent"),
                "export_cent_used": export_rate_cent(p),
                "discount_included_pct": p.get("discount_included_pct", 0),
                "vat_included": bool(p.get("vat_included", True)),
                "effective_from": p.get("effective_from"),
                "research_date": p.get("research_date"),
                "source_url": " | ".join(p.get("source_url", [])),
                "notes": p.get("notes", ""),
            }
        )
    return pd.DataFrame(rows)
