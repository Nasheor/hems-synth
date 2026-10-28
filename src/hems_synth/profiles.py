"""Site profiles: what each synthetic household looks like.

Profiles load from ``config/sites.yaml`` so more sites can be added without code. Every identifier
(site id, station prefix, MPRN, meter serial) comes from that file and is synthetic.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "sites.yaml"
FAMILIES = ("sofar", "sigenergy")


@dataclass(frozen=True)
class SiteProfile:
    """One household. Power limits are kW, energies kWh, loads W."""

    site_id: str
    inverter_family: str
    pv_kwp: float
    battery_kwh: float
    battery_power_kw: float
    ev_charger_kw: float | None
    occupants: int
    base_load_w: float
    tariff_key: str
    latitude: float
    seed: int
    longitude: float = -8.5
    inverter_ac_kw: float | None = None
    winter_load_factor: float = 0.25
    ev_nights_per_week: float = 3.5
    ev_daily_kwh_min: float = 8.0
    ev_daily_kwh_max: float = 30.0
    station_prefix: str | None = None
    inverter_id: str | None = None
    frient_prefix: str | None = None
    mprn: str = "10000000000"
    meter_serial: str = "SYN0000000"
    lifetime_import_kwh: float = 0.0
    lifetime_export_kwh: float = 0.0

    def __post_init__(self) -> None:
        if self.inverter_family not in FAMILIES:
            raise ValueError(f"inverter_family must be one of {FAMILIES}, got {self.inverter_family!r}")
        if self.inverter_family == "sigenergy" and not (self.station_prefix and self.inverter_id):
            raise ValueError(f"{self.site_id}: a sigenergy site needs station_prefix and inverter_id")
        if len(self.mprn) != 11 or not self.mprn.isdigit():
            raise ValueError(f"{self.site_id}: mprn must be eleven digits")

    @property
    def ac_limit_w(self) -> float:
        kw = self.inverter_ac_kw if self.inverter_ac_kw is not None else self.pv_kwp
        return float(kw) * 1000.0

    def as_dict(self) -> dict:
        return asdict(self)


def load_profiles(path: Path | str = DEFAULT_CONFIG) -> dict[str, SiteProfile]:
    """All profiles in the YAML file, keyed by site id."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out: dict[str, SiteProfile] = {}
    for item in raw["sites"]:
        p = SiteProfile(**item)
        if p.site_id in out:
            raise ValueError(f"duplicate site id {p.site_id}")
        out[p.site_id] = p
    return out


def get_profile(site_id: str, path: Path | str = DEFAULT_CONFIG) -> SiteProfile:
    profiles = load_profiles(path)
    if site_id not in profiles:
        raise KeyError(f"unknown site {site_id!r}, known: {sorted(profiles)}")
    return profiles[site_id]
