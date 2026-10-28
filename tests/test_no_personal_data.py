"""No real household identifier: every station prefix, MPRN and site id comes from config/sites.yaml,
and no file under the repo matches an Eircode pattern."""
import re
from pathlib import Path

import yaml

from hems_synth.profiles import load_profiles

EIRCODE = re.compile(r"(?<![A-Z0-9-])(?:D6W|[ACDEFHKNPRTVWXY]\d{2})\s?[AC-FHKNPRTV-Y0-9]{4}(?![A-Z0-9])")
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "dist", "build", "data"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".json", ".cfg", ".csv"}


def _text_files(root: Path):
    for p in root.rglob("*"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.is_file() and p.suffix in TEXT_SUFFIXES:
            yield p


def test_no_eircode_pattern_in_repo_files(repo_root):
    hits = []
    for p in _text_files(repo_root):
        text = p.read_text(encoding="utf-8", errors="ignore")
        for m in EIRCODE.finditer(text):
            hits.append(f"{p.relative_to(repo_root)}: {m.group(0)}")
    assert not hits, "Eircode-like strings found:\n" + "\n".join(hits)


def test_identifiers_come_from_the_config(repo_root):
    raw = yaml.safe_load((repo_root / "config" / "sites.yaml").read_text(encoding="utf-8"))
    profiles = load_profiles()
    assert set(profiles) == {s["site_id"] for s in raw["sites"]}
    for s in raw["sites"]:
        p = profiles[s["site_id"]]
        assert p.mprn == s["mprn"] and len(p.mprn) == 11 and p.mprn.startswith("10")
        assert p.meter_serial == s["meter_serial"]
        if p.inverter_family == "sigenergy":
            assert p.station_prefix == s["station_prefix"] and "_" in p.station_prefix


def test_smoke_exports_use_only_config_identifiers(repo_root):
    profiles = load_profiles()
    exports = repo_root / "data" / "smoke" / "exports"
    if not exports.exists():
        return
    for site_dir in exports.iterdir():
        p = profiles[site_dir.name]
        for f in site_dir.glob("*.csv.gz"):
            name = f.name
            if p.inverter_family == "sigenergy":
                assert name.startswith(p.station_prefix) or name.startswith(p.frient_prefix), name
            else:
                assert name.startswith("inverter_"), name
        for f in site_dir.glob("HDF_*.csv"):
            assert f.name.endswith(f"{p.mprn}.csv")
            head = f.read_text(encoding="utf-8").splitlines()[:2]
            assert head[0] == "MPRN,Meter Serial Number,Read Value,Read Type,Read Date and End Time"
            assert head[1].startswith(f"{p.mprn},{p.meter_serial},")
