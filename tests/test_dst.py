"""Europe/Dublin DST handling: 46 half-hours on 29 March 2026, 50 on 25 October 2026, in the truth, the
loader output, the mart and the HDF file."""
import json

import pandas as pd
import pytest

from hems_synth.export_esb import export_esb
from hems_synth.export_ha import export_sofar
from hems_synth.loader import load_site, read_esb_hdf
from hems_synth.profiles import get_profile
from hems_synth.simulate import local_grid, simulate


def test_local_grid_has_23_and_25_hour_days():
    g = local_grid([("2026-03-29", 1)])
    assert len(g) == 23 * 720
    g = local_grid([("2026-10-25", 1)])
    assert len(g) == 25 * 720


@pytest.fixture(scope="module")
def dst_site(tmp_path_factory):
    out = tmp_path_factory.mktemp("dst")
    sim = simulate(get_profile("S01"), [("2026-03-28", 2), ("2026-10-24", 2)], seed=7)
    export_sofar(sim, out / "S01", 7, artefacts=False)
    export_esb(sim, out / "S01")
    return sim, out / "S01"


def test_truth_half_hours_on_dst_days(dst_site):
    sim, _ = dst_site
    counts = sim.half_hour.groupby(sim.half_hour["ts_local"].dt.date).size()
    assert counts[pd.Timestamp("2026-03-29").date()] == 46
    assert counts[pd.Timestamp("2026-10-25").date()] == 50


def test_loader_half_hours_on_dst_days(dst_site):
    _, folder = dst_site
    site = load_site(folder)
    e30 = site["energy30"]
    counts = e30.groupby(e30.index.date).size()
    assert counts[pd.Timestamp("2026-03-29").date()] == 46
    assert counts[pd.Timestamp("2026-10-25").date()] == 50
    assert str(e30.index.tz) == "Europe/Dublin"


def test_esb_file_repeats_the_dst_end_hour_and_local_read_drops_it(dst_site):
    sim, folder = dst_site
    path = folder / f"HDF_calckWh_{sim.profile.mprn}.csv"
    raw = pd.read_csv(path, dtype=str)
    ends = raw["Read Date and End Time"]
    imp = raw["Read Type"] == "Active Import Interval (kWh)"
    assert int((imp & ends.str.endswith("25-10-2026 01:30")).sum()) == 2
    assert int((imp & ends.str.endswith("25-10-2026 01:00")).sum()) == 2
    assert int((imp & ends.str.startswith("29-03-2026 01:00")).sum()) == 0
    assert int((imp & ends.str.startswith("29-03-2026 01:30")).sum()) == 0
    assert int((imp & ends.str.startswith("29-03-2026 02:00")).sum()) == 1
    local = read_esb_hdf(path, clock="local")
    day = local[local.index.strftime("%Y-%m-%d") == "2026-10-25"]
    assert len(day) == 46
    assert day.index.is_unique
    starts_utc = day.index.tz_convert("UTC")
    dropped = (starts_utc >= pd.Timestamp("2026-10-24T23:30:00Z")) & (starts_utc < pd.Timestamp("2026-10-25T01:30:00Z"))
    assert not dropped.any()
    utc = read_esb_hdf(path, clock="utc")
    # read as UTC, the repeated wall-clock labels collapse to one row each, so two of the four days' 192 rows are lost
    assert len(utc) == 192 - 2 and len(local) == 192 - 4
    assert local.attrs["unit"] == "kWh"


def test_smoke_results_report_dst_counts(repo_root):
    val = json.loads((repo_root / "results" / "validation.json").read_text(encoding="utf-8"))
    for sid in ("S01", "S02"):
        assert val["sites"][sid]["half_hours_2026_03_29"] == 46
        assert val["sites"][sid]["half_hours_2026_10_25"] == 50
