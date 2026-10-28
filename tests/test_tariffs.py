"""Eleven plans, 48-slot vectors, wrap-around windows, band priority and the bill arithmetic."""
import numpy as np
import pandas as pd

from hems_synth import tariffs


def test_eleven_plans_load_with_sources_and_dates():
    assert len(tariffs.PLANS) == 11
    for key, p in tariffs.PLANS.items():
        assert p["source_url"], key
        assert p["research_date"] == "2026-10-28"
        assert p["standing_eur_year"] > 0
        assert p["day"] > 0


def test_every_rate_vector_has_48_entries():
    for p in tariffs.PLANS.values():
        v = tariffs.rate_vector(p)
        assert len(v) == 48 and all(x > 0 for x in v)


def test_window_wraps_past_midnight():
    assert tariffs.in_window(23.5, (23, 8))
    assert tariffs.in_window(2.0, (23, 8))
    assert not tariffs.in_window(8.0, (23, 8))
    assert tariffs.in_window(17.5, (17, 19))
    assert not tariffs.in_window(19.0, (17, 19))


def test_band_priority_ev_then_peak_then_night_then_day():
    plan = {"day": 40.0, "night": 20.0, "peak": 50.0, "ev": 10.0, "windows": {"night": [[23, 8]], "peak": [[17, 19]], "ev": [[2, 5]]}}
    v = tariffs.rate_vector(plan)
    assert v[0] == 20.0 and v[4] == 10.0 and v[10] == 20.0 and v[16] == 40.0 and v[34] == 50.0 and v[38] == 40.0 and v[46] == 20.0


def test_bill_arithmetic_on_a_flat_day():
    plan = {"day": 40.0, "night": 20.0, "peak": None, "ev": None, "windows": {"night": [[23, 8]], "peak": [], "ev": []}, "standing_eur_year": 365.0, "export_cent": 20.0}
    idx = pd.date_range("2026-06-01", periods=48, freq="30min", tz="Europe/Dublin")
    imp = pd.Series(1.0, index=idx)
    exp = pd.Series(0.5, index=idx)
    b = tariffs.bill(imp, exp, plan, days=1)
    assert b["kwh_night"] == 18.0 and b["kwh_day"] == 30.0 and b["kwh_peak"] == 0.0
    assert abs(b["unit_eur"] - (18 * 0.20 + 30 * 0.40)) < 1e-9
    assert abs(b["standing_eur"] - 1.0) < 1e-9
    assert abs(b["export_credit_eur"] - 24 * 0.20) < 1e-9
    assert abs(b["net_eur"] - (b["unit_eur"] + b["standing_eur"] + b["pso_eur"] - b["export_credit_eur"])) < 1e-9


def test_brute_force_matches_vector_for_every_plan():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2026-10-24", "2026-10-27", freq="5s", tz="Europe/Dublin", inclusive="left")
    power = pd.Series(rng.uniform(0, 3000, len(idx)), index=idx)
    kwh = power * 5 / 3600 / 1000
    hh = kwh.groupby(idx.floor("30min")).sum()
    for p in tariffs.PLANS.values():
        vec = tariffs.bill(hh, hh * 0, p, 3)["unit_eur"]
        brute = tariffs.brute_force_unit_eur(power, p, 5)
        assert abs(vec - brute) < 0.01, p["key"]
