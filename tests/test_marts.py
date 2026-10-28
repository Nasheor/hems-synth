"""Marts on a small generated set: Parquet and DuckDB present, row counts, equal sums, tariff rows, API."""
import duckdb
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from hems_synth.__main__ import generate
from hems_synth.marts import TABLES, build_marts, duckdb_sums
from hems_synth.serve import create_app
from hems_synth.tariffs import PLANS


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("marts")
    generate(["S01", "S02"], [("2026-06-01", 2)], 3, True, out)
    info = build_marts(out, out / "marts", ["S01", "S02"])
    return out, info


def test_parquet_and_duckdb_tables_exist(small_run):
    out, info = small_run
    for t in TABLES:
        assert (out / "marts" / f"{t}.parquet").exists()
    con = duckdb.connect(str(out / "marts" / "hems.duckdb"), read_only=True)
    try:
        names = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    finally:
        con.close()
    assert set(TABLES) <= names


def test_energy_30min_rows_per_site_and_source(small_run):
    out, _ = small_run
    e = pd.read_parquet(out / "marts" / "energy_30min.parquet")
    counts = e.groupby(["site_id", "source"]).size()
    for sid in ("S01", "S02"):
        assert counts[(sid, "truth")] == 2 * 48
        assert counts[(sid, "export")] == 2 * 48
    assert e["ts_local"].dtype.kind == "M" and e["ts_local"].dt.tz is None
    assert str(e["ts_utc"].dt.tz) == "UTC"


def test_duckdb_sums_equal_parquet_sums(small_run):
    out, _ = small_run
    e = pd.read_parquet(out / "marts" / "energy_30min.parquet")
    cols = ["import_kwh", "export_kwh", "pv_kwh", "load_kwh"]
    duck = duckdb_sums(out / "marts" / "hems.duckdb", "energy_30min", cols)
    for c in cols:
        assert abs(duck[c] - float(e[c].sum())) < 1e-6


def test_tariff_cost_has_eleven_rows_per_site_and_month(small_run):
    out, _ = small_run
    tc = pd.read_parquet(out / "marts" / "tariff_cost.parquet")
    per = tc.groupby(["site_id", "month"]).size()
    assert (per == len(PLANS)).all() and len(per) == 2
    assert (tc["net_eur"] == tc["unit_eur"] + tc["standing_eur"] + tc["pso_eur"] - tc["export_credit_eur"]).all() or (tc["net_eur"] - (tc["unit_eur"] + tc["standing_eur"] + tc["pso_eur"] - tc["export_credit_eur"])).abs().max() < 1e-6


def test_truth_vs_export_and_sites_tables(small_run):
    out, _ = small_run
    tve = pd.read_parquet(out / "marts" / "truth_vs_export.parquet")
    assert set(tve["flow"]) >= {"import", "export", "pv", "load"}
    sites = pd.read_parquet(out / "marts" / "sites.parquet")
    assert list(sites["site_id"]) == ["S01", "S02"]
    assert "name" not in sites.columns


def test_api_reads_the_marts(small_run):
    out, _ = small_run
    client = TestClient(create_app(out / "marts" / "hems.duckdb"))
    assert client.get("/healthz").json()["status"] == "ok"
    sites = client.get("/sites").json()
    assert [s["site_id"] for s in sites] == ["S01", "S02"]
    usage = client.get("/usage", params={"site_id": "S01", "start": "2026-06-01", "end": "2026-06-02", "source": "truth"}).json()
    assert len(usage) == 48
    daily = client.get("/usage", params={"site_id": "S01", "grain": "daily"}).json()
    assert len(daily) == 2
    cost = client.get("/tariff-cost", params={"site_id": "S02", "month": "2026-06"}).json()
    assert len(cost) == len(PLANS)
