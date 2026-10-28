"""Two runs with the same seed give byte-identical truth Parquet files and identical exports."""
import filecmp
import json

from hems_synth.export_ha import export_sofar
from hems_synth.profiles import get_profile
from hems_synth.simulate import simulate, write_truth


def test_same_seed_same_bytes(tmp_path):
    prof = get_profile("S01")
    dirs = []
    for k in (1, 2):
        sim = simulate(prof, [("2026-06-01", 1)], seed=42)
        d = tmp_path / f"run{k}"
        write_truth(sim, d / "truth")
        export_sofar(sim, d / "exports", 42, artefacts=True)
        dirs.append(d)
    for name in ("truth_5s.parquet", "truth_30min.parquet", "truth_daily.parquet", "ev_sessions.json"):
        assert filecmp.cmp(dirs[0] / "truth" / name, dirs[1] / "truth" / name, shallow=False), name
    for f in sorted((dirs[0] / "exports").glob("*.csv.gz")):
        assert filecmp.cmp(f, dirs[1] / "exports" / f.name, shallow=False), f.name


def test_different_seed_different_truth(tmp_path):
    prof = get_profile("S01")
    a = simulate(prof, [("2026-06-01", 1)], seed=1).truth
    b = simulate(prof, [("2026-06-01", 1)], seed=2).truth
    assert not a["load_w"].equals(b["load_w"])


def test_smoke_reports_determinism(repo_root):
    val = json.loads((repo_root / "results" / "validation.json").read_text(encoding="utf-8"))
    assert all(v == "yes" for v in val["determinism_truth_parquet_identical"].values())
