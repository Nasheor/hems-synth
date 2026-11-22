"""The wheel must import on its own, outside the checkout.

The site profiles and the tariff plans are read at import time. The wheel ships packaged copies of both
files, and these tests check that the copies match the repo files and that a freshly built wheel imports
in a clean virtual environment.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
PKG_DATA = ROOT / "src" / "hems_synth" / "_data"


@pytest.mark.parametrize(
    ("repo_file", "packaged"),
    [("config/sites.yaml", "sites.yaml"), ("data/tariffs/plans.json", "plans.json")],
)
def test_packaged_copy_matches_repo(repo_file: str, packaged: str) -> None:
    assert (ROOT / repo_file).read_bytes() == (PKG_DATA / packaged).read_bytes(), (
        f"src/hems_synth/_data/{packaged} differs from {repo_file}, copy the repo file over it"
    )


def test_wheel_imports_outside_checkout(tmp_path: Path) -> None:
    pytest.importorskip("build")
    dist = tmp_path / "dist"
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist), str(ROOT)],
        check=True,
        capture_output=True,
    )
    wheel = next(dist.glob("hems_synth-*.whl"))
    target = tmp_path / "site"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(target), "-q", str(wheel)],
        check=True,
    )
    # Run the same interpreter (its dependencies are installed) with the unpacked wheel first on sys.path,
    # from a directory outside the checkout, and check the module really came from the wheel.
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import hems_synth.tariffs as t, hems_synth.profiles as p; "
        "print(t.__file__.startswith(sys.argv[1]), len(t.load_plans()['plans']), "
        "len(p.load_profiles()), t.PLANS_PATH.parent.name, p.DEFAULT_CONFIG.parent.name)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, str(target)], check=True, capture_output=True, text=True, cwd=tmp_path
    ).stdout.split()
    n_sites = len(yaml.safe_load((ROOT / "config" / "sites.yaml").read_text(encoding="utf-8"))["sites"])
    assert out == ["True", "11", str(n_sites), "_data", "_data"], out
