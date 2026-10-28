"""Command line entry point.

``generate`` simulates sites and writes the truth and the device-style exports, ``marts`` builds the
Parquet and DuckDB marts, ``validate`` compares the loader with the truth, ``serve`` starts the
read-only API and ``smoke`` runs the whole chain on a fixed twenty-day set (two five-day windows around
the 2026 DST changes plus ten ordinary days), writes every ``results/*.json`` and regenerates the README
results table. The smoke run finishes in under ten minutes and is deterministic (seed 42).
"""
from __future__ import annotations

import argparse
import datetime as dt
import filecmp
import json
import shutil
import sys
import time
from pathlib import Path

from .export_esb import export_esb
from .export_ha import export_sofar
from .export_sigen import export_sigenergy
from .marts import build_marts
from .profiles import load_profiles
from .simulate import simulate, write_truth
from .validate import validate, write_json

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
SMOKE_WINDOWS = [("2026-03-27", 5), ("2026-06-01", 10), ("2026-10-23", 5)]
SMOKE_SITES = ["S01", "S02"]
README_START = "<!-- results:start -->"
README_END = "<!-- results:end -->"


def _parse_windows(text: str) -> list[tuple[str, int]]:
    out = []
    for part in text.split(","):
        start, _, days = part.strip().partition(":")
        out.append((start, int(days or 1)))
    return out


def generate(sites: list[str], windows: list[tuple[str, int]], seed: int, artefacts: bool, out: Path, config: Path | None = None) -> dict:
    """Simulate each site, write truth and exports under ``out``. Returns per-site statistics."""
    profiles = load_profiles(config) if config else load_profiles()
    out = Path(out)
    stats: dict = {"seed": seed, "windows": [[s, d] for s, d in windows], "artefacts": artefacts, "sites": {}}
    for sid in sites:
        prof = profiles[sid]
        t0 = time.perf_counter()
        sim = simulate(prof, windows, seed=seed)
        t1 = time.perf_counter()
        truth_dir = out / "truth" / sid
        export_dir = out / "exports" / sid
        if export_dir.exists():
            shutil.rmtree(export_dir)
        meta = write_truth(sim, truth_dir)
        if prof.inverter_family == "sigenergy":
            inv = export_sigenergy(sim, export_dir, seed, artefacts=artefacts)
        else:
            inv = export_sofar(sim, export_dir, seed, artefacts=artefacts)
        esb = export_esb(sim, export_dir)
        t2 = time.perf_counter()
        (truth_dir / "artefacts.json").write_text(json.dumps(inv["artefacts"], indent=2), encoding="utf-8")
        if "prefixes" in inv:
            meta["export_prefixes"] = inv["prefixes"]
            (truth_dir / "profile.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        stats["sites"][sid] = {
            "family": prof.inverter_family,
            "days": int(len(sim.daily)),
            "steps": int(len(sim.truth)),
            "entities": inv["entities"],
            "points": inv["points"],
            "esb_rows": {k: v["rows"] for k, v in esb.items()},
            "artefacts": {k: len(v) for k, v in inv["artefacts"].items()},
            "simulate_s": round(t1 - t0, 2),
            "export_s": round(t2 - t1, 2),
        }
        print(f"{sid}: {len(sim.daily)} days, {inv['entities']} entities, {inv['points']} points, {t2 - t0:.1f} s")
    return stats


def _determinism(sites: list[str], windows: list[tuple[str, int]], seed: int, scratch: Path, config: Path | None = None) -> dict:
    """Two runs with the same seed must give byte-identical truth Parquet files."""
    profiles = load_profiles(config) if config else load_profiles()
    out = {}
    for sid in sites:
        dirs = []
        for k in (1, 2):
            d = scratch / f"run{k}" / sid
            if d.exists():
                shutil.rmtree(d)
            write_truth(simulate(profiles[sid], windows, seed=seed), d)
            dirs.append(d)
        same = all(filecmp.cmp(dirs[0] / f, dirs[1] / f, shallow=False) for f in ("truth_5s.parquet", "truth_30min.parquet", "truth_daily.parquet"))
        out[sid] = "yes" if same else "no"
    return out


def _readme_table(gen: dict, val: dict, load_s: dict[str, float]) -> tuple[str, dict]:
    header = "| Site | Inverter style | Days | Entities exported | Points written | Daily kWh max error vs truth (%) | Half-hours on 29 Mar / 25 Oct 2026 | Spikes injected / removed | Gaps injected / found | Plans costed | Cheapest plan (net €/month) | Generate + load time (s) |"
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|"
    rows = []
    numbers: dict[str, str] = {}
    for sid, g in gen["sites"].items():
        v = val["sites"][sid]
        if g["family"] == "sofar":
            spikes = f"{v['counters']['spikes_injected']} / {v['counters']['spikes_removed']}"
            gaps = f"{v['gaps']['injected']} / {v['gaps']['found']}"
            style = "Sofar, Home Assistant"
        else:
            spikes = f"{v['sigen']['dips_injected'] + v['sigen']['spikes_injected']} / {v['sigen']['artefacts_removed']}"
            gaps = "none injected"
            style = "Sigenergy"
        err = v["daily_max_pct_ordinary_days"]
        err_s = f"{err:.2f}" if err is not None else "n/a"
        hh = f"{v['half_hours_2026_03_29']} / {v['half_hours_2026_10_25']}"
        cheapest = val["tariffs"]["cheapest_per_site"][sid]
        cheap_s = f"{cheapest['plan_key']} (€{cheapest['net_eur_per_30_days']:.2f})"
        secs = int(round(g["simulate_s"] + g["export_s"] + load_s.get(sid, 0.0)))
        rows.append(f"| {sid} | {style} | {g['days']} | {g['entities']} | {g['points']:,} | {err_s} | {hh} | {spikes} | {gaps} | {val['tariffs']['plans_costed']} | {cheap_s} | {secs} |")
        numbers[f"{sid}_days"] = str(g["days"])
        numbers[f"{sid}_entities"] = str(g["entities"])
        numbers[f"{sid}_points"] = f"{g['points']:,}"
        numbers[f"{sid}_daily_max_pct"] = err_s
        numbers[f"{sid}_half_hours"] = hh
        numbers[f"{sid}_spikes"] = spikes
        numbers[f"{sid}_gaps"] = gaps
        numbers[f"{sid}_cheapest"] = cheap_s
        numbers[f"{sid}_seconds"] = str(secs)
    numbers["plans_costed"] = str(val["tariffs"]["plans_costed"])
    return "\n".join([header, sep, *rows]), numbers


def _update_readme(table: str, extra_lines: list[str]) -> None:
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        return
    before, _, rest = text.partition(README_START)
    _, _, after = rest.partition(README_END)
    block = "\n".join([README_START, table, "", *extra_lines, README_END])
    readme.write_text(before + block + after, encoding="utf-8", newline="\n")


def smoke(out: Path | None = None, seed: int = 42) -> int:
    out = Path(out) if out else ROOT / "data" / "smoke"
    RESULTS.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    gen = generate(SMOKE_SITES, SMOKE_WINDOWS, seed, True, out)
    t1 = time.perf_counter()
    marts = build_marts(out, out / "marts", SMOKE_SITES)
    t2 = time.perf_counter()
    val = validate(out, out / "marts", loaded=marts["loaded"])
    t3 = time.perf_counter()
    det = _determinism(SMOKE_SITES, [("2026-06-01", 2)], seed, out / "_determinism")
    t4 = time.perf_counter()
    load_s = {sid: (t2 - t1) / len(SMOKE_SITES) for sid in SMOKE_SITES}
    gen["timing_s"] = {"generate": round(t1 - t0, 2), "marts_and_load": round(t2 - t1, 2), "validate": round(t3 - t2, 2), "determinism": round(t4 - t3, 2), "total": round(t4 - t0, 2)}
    gen["marts_rows"] = marts["rows"]
    val["determinism_truth_parquet_identical"] = det
    val["smoke_seconds"] = round(t4 - t0, 2)
    write_json(RESULTS / "generate.json", gen)
    write_json(RESULTS / "validation.json", val)
    write_json(RESULTS / "tariffs.json", val["tariffs"])
    table, numbers = _readme_table(gen, val, load_s)
    numbers["smoke_seconds"] = str(int(round(t4 - t0)))
    numbers["determinism"] = ", ".join(f"{k} {v}" for k, v in det.items())
    extra = [
        f"Smoke run: {int(round(t4 - t0))} s in total on this PC, seed {seed}. Determinism of the truth Parquet files over two runs: {numbers['determinism']}.",
        "Net €/month is the cheapest plan's net cost over the smoke days scaled to thirty days, standing charge and PSO levy included, export credit deducted.",
    ]
    _update_readme(table, extra)
    write_json(RESULTS / "readme_numbers.json", numbers)
    write_json(RESULTS / "smoke.json", {"smoke_ok": "yes", "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "seconds": round(t4 - t0, 2)})
    print(table)
    print(f"smoke ok in {t4 - t0:.1f} s")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hems_synth")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="simulate sites and write truth plus device-style exports")
    g.add_argument("--sites", default="S01,S02,S03")
    g.add_argument("--start", default="2026-01-01")
    g.add_argument("--days", type=int, default=365)
    g.add_argument("--windows", default=None, help="comma list of start:days, overrides --start and --days")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--artefacts", action="store_true")
    g.add_argument("--out", default="data")
    g.add_argument("--config", default=None)

    m = sub.add_parser("marts", help="build the Parquet and DuckDB marts")
    m.add_argument("--in", dest="inp", default="data")
    m.add_argument("--out", default="data/marts")
    m.add_argument("--sites", default=None)

    v = sub.add_parser("validate", help="compare the loader with the truth and write results/")
    v.add_argument("--in", dest="inp", default="data")
    v.add_argument("--marts", default="data/marts")

    s = sub.add_parser("serve", help="read-only API over the DuckDB marts")
    s.add_argument("--db", default=str(ROOT / "data" / "marts" / "hems.duckdb"))
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    sm = sub.add_parser("smoke", help="short reproducible run that regenerates results/")
    sm.add_argument("--out", default=None)

    args = p.parse_args(argv)
    if args.cmd == "generate":
        windows = _parse_windows(args.windows) if args.windows else [(args.start, args.days)]
        stats = generate(args.sites.split(","), windows, args.seed, args.artefacts, Path(args.out), Path(args.config) if args.config else None)
        write_json(RESULTS / "generate.json", stats)
        return 0
    if args.cmd == "marts":
        info = build_marts(Path(args.inp), Path(args.out), args.sites.split(",") if args.sites else None)
        print(json.dumps(info["rows"], indent=2))
        return 0
    if args.cmd == "validate":
        val = validate(Path(args.inp), Path(args.marts))
        write_json(RESULTS / "validation.json", val)
        write_json(RESULTS / "tariffs.json", val["tariffs"])
        print(json.dumps({k: v.get("daily_max_pct_ordinary_days") for k, v in val["sites"].items()}, indent=2))
        return 0
    if args.cmd == "serve":
        from .serve import run

        run(Path(args.db), args.host, args.port)
        return 0
    if args.cmd == "smoke":
        return smoke(Path(args.out) if args.out else None)
    return 2


if __name__ == "__main__":
    sys.exit(main())
