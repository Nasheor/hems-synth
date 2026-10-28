"""Read-only FastAPI over the DuckDB marts (optional, for the later Kubernetes unit).

``GET /healthz``, ``GET /sites``, ``GET /usage?site_id=S01&start=2026-06-01&end=2026-06-03&source=truth``
and ``GET /tariff-cost?site_id=S01&month=2026-06``.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
from fastapi import FastAPI, HTTPException, Query

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "data" / "marts" / "hems.duckdb"


def create_app(db_path: Path | str = DEFAULT_DB) -> FastAPI:
    db_path = Path(db_path)
    app = FastAPI(title="hems-synth marts", version="0.1.0")

    def query(sql: str, params: list) -> list[dict]:
        if not db_path.exists():
            raise HTTPException(status_code=503, detail=f"marts not built: {db_path}")
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            con.execute("SET TimeZone = 'UTC'")
            cur = con.execute(sql, params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, (v.isoformat() if hasattr(v, "isoformat") else v for v in row))) for row in cur.fetchall()]
        finally:
            con.close()

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "db": str(db_path), "exists": db_path.exists()}

    @app.get("/sites")
    def sites() -> list[dict]:
        return query("SELECT * FROM sites ORDER BY site_id", [])

    @app.get("/usage")
    def usage(
        site_id: str = Query(...),
        start: str | None = Query(None, description="local date or timestamp, inclusive"),
        end: str | None = Query(None, description="local date or timestamp, exclusive"),
        source: str = Query("truth"),
        grain: str = Query("30min", pattern="^(30min|daily)$"),
    ) -> list[dict]:
        if grain == "daily":
            sql = "SELECT * FROM daily WHERE site_id = ? AND source = ?"
            params: list = [site_id, source]
            if start:
                sql += " AND date_local >= ?"
                params.append(start)
            if end:
                sql += " AND date_local < ?"
                params.append(end)
            return query(sql + " ORDER BY date_local", params)
        sql = "SELECT * FROM energy_30min WHERE site_id = ? AND source = ?"
        params = [site_id, source]
        if start:
            sql += " AND ts_local >= ?"
            params.append(start)
        if end:
            sql += " AND ts_local < ?"
            params.append(end)
        return query(sql + " ORDER BY ts_utc", params)

    @app.get("/tariff-cost")
    def tariff_cost(site_id: str = Query(...), month: str | None = Query(None)) -> list[dict]:
        sql = "SELECT * FROM tariff_cost WHERE site_id = ?"
        params: list = [site_id]
        if month:
            sql += " AND month = ?"
            params.append(month)
        return query(sql + " ORDER BY month, net_eur", params)

    return app


def run(db_path: Path | str = DEFAULT_DB, host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(create_app(db_path), host=host, port=port)
