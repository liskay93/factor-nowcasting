"""data/dashboard/*.csv → DB. 접속 문자열은 환경변수 ALTNOW_DB_URL (없으면 sqlite:///data/dashboard/altnow.db).

    beta_timeseries: 키 (date, program, factor, basis, method, window_q) 기준 upsert (같은 키는 덮어씀)
    beta_latest, factor_latest: 통째로 교체
예) ALTNOW_DB_URL=postgresql+psycopg2://user:pw@host/db python scripts/load_db.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import sqlalchemy as sa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.altnow.config import load_config, resolve  # noqa: E402

TS_KEYS = ["date", "program", "factor", "basis", "method", "window_q"]


def upsert_timeseries(engine: sa.Engine, df: pd.DataFrame) -> int:
    """휴대성을 위해 '같은 키 삭제 후 삽입' 으로 upsert 한다 (DB 방언별 ON CONFLICT 를 피함)."""
    with engine.begin() as con:
        insp = sa.inspect(engine)
        if "beta_timeseries" in insp.get_table_names():
            keys = df[TS_KEYS].drop_duplicates()
            # 방언 무관하게: 이번 적재의 (basis, method, window_q) 조합 × 날짜 범위를 지우고 다시 넣는다
            for (b, m, w), g in keys.groupby(["basis", "method", "window_q"]):
                con.execute(sa.text("DELETE FROM beta_timeseries WHERE basis=:b AND method=:m AND window_q=:w AND date BETWEEN :d0 AND :d1"),
                            {"b": b, "m": m, "w": int(w), "d0": str(g.date.min()), "d1": str(g.date.max())})
        df.to_sql("beta_timeseries", con, if_exists="append", index=False)
    return len(df)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/nowcast.yaml")
    ap.add_argument("--src", default="data/dashboard")
    ap.add_argument("--db-url", default=None)
    ap.add_argument("--schema", action="store_true", help="db/schema.sql 을 먼저 실행 (PK 포함 테이블 생성)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    env = (cfg.get("dashboard") or {}).get("db_url_env", "ALTNOW_DB_URL")
    url = a.db_url or os.environ.get(env) or f"sqlite:///{resolve('data/dashboard/altnow.db')}"
    src = resolve(a.src)
    engine = sa.create_engine(url)
    if a.schema:
        ddl = (ROOT / "db/schema.sql").read_text(encoding="utf-8")
        with engine.begin() as con:
            for stmt in [s.strip() for s in ddl.split(";") if s.strip()]:
                con.execute(sa.text(stmt))
    ts = pd.read_csv(src / "beta_timeseries.csv", parse_dates=["date"])
    ts["date"] = ts["date"].dt.date
    n = upsert_timeseries(engine, ts)
    for name, dates in [("beta_latest", ["asof_quarter"]), ("factor_latest", ["last_full_quarter", "qtd_asof", "qtd_quarter"])]:
        df = pd.read_csv(src / f"{name}.csv", parse_dates=dates)
        for c in dates:
            df[c] = df[c].dt.date
        with engine.begin() as con:
            con.execute(sa.text(f"DELETE FROM {name}")) if name in sa.inspect(engine).get_table_names() else None
            df.to_sql(name, con, if_exists="append", index=False)
    with engine.connect() as con:
        cnt = {t: con.execute(sa.text(f"SELECT COUNT(*) FROM {t}")).scalar() for t in ["beta_timeseries", "beta_latest", "factor_latest"]}
    print(f"db: {url}\n  loaded beta_timeseries {n} rows → table counts {cnt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
