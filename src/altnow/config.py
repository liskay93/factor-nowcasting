"""설정 로드. 경로는 저장소 루트 기준으로 푼다. macro_factor 는 환경변수 MACRO_FACTOR_ROOT 로 덮어쓸 수 있다."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "config/nowcast.yaml") -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    cfg["_config_path"] = str(p)
    return cfg


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()


def macro_factor_root(cfg: dict) -> Path:
    env = os.environ.get("MACRO_FACTOR_ROOT")
    return Path(env).resolve() if env else resolve(cfg["paths"]["macro_factor_root"])
