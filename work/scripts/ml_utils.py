from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parent
WORK_DIR = SCRIPTS_DIR.parent
REPO_ROOT = WORK_DIR.parent

RAW_PATH = REPO_ROOT / "data" / "raw" / "content_refresh_anonymized.csv"
OUTPUT_DIR = WORK_DIR / "outputs"

SEED = 42

WAREHOUSE = "hf://datasets/FlyRank/internship-warehouse"
ANCHOR = "2026-03-31"


def hf_token() -> str:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    env = REPO_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("HF_TOKEN") and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    try:
        from google.colab import userdata
        return userdata.get("HF_TOKEN")
    except Exception:
        from getpass import getpass
        return getpass("HF_TOKEN: ")


def warehouse_connection():
    import duckdb
    con = duckdb.connect()
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    con.execute(f"CREATE OR REPLACE SECRET (TYPE huggingface, TOKEN '{hf_token()}')")
    return con


NUM_FEATURES = [
    "log_impressions_prev_30d",
    "log_clicks_prev_30d",
    "log_sessions_prev_30d",
    "log_impr_older_30d",
    "ctr_prev_30d",
    "hist_impr_momentum",
    "log_search_volume",
    "competition",
    "cpc",
    "word_count",
    "char_count",
    "content_age_days",
    "days_since_last_update",
    "age_tier_order",
    "has_keyword_data",
    "has_word_count",
]

CAT_FEATURES = ["content_type", "main_intent", "competition_level"]


FORBIDDEN_FEATURES = {
    "trend_pct",
    "trend_direction",
    "is_declining_label",
    "impressions_last_30d",
    "clicks_last_30d",
    "sessions_last_30d",
    "impressions_90d",
    "clicks_90d",
    "pageviews_90d",
    "sessions_90d",
    "users_90d",
    "engaged_sessions_90d",
    "ai_sessions_90d",
    "scroll_events_90d",
    "ctr",
    "avg_position",
    "engagement_rate",
    "scroll_rate",
    "ai_traffic_pct",
    "days_with_impressions",
    "days_with_sessions",
    "impression_tier",
    "position_tier",
    "provider_used",
    "model_used",
    "freshness_tier",
}


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def display_path(path: Path | str) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def precision_at_k(y_true: Iterable[int], scores: Iterable[float], k: int) -> float:
    y = np.asarray(list(y_true))
    s = np.asarray(list(scores), dtype=float)
    if len(y) == 0:
        return 0.0
    order = np.argsort(-s, kind="stable")[: min(k, len(y))]
    return float(y[order].mean())
