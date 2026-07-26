from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml_utils import (
    ANCHOR,
    CAT_FEATURES,
    FORBIDDEN_FEATURES,
    NUM_FEATURES,
    OUTPUT_DIR,
    WAREHOUSE,
    display_path,
    ensure_dirs,
    warehouse_connection,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the history-only feature vector and forward-decline label from the warehouse.")
    parser.add_argument("--output", default=str(OUTPUT_DIR / "feature_vector.csv"))
    return parser.parse_args()


def fetch(con) -> pd.DataFrame:
    fact = "[" + ",".join(f"'{WAREHOUSE}/fact_content_daily_performance/month=2026-0{i}/*.parquet'"
                          for i in [1, 2, 3, 4]) + "]"
    sql = f"""
    WITH elig AS (
      SELECT client_hash_id FROM read_parquet('{WAREHOUSE}/dim_clients.parquet')
      WHERE has_gsc_access AND gsc_data_start <= DATE '2026-01-01'
    ),
    perf AS (
      SELECT client_hash_id, content_hash_id,
        SUM(gsc_impressions) FILTER (WHERE month='2026-03')                        AS impr_prev,
        SUM(gsc_clicks)      FILTER (WHERE month='2026-03')                        AS clicks_prev,
        SUM(ga4_sessions)    FILTER (WHERE month='2026-03' AND ga4_data_available) AS sess_prev,
        SUM(gsc_impressions) FILTER (WHERE month='2026-02')                        AS impr_mid,
        SUM(gsc_impressions) FILTER (WHERE month='2026-01')                        AS impr_old,
        SUM(gsc_impressions) FILTER (WHERE month='2026-04')                        AS impr_future
      FROM read_parquet({fact}, hive_partitioning=true, union_by_name=true)
      WHERE client_hash_id IN (SELECT client_hash_id FROM elig)
      GROUP BY 1, 2
    )
    SELECT p.client_hash_id, p.content_hash_id,
           p.impr_prev, p.clicks_prev, p.sess_prev, p.impr_mid, p.impr_old, p.impr_future,
           c.content_type, c.main_intent, c.competition_level,
           c.search_volume, c.competition, c.cpc, c.word_count, c.char_count,
           c.content_created_date, c.content_updated_date
    FROM perf p
    JOIN read_parquet('{WAREHOUSE}/dim_content.parquet') c USING (client_hash_id, content_hash_id)
    WHERE p.impr_prev > 0
      AND c.is_published AND NOT c.is_deleted
      AND c.content_created_date <= DATE '{ANCHOR}'
    """
    return con.sql(sql).df()


def engineer(raw: pd.DataFrame) -> pd.DataFrame:
    anchor = pd.Timestamp(ANCHOR)
    for col in ["impr_prev", "clicks_prev", "sess_prev", "impr_mid", "impr_old", "impr_future"]:
        raw[col] = raw[col].fillna(0.0)

    raw["future_decline"] = (raw["impr_future"] < 0.8 * raw["impr_prev"]).astype(int)

    eps = 1.0
    raw["ctr_prev_30d"] = raw["clicks_prev"] / (raw["impr_prev"] + eps)
    raw["hist_impr_momentum"] = (raw["impr_prev"] - raw["impr_mid"]) / (raw["impr_mid"] + eps)
    raw["log_impressions_prev_30d"] = np.log1p(raw["impr_prev"])
    raw["log_clicks_prev_30d"] = np.log1p(raw["clicks_prev"])
    raw["log_sessions_prev_30d"] = np.log1p(raw["sess_prev"])
    raw["log_impr_older_30d"] = np.log1p(raw["impr_old"])
    raw["log_search_volume"] = np.log1p(raw["search_volume"].fillna(0))

    created = pd.to_datetime(raw["content_created_date"])
    updated = pd.to_datetime(raw["content_updated_date"])
    eff_update = updated.where(updated.notna() & (updated <= anchor), created)
    raw["content_age_days"] = (anchor - created).dt.days.clip(lower=0)
    raw["days_since_last_update"] = (anchor - eff_update).dt.days.clip(lower=0)
    raw["age_tier_order"] = pd.cut(raw["content_age_days"], [-1, 7, 30, 90, 180, 365, 1e9], labels=False) + 1

    raw["has_keyword_data"] = raw["search_volume"].notna().astype(int)
    raw["has_word_count"] = raw["word_count"].notna().astype(int)
    raw["reach_impr"] = raw["impr_prev"].round().astype(int)
    raw["content_id"] = raw["content_hash_id"]
    raw["client_id"] = raw["client_hash_id"]

    for col in NUM_FEATURES:
        raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0.0)
    for col in CAT_FEATURES:
        raw[col] = raw[col].fillna("unknown").astype(str)
    return raw


def main() -> None:
    args = parse_args()
    ensure_dirs()

    con = warehouse_connection()
    raw = fetch(con)
    print(f"fetched {len(raw):,} pairs from the warehouse ({raw.client_hash_id.nunique()} clients)")

    df = engineer(raw)
    keep = ["content_id", "client_id"] + NUM_FEATURES + CAT_FEATURES + ["future_decline", "reach_impr"]
    features = df[keep].sort_values("content_id").reset_index(drop=True)

    leaked = FORBIDDEN_FEATURES & set(features.columns)
    if leaked:
        raise ValueError(f"Forbidden columns leaked into feature vector: {leaked}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)

    write_json(OUTPUT_DIR / "feature_metadata.json", {
        "source": "warehouse fact_content_daily_performance months 2026-01..2026-04",
        "anchor": ANCHOR,
        "output": display_path(output_path),
        "modeled_rows": int(len(features)),
        "clients": int(features["client_id"].nunique()),
        "decline_base_rate": float(features["future_decline"].mean()),
        "numeric_features": NUM_FEATURES,
        "categorical_features": CAT_FEATURES,
        "label_definition": "April 2026 impressions < 0.8 * March 2026 impressions",
        "feature_window": "trailing 90 days (Jan-Mar 2026)",
        "label_window": "April 2026 (forward 30 days)",
    })

    print(f"Prepared {len(features):,} rows | decline base rate {features['future_decline'].mean():.3f}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
