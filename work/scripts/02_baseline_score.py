from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml_utils import OUTPUT_DIR, precision_at_k, write_json


FEATURE_PATH = OUTPUT_DIR / "feature_vector.csv"
OUTPUT_PATH = OUTPUT_DIR / "baseline_action_score.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transparent rule baseline for the review queue.")
    parser.add_argument("--input", default=str(FEATURE_PATH))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    return parser.parse_args()


def score_rule(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    med_ctr = df["ctr_prev_30d"].median()

    conditions = {
        "thin_ctr": df["ctr_prev_30d"] < med_ctr,
        "stale_content": df["days_since_last_update"] >= 31,
        "losing_momentum": df["hist_impr_momentum"] < -0.05,
    }
    points = sum(c.astype(int) for c in conditions.values()).values
    matrix = pd.DataFrame({k: v.values for k, v in conditions.items()})
    codes = matrix.apply(lambda r: "|".join(matrix.columns[r.values]) or "none", axis=1).values
    return points, codes


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    if df.empty:
        raise ValueError("Feature vector is empty")

    base_rate = df["future_decline"].mean()

    df["risk_points"], df["reason_codes"] = score_rule(df)
    df["reach"] = df["reach_impr"]
    df["ctr_rank"] = df["ctr_prev_30d"].rank(pct=True)
    df["action"] = np.select(
        [df["risk_points"] >= 2, df["risk_points"] == 1],
        ["Priority review", "Review"],
        default="Monitor / deprioritize",
    )
    df["confidence"] = np.select(
        [df["risk_points"] >= 2, df["risk_points"] == 1],
        ["higher", "medium"],
        default="low",
    )

    df = df.sort_values(["risk_points", "ctr_rank", "content_id"], ascending=[False, True, True]).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    output_columns = [
        "rank", "content_id", "client_id", "risk_points", "reason_codes",
        "reach", "action", "confidence", "future_decline",
    ]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[output_columns].to_csv(output_path, index=False)

    labels = df["future_decline"].to_numpy()
    scores = -df["rank"].to_numpy()
    write_json(OUTPUT_DIR / "baseline_metadata.json", {
        "rows": int(len(df)),
        "base_rate": float(base_rate),
        "precision_at_20": precision_at_k(labels, scores, 20),
        "precision_at_50": precision_at_k(labels, scores, 50),
        "precision_at_100": precision_at_k(labels, scores, 100),
    })

    print(f"Wrote baseline queue: {output_path}")
    print(f"precision@50: {precision_at_k(labels, scores, 50):.3f}  (base rate {base_rate:.3f})")


if __name__ == "__main__":
    main()
