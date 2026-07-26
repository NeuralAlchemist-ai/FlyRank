from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_utils import CAT_FEATURES, NUM_FEATURES, OUTPUT_DIR, SEED, precision_at_k, write_json


FEATURE_PATH = OUTPUT_DIR / "feature_vector.csv"
OUTPUT_PATH = OUTPUT_DIR / "model_scores.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare the decline-risk models (grouped out-of-fold).")
    parser.add_argument("--features", default=str(FEATURE_PATH))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    return parser.parse_args()


def logistic_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, random_state=SEED))])


def hgb_pipeline() -> Pipeline:
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES)], remainder="passthrough")
    return Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier(random_state=SEED))])


def rule_points(df: pd.DataFrame) -> np.ndarray:
    med_ctr = df["ctr_prev_30d"].median()
    points = ((df["ctr_prev_30d"] < med_ctr).astype(int)
              + (df["days_since_last_update"] >= 31).astype(int)
              + (df["hist_impr_momentum"] < -0.05).astype(int))
    return points.to_numpy() + (0.5 - df["ctr_prev_30d"].rank(pct=True).to_numpy()) * 1e-3


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.features)
    if df["future_decline"].nunique() < 2:
        raise ValueError("Label has only one class; cannot train")

    y = df["future_decline"].to_numpy()
    groups = df["client_id"].to_numpy()
    base_rate = float(y.mean())
    gkf = GroupKFold(n_splits=5)

    p_logit = cross_val_predict(logistic_pipeline(), df[NUM_FEATURES + CAT_FEATURES], y,
                                cv=gkf, groups=groups, method="predict_proba")[:, 1]
    p_hgb = cross_val_predict(hgb_pipeline(), df[NUM_FEATURES + CAT_FEATURES], y,
                              cv=gkf, groups=groups, method="predict_proba")[:, 1]
    baseline = rule_points(df)

    metrics = {"base_rate": base_rate}
    for name, scores, has_auc in [("baseline", baseline, False), ("logistic", p_logit, True), ("hgb", p_hgb, True)]:
        metrics[name] = {
            "auc": float(roc_auc_score(y, scores)) if has_auc else None,
            "precision_at_20": precision_at_k(y, scores, 20),
            "precision_at_50": precision_at_k(y, scores, 50),
            "precision_at_100": precision_at_k(y, scores, 100),
        }

    df["p_logit"] = p_logit.round(4)
    df["p_hgb"] = p_hgb.round(4)
    df["baseline_points"] = np.rint(baseline).astype(int)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[["content_id", "client_id", "reach_impr", "p_logit", "p_hgb", "baseline_points", "future_decline"]].to_csv(output_path, index=False)

    write_json(OUTPUT_DIR / "model_results.json", {
        "rows": int(len(df)),
        "clients": int(df["client_id"].nunique()),
        "split": "GroupKFold(5) out-of-fold by client",
        "deployed_model": "logistic",
        "metrics": metrics,
    })

    print(f"Trained on {len(df):,} rows / {df['client_id'].nunique()} clients (grouped out-of-fold)")
    print(f"logistic precision@50: {metrics['logistic']['precision_at_50']:.3f}  (base rate {base_rate:.3f})")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
