from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_utils import CAT_FEATURES, NUM_FEATURES, OUTPUT_DIR, SEED, precision_at_k, write_json


REACH_MIN = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the split, then export the action queue and figure.")
    parser.add_argument("--features", default=str(OUTPUT_DIR / "feature_vector.csv"))
    parser.add_argument("--scores", default=str(OUTPUT_DIR / "model_scores.csv"))
    parser.add_argument("--queue", default=str(OUTPUT_DIR / "content_action_queue.csv"))
    parser.add_argument("--figure", default=str(OUTPUT_DIR / "fig_precision_at_k.png"))
    return parser.parse_args()


def logistic_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, random_state=SEED))])


def validation_gap(df: pd.DataFrame) -> dict:
    y = df["future_decline"].to_numpy()

    x_tr, x_te, y_tr, y_te = train_test_split(df[NUM_FEATURES + CAT_FEATURES], y, test_size=0.3, random_state=SEED)
    random_auc = roc_auc_score(y_te, logistic_pipeline().fit(x_tr, y_tr).predict_proba(x_te)[:, 1])

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED).split(df, groups=df["client_id"]))
    grouped_auc = roc_auc_score(y[te], logistic_pipeline().fit(df.iloc[tr][NUM_FEATURES + CAT_FEATURES], y[tr])
                                .predict_proba(df.iloc[te][NUM_FEATURES + CAT_FEATURES])[:, 1])
    return {"random_split_auc": float(random_auc), "grouped_split_auc": float(grouped_auc)}


def build_queue(baseline_path: str) -> pd.DataFrame:
    base_q = pd.read_csv(baseline_path)
    queue = base_q[base_q["reach"] >= REACH_MIN].reset_index(drop=True)
    queue["rank"] = np.arange(1, len(queue) + 1)
    queue["confidence"] = np.select([queue["risk_points"] >= 2, queue["risk_points"] == 1],
                                     ["higher", "medium"], default="low")
    queue["human_review_required"] = True
    return queue


def export_figure(scores_path: str, features_path: str, path: Path) -> None:
    scores = pd.read_csv(scores_path)
    feat = pd.read_csv(features_path)[["content_id", "client_id", "ctr_prev_30d"]]
    m = scores.merge(feat, on=["content_id", "client_id"])
    sub = m[m["reach_impr"] >= REACH_MIN]
    y = sub["future_decline"].to_numpy()
    ks = [10, 20, 30, 50, 75, 100, 150, 200]
    base_score = sub["baseline_points"].to_numpy() + (0.5 - sub["ctr_prev_30d"].rank(pct=True).to_numpy()) * 1e-3
    base_pk = [precision_at_k(y, base_score, k) for k in ks]
    model_pk = [precision_at_k(y, sub["p_logit"].to_numpy(), k) for k in ks]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.axhline(y.mean(), ls="--", color="grey", label=f"base rate {y.mean():.2f}")
    ax.plot(ks, base_pk, "o-", color="#B279A7", label="rule baseline (deployed)")
    ax.plot(ks, model_pk, "s-", color="#4C78A8", label="logistic model")
    ax.set_xlabel("K (top-ranked pages reviewed)")
    ax.set_ylabel("precision@K (share that declined)")
    ax.set_title("Queue quality on editorial-relevant pages (reach >= 100)")
    ax.set_ylim(0.4, 0.9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features)

    gap = validation_gap(features)

    queue = build_queue(str(OUTPUT_DIR / "baseline_action_score.csv"))
    columns = ["rank", "content_id", "client_id", "reach", "risk_points", "reason_codes",
               "action", "confidence", "human_review_required", "future_decline"]
    queue_path = Path(args.queue)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue[columns].to_csv(queue_path, index=False)

    export_figure(args.scores, args.features, Path(args.figure))

    write_json(OUTPUT_DIR / "summary.json", {
        "validation": gap,
        "queue_rows": int(len(queue)),
        "queue_reach_min": REACH_MIN,
        "queue_top50_decline_rate": float(queue["future_decline"].head(50).mean()),
        "base_rate": float(features["future_decline"].mean()),
        "queue_output": str(queue_path),
        "figure_output": str(args.figure),
    })

    print(f"Validation AUC  random={gap['random_split_auc']:.3f}  grouped={gap['grouped_split_auc']:.3f}")
    print(f"Wrote queue ({len(queue):,} rows): {queue_path}")
    print(f"Wrote figure: {args.figure}")


if __name__ == "__main__":
    main()
