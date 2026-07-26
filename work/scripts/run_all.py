from __future__ import annotations

import subprocess
import sys

from ml_utils import OUTPUT_DIR, SCRIPTS_DIR, read_json


STEPS = [
    ("01_prepare_features.py", "Prepare the history-only feature vector and forward-decline label from the warehouse"),
    ("02_baseline_score.py", "Build the transparent rule baseline and ranked queue"),
    ("03_train_model.py", "Train and compare the models on a grouped out-of-fold split"),
    ("04_evaluate_and_export.py", "Validate the split, export the action queue and figure"),
]


def run_step(index: int, script: str, label: str) -> None:
    print(f"\n{'=' * 70}\nStep {index}/{len(STEPS)} — {label}\n{'=' * 70}", flush=True)
    subprocess.run([sys.executable, str(SCRIPTS_DIR / script)], cwd=SCRIPTS_DIR, check=True)


def main() -> None:
    for index, (script, label) in enumerate(STEPS, start=1):
        run_step(index, script, label)

    summary_path = OUTPUT_DIR / "summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        print("\nPipeline complete")
        print(f"Queue rows: {summary['queue_rows']:,}")
        print(f"Top-50 decline rate: {summary['queue_top50_decline_rate']:.3f}  (base {summary['base_rate']:.3f})")
        print(f"Grouped-split AUC: {summary['validation']['grouped_split_auc']:.3f}")


if __name__ == "__main__":
    main()
