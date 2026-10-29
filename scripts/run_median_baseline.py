import json
from pathlib import Path
import numpy as np
import pandas as pd
from evaluation.prediction_metrics import score_log_scale, score_minutes, score_by_bucket

jobs = pd.read_csv("data/processed/jobs.csv")
train = jobs[jobs["split"] == "train"]

# The baseline: one constant for every job, taken from the TRAINING jobs only.
median_minutes = train["duration_minutes"].median()
median_log = np.log1p(median_minutes)
print(f"Training median duration: {median_minutes:.2f} minutes")

results = {"training_median_minutes": float(median_minutes)}
for split_name in ["validation", "test"]:
    part = jobs[jobs["split"] == split_name]
    true_log = part["log_minutes"].to_numpy()
    true_minutes = part["duration_minutes"].to_numpy()
    predicted_log = np.full(len(part), median_log)
    predicted_minutes = np.full(len(part), median_minutes)

    scores = {}
    scores.update(score_log_scale(true_log, predicted_log))
    scores.update(score_minutes(true_minutes, predicted_minutes))
    scores["by_bucket"] = score_by_bucket(true_minutes, predicted_minutes)
    results[split_name] = scores

    print(f"\n{split_name} ({len(part)} jobs)")
    for name, value in scores.items():
        if name != "by_bucket":
            print(f"  {name:<26} {value:.4f}")
    for bucket, bucket_scores in scores["by_bucket"].items():
        print(f"  {bucket:<8} jobs={bucket_scores['jobs']:>6}  "
                f"mae_minutes={bucket_scores['mae_minutes']:.2f}")

Path("results").mkdir(exist_ok=True)
with open("results/median_baseline.json", "w") as file:
    json.dump(results, file, indent=2)
print("\nSaved to results/median_baseline.json")