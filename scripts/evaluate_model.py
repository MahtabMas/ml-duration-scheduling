import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from ml.features import make_features
from evaluation.prediction_metrics import score_log_scale, score_minutes, score_by_bucket

# Develop on "validation". Run "test" only for the final results.
split_name = sys.argv[1] if len(sys.argv) > 1 else "validation"

jobs = pd.read_csv("data/processed/jobs.csv")
part = jobs[jobs["split"] == split_name]

model = joblib.load("models/gb_model.joblib")
predicted_log = model.predict(make_features(part))
predicted_minutes = np.maximum(np.expm1(predicted_log), 0.0)   # a duration cannot be negative

true_log = part["log_minutes"].to_numpy()
true_minutes = part["duration_minutes"].to_numpy()

scores = {}
scores.update(score_log_scale(true_log, predicted_log))
scores.update(score_minutes(true_minutes, predicted_minutes))
scores["by_bucket"] = score_by_bucket(true_minutes, predicted_minutes)

baseline = json.load(open("results/median_baseline.json"))[split_name]

print(f"{split_name}: {len(part)} jobs\n")
print(f"{'metric':<28}{'median':>12}{'grad. boost.':>14}")
for name in ["mae_log", "rmse_log", "r2", "mae_minutes", "median_abs_error_minutes"]:
    print(f"{name:<28}{baseline[name]:>12.4f}{scores[name]:>14.4f}")

print(f"\n{'bucket (true duration)':<28}{'jobs':>6}{'median MAE':>12}{'model MAE':>12}   (minutes)")
for bucket in ["short", "medium", "long"]:
    print(f"{bucket:<28}{scores['by_bucket'][bucket]['jobs']:>6}"
          f"{baseline['by_bucket'][bucket]['mae_minutes']:>12.2f}"
          f"{scores['by_bucket'][bucket]['mae_minutes']:>12.2f}")

Path("results").mkdir(exist_ok=True)
with open(f"results/gb_{split_name}.json", "w") as file:
    json.dump(scores, file, indent=2)
print(f"\nSaved to results/gb_{split_name}.json")