import sys
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scheduler.job import Job
from scheduler.first_fit import FirstFit
from scheduler.greedy import Greedy
from scheduler.predictive_greedy import PredictiveGreedy
from evaluation.metrics import compute_opt, print_summary
from ml.features import make_features


# As in the paper: sizes are whole numbers and every server has capacity 1000.
SIZE_SCALE = 1000
CAPACITY = [SIZE_SCALE] * 4
MINUTES_PER_DAY = 24 * 60

# Develop on "validation". Run "test" only for the final results.
split_name = sys.argv[1] if len(sys.argv) > 1 else "validation"

all_jobs = pd.read_csv("data/processed/jobs.csv")
train = all_jobs[all_jobs["split"] == "train"]
period = all_jobs[all_jobs["split"] == split_name].sort_values("starttime", kind="stable")
assert period["vmId"].is_unique
print(f"Scheduling the {split_name} days: {len(period)} jobs "
      f"(days {period['start_day'].min()} to {period['start_day'].max()})")


def to_size(fraction):
    return int(round(fraction * SIZE_SCALE))


jobs = [
    Job(int(row.vmId), row.starttime, row.duration_days,
        [to_size(row.core), to_size(row.memory), to_size(row.ssd), to_size(row.nic)])
    for row in period.itertuples()
]

# Predictor 1, no information: the same duration (in days) for every job,
# taken from the TRAINING jobs only.
median_days = train["duration_days"].median()

# Predictor 2, the trained model: it sees only size and submission hour.
model = joblib.load("models/gb_model.joblib")
predicted_minutes = np.maximum(np.expm1(model.predict(make_features(period))), 0.0)
model_days = dict(zip(period["vmId"].astype(int), predicted_minutes / MINUTES_PER_DAY))


def run(scheduler):
    for job in jobs:
        scheduler.add_job(job)
    return scheduler


first_fit = run(FirstFit(CAPACITY))
greedy = run(Greedy(CAPACITY))
greedy_median = run(PredictiveGreedy(CAPACITY, predict_duration=lambda job: median_days))
greedy_model = run(PredictiveGreedy(CAPACITY, predict_duration=lambda job: model_days[job.job_id]))

opt = compute_opt(jobs, capacity=SIZE_SCALE)

results = {
    "First Fit": (first_fit.cost(), first_fit.num_servers()),
    "Greedy (true durations)": (greedy.cost(), greedy.num_servers()),
    "Greedy (median duration)": (greedy_median.cost(), greedy_median.num_servers()),
    "Greedy (model durations)": (greedy_model.cost(), greedy_model.num_servers()),
}
print_summary(results, opt)



Path("results").mkdir(exist_ok=True)
saved = {
    name: {"cost": cost, "servers": servers, "ratio": cost / opt}
    for name, (cost, servers) in results.items()
}
saved["opt_lower_bound"] = opt
with open(f"results/scheduling_{split_name}.json", "w") as file:
    json.dump(saved, file, indent=2)
print(f"Saved to results/scheduling_{split_name}.json")