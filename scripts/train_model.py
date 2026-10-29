import json
import random
import time
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from ml.features import make_features
from evaluation.prediction_metrics import score_log_scale

SEED = 42
N_CONFIGS = 10      # the fixed search budget

# The small grid the random search draws from.
GRID = {
    "n_estimators": [100, 200, 400],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.02, 0.05, 0.1],
    "min_samples_leaf": [10, 50],
}

jobs = pd.read_csv("data/processed/jobs.csv")
train = jobs[jobs["split"] == "train"]
validation = jobs[jobs["split"] == "validation"]

train_features = make_features(train)
train_target = train["log_minutes"]
validation_features = make_features(validation)
validation_target = validation["log_minutes"]

# Draw N_CONFIGS different settings, repeatable thanks to the seed.
rng = random.Random(SEED)
configs = []
while len(configs) < N_CONFIGS:
    config = {name: rng.choice(values) for name, values in GRID.items()}
    if config not in configs:
        configs.append(config)

results = []
best_model = None
best_mae = float("inf")
for number, config in enumerate(configs, start=1):
    began = time.time()
    model = GradientBoostingRegressor(random_state=SEED, **config)
    model.fit(train_features, train_target)
    seconds = time.time() - began

    scores = score_log_scale(validation_target, model.predict(validation_features))
    results.append({"config": config, **scores, "fit_seconds": round(seconds, 1)})
    print(f"{number:>2}/{N_CONFIGS}  {config}  "
          f"validation mae_log={scores['mae_log']:.4f}  r2={scores['r2']:.4f}  ({seconds:.0f} s)")

    if scores["mae_log"] < best_mae:
        best_mae = scores["mae_log"]
        best_model = model
        best_config = config

print(f"\nBest by validation MAE (log scale): {best_config}  ->  {best_mae:.4f}")
print("Median baseline on validation was: 1.3282 (R2 -0.1522)")

Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)
joblib.dump(best_model, "models/gb_model.joblib")
with open("results/gb_search.json", "w") as file:
    json.dump({"seed": SEED, "best_config": best_config, "runs": results}, file, indent=2)
print("Saved models/gb_model.joblib and results/gb_search.json")