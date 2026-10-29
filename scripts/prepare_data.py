import numpy as np
import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/Azure_Data2.csv")
OUT_PATH = Path("data/processed/jobs.csv")

# Time-based split by submission day (no shuffling).
TRAIN_LAST_DAY = 9          # train:      days 0 to 9
VALIDATION_LAST_DAY = 11    # validation: days 10 and 11, test: days 12 and 13

jobs = pd.read_csv(RAW_PATH)

# Target: how long the job ran. All jobs are kept, even very short ones.
jobs["duration_days"] = jobs["endtime"] - jobs["starttime"]
jobs["duration_minutes"] = jobs["duration_days"] * 24 * 60
jobs["log_minutes"] = np.log1p(jobs["duration_minutes"])

# Time features known at submission.
jobs["start_day"] = jobs["starttime"].astype(int)
jobs["hour_of_day"] = (jobs["starttime"] % 1) * 24

jobs["split"] = "test"
jobs.loc[jobs["start_day"] <= VALIDATION_LAST_DAY, "split"] = "validation"
jobs.loc[jobs["start_day"] <= TRAIN_LAST_DAY, "split"] = "train"

# Schedulers need jobs in order of arrival.
jobs = jobs.sort_values("starttime").reset_index(drop=True)

# endtime is not saved: it is only known after the job finishes.
# machineId and priority are not saved: they are constant in this data.
columns = [
    "vmId", "vmTypeId", "starttime", "start_day", "hour_of_day",
    "core", "memory", "ssd", "nic",
    "duration_days", "duration_minutes", "log_minutes", "split",
]
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
jobs[columns].to_csv(OUT_PATH, index=False)

summary = jobs.groupby("split").agg(
    jobs=("vmId", "count"),
    first_day=("start_day", "min"),
    last_day=("start_day", "max"),
    median_minutes=("duration_minutes", "median"),
)
print(summary)
print("\nTotal jobs saved:", len(jobs))
print("Saved to:", OUT_PATH)