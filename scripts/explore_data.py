import pandas as pd

jobs = pd.read_csv("data/raw/Azure_Data2.csv")
jobs["duration_days"] = jobs["endtime"] - jobs["starttime"]
jobs["duration_minutes"] = jobs["duration_days"] * 24 * 60

print("Rows:", len(jobs))
print("\nMissing values per column:\n", jobs.isna().sum())

print("\nmachineId counts:\n", jobs["machineId"].value_counts().sort_index())
print("\npriority counts:\n", jobs["priority"].value_counts().sort_index())
print("\nDistinct vmTypeId:", jobs["vmTypeId"].nunique())

print("\nstarttime range:", jobs["starttime"].min(), "to", jobs["starttime"].max())
print("endtime range:  ", jobs["endtime"].min(), "to", jobs["endtime"].max())

print("\nJobs per start day:\n", jobs.groupby(jobs["starttime"].astype(int)).size())

print("\nDuration in minutes:\n",
        jobs["duration_minutes"].describe(percentiles=[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]))
print("\nDuration <= 0 minutes:", (jobs["duration_minutes"] <= 0).sum())
print("Duration < 1 minute:   ", (jobs["duration_minutes"] < 1).sum())
print("Duration < 5 minutes:  ", (jobs["duration_minutes"] < 5).sum())

last_end = jobs["endtime"].max()
one_minute = 1 / (24 * 60)
print("\nJobs ending within 1 minute of the last endtime:",
        (jobs["endtime"] > last_end - one_minute).sum())