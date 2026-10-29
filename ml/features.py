import numpy as np
import pandas as pd

# Everything here must be known when the job is submitted.
FEATURE_NAMES = ["core", "memory", "ssd", "nic", "hour_sin", "hour_cos"]


def make_features(jobs):
    # starttime is in days: the part after the decimal point is the time of day.
    hour = (jobs["starttime"] % 1) * 24
    features = pd.DataFrame({
        "core": jobs["core"],
        "memory": jobs["memory"],
        "ssd": jobs["ssd"],
        "nic": jobs["nic"],
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
    })
    return features[FEATURE_NAMES]