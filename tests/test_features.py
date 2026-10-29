import pandas as pd
import pytest

from ml.features import make_features, FEATURE_NAMES


def test_features_for_a_job_submitted_at_six_in_the_morning():
    jobs = pd.DataFrame({
        "starttime": [3.25],          # day 3, 0.25 of the day = 6 am
        "core": [0.1], "memory": [0.2], "ssd": [0.3], "nic": [0.4],
        "duration_days": [99.0],      # must NOT be used as a feature
    })
    features = make_features(jobs)
    assert list(features.columns) == FEATURE_NAMES
    assert features["hour_sin"][0] == pytest.approx(1.0)
    assert features["hour_cos"][0] == pytest.approx(0.0, abs=1e-9)
    assert features["core"][0] == pytest.approx(0.1)
    assert "duration_days" not in features.columns