import numpy as np
from sklearn.metrics import r2_score

# Duration buckets, in minutes of TRUE duration
SHORT_MAX_MINUTES = 30          # short:  under 30 minutes
MEDIUM_MAX_MINUTES = 12 * 60    # medium: 30 minutes to 12 hours, long: 12 hours or more


def score_log_scale(true_log, predicted_log):
    # Scores on the log1p(minutes) scale, the scale the models are trained on.
    errors = np.asarray(predicted_log) - np.asarray(true_log)
    return {
        "mae_log": float(np.mean(np.abs(errors))),
        "rmse_log": float(np.sqrt(np.mean(errors ** 2))),
        "r2": float(r2_score(true_log, predicted_log)),
    }


def score_minutes(true_minutes, predicted_minutes):
    # Scores in real minutes. The median error ignores a few huge mistakes on very long jobs.
    abs_errors = np.abs(np.asarray(predicted_minutes) - np.asarray(true_minutes))
    return {
        "mae_minutes": float(np.mean(abs_errors)),
        "median_abs_error_minutes": float(np.median(abs_errors)),
    }


def score_by_bucket(true_minutes, predicted_minutes):
    # Mean absolute error in minutes, separately for short, medium and long jobs.
    true_minutes = np.asarray(true_minutes)
    abs_errors = np.abs(np.asarray(predicted_minutes) - true_minutes)

    buckets = {
        "short": true_minutes < SHORT_MAX_MINUTES,
        "medium": (true_minutes >= SHORT_MAX_MINUTES) & (true_minutes < MEDIUM_MAX_MINUTES),
        "long": true_minutes >= MEDIUM_MAX_MINUTES,
    }
    scores = {}
    for name, in_bucket in buckets.items():
        scores[name] = {
            "jobs": int(in_bucket.sum()),
            "mae_minutes": float(abs_errors[in_bucket].mean()) if in_bucket.any() else float("nan"),
        }
    return scores