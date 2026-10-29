import pytest

from evaluation.prediction_metrics import score_log_scale, score_minutes, score_by_bucket


def test_perfect_predictions():
    scores = score_log_scale([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert scores["mae_log"] == pytest.approx(0.0)
    assert scores["rmse_log"] == pytest.approx(0.0)
    assert scores["r2"] == pytest.approx(1.0)


def test_predicting_the_average_gives_r2_of_zero():
    scores = score_log_scale([1.0, 2.0, 3.0], [2.0, 2.0, 2.0])
    assert scores["mae_log"] == pytest.approx(2 / 3)
    assert scores["rmse_log"] == pytest.approx((2 / 3) ** 0.5)
    assert scores["r2"] == pytest.approx(0.0)


def test_scores_in_minutes():
    scores = score_minutes([10, 20, 30], [12, 18, 40])   # errors are 2, 2 and 10
    assert scores["mae_minutes"] == pytest.approx(14 / 3)
    assert scores["median_abs_error_minutes"] == pytest.approx(2.0)


def test_scores_by_duration_bucket():
    true_minutes = [10, 100, 1000]        # short, medium, long
    predicted_minutes = [20, 90, 800]
    scores = score_by_bucket(true_minutes, predicted_minutes)
    assert scores["short"]["mae_minutes"] == pytest.approx(10)
    assert scores["medium"]["mae_minutes"] == pytest.approx(10)
    assert scores["long"]["mae_minutes"] == pytest.approx(200)
    assert scores["long"]["jobs"] == 1