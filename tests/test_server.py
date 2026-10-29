from scheduler.job import Job
from scheduler.server import Server

CAPACITY = [1.0, 1.0, 1.0, 1.0]


def make_job(job_id, start, true_duration, size, predicted_duration=None):
    return Job(job_id, start, true_duration, [size] * 4, predicted_duration)


def test_believed_end_defaults_to_true_end():
    job = make_job(1, 0.0, 2.0, 0.5)
    assert job.end == 2.0
    assert job.believed_end == 2.0


def test_believed_end_uses_prediction_when_given():
    job = make_job(1, 0.0, 2.0, 0.5, predicted_duration=0.5)
    assert job.end == 2.0
    assert job.believed_end == 0.5


def test_capacity_is_not_freed_at_predicted_end():
    # Job 1 really runs until t=2.0, but we predicted it would end at t=0.5.
    server = Server(CAPACITY, make_job(1, 0.0, 2.0, 0.6, predicted_duration=0.5))
    # At t=1.0 job 1 is still running, so another job of size 0.6 must not fit.
    assert server.add_job(make_job(2, 1.0, 1.0, 0.6)) is False


def test_capacity_is_freed_after_true_end():
    server = Server(CAPACITY, make_job(1, 0.0, 2.0, 0.6, predicted_duration=0.5))
    server.add_job(make_job(3, 0.1, 5.0, 0.2))
    # At t=2.5 job 1 has really ended, so size 0.6 fits next to job 3 (0.2).
    assert server.add_job(make_job(2, 2.5, 1.0, 0.6)) is True


def test_server_stays_active_until_true_end():
    server = Server(CAPACITY, make_job(1, 0.0, 2.0, 0.6, predicted_duration=0.5))
    assert server.is_active_at(1.0) is True
    assert server.is_active_at(2.5) is False


def test_ranking_uses_believed_finishing_time():
    server = Server(CAPACITY, make_job(1, 0.0, 2.0, 0.6, predicted_duration=0.5))
    assert server.believed_finishing_time() == 0.5
    assert server.finishing_time() == 2.0