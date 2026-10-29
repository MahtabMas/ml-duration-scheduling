import pytest

from scheduler.job import Job
from scheduler.greedy import Greedy

CAPACITY = [1.0, 1.0, 1.0, 1.0]


def make_job(job_id, start, true_duration, size, predicted_duration=None):
    return Job(job_id, start, true_duration, [size] * 4, predicted_duration)


def test_greedy_with_exact_durations():
    jobs = [
        make_job(1, 0.0, 2.0, 0.5),
        make_job(2, 0.0, 1.0, 0.5),
        make_job(3, 0.5, 2.0, 0.5),
        make_job(4, 1.5, 1.0, 0.5),
    ]
    greedy = Greedy(CAPACITY)
    for job in jobs:
        greedy.add_job(job)
    assert greedy.num_servers() == 2
    assert greedy.cost() == pytest.approx(4.0)


def test_ties_go_to_the_server_opened_first():
    job1 = make_job(1, 0.0, 5.0, 0.6)     # server A, finishes at 5.0
    job2 = make_job(2, 0.1, 8.9, 0.6)     # server B, finishes at 9.0
    job3 = make_job(3, 0.2, 2.8, 0.3)     # ends at 3.0: extension 0 for both
    greedy = Greedy(CAPACITY)
    for job in (job1, job2, job3):
        greedy.add_job(job)
    assert greedy.all_servers[0].cur_size[0] == pytest.approx(0.9)


def test_picks_the_smallest_extension_when_every_server_must_be_extended():
    job1 = make_job(1, 0.0, 1.0, 0.6)     # server A, finishes at 1.0
    job2 = make_job(2, 0.1, 1.4, 0.6)     # server B, finishes at 1.5
    job3 = make_job(3, 0.2, 5.0, 0.3)     # ends at 5.2: extends A by 4.2 or B by 3.7
    greedy = Greedy(CAPACITY)
    for job in (job1, job2, job3):
        greedy.add_job(job)
    assert greedy.all_servers[1].cur_size[0] == pytest.approx(0.9)
    assert greedy.all_servers[0].cur_size[0] == pytest.approx(0.6)


def test_extension_is_measured_with_believed_times_not_true_times():
    job1 = make_job(1, 0.0, 10.0, 0.6, predicted_duration=1.0)   # server A: believed end 1.0
    job2 = make_job(2, 0.1, 5.0, 0.6, predicted_duration=6.0)    # server B: believed end 6.1
    job3 = make_job(3, 0.2, 1.0, 0.3)                            # ends at 1.2
    greedy = Greedy(CAPACITY)
    for job in (job1, job2, job3):
        greedy.add_job(job)
    # Believed: A would be extended by 0.2, B by 0, so B. (With true times both would be 0 and A would win.)
    assert greedy.all_servers[1].cur_size[0] == pytest.approx(0.9)