import random

import pytest

from scheduler.job import Job
from scheduler.greedy import Greedy
from scheduler.predictive_greedy import PredictiveGreedy
from evaluation.metrics import compute_opt

CAPACITY = [1.0, 1.0, 1.0, 1.0]
SIZES = [0.05, 0.1, 0.2, 0.3, 0.5]


def random_jobs(n, seed):
    # Synthetic jobs: about 100 arrivals per day, lognormal durations, mixed sizes.
    rng = random.Random(seed)
    jobs, time = [], 0.0
    for job_id in range(n):
        time += rng.expovariate(100)
        duration = rng.lognormvariate(-2.0, 1.2)
        size = rng.choice(SIZES)
        jobs.append(Job(job_id, time, duration, [size] * 4))
    return jobs


def max_true_load(server):
    # Independent check, not using the Server's own bookkeeping:
    # sweep over the TRUE start and end times of every job ever placed on the server.
    events = []
    for job in server.placed_jobs:
        events.append((job.start, 1, job))
        events.append((job.end, -1, job))
    events.sort(key=lambda event: (event[0], event[1]))  # at equal times, ends come before starts
    load = [0.0] * 4
    worst = 0.0
    for _, sign, job in events:
        for d in range(4):
            load[d] += sign * job.size[d]
        worst = max(worst, max(load))
    return worst


def test_oracle_predictor_equals_greedy():
    jobs = random_jobs(500, seed=1)
    plain = Greedy(CAPACITY)
    oracle = PredictiveGreedy(CAPACITY, predict_duration=lambda job: job.duration)
    for job in jobs:
        plain.add_job(job)
        oracle.add_job(job)
    assert oracle.num_servers() == plain.num_servers()
    assert oracle.cost() == pytest.approx(plain.cost())


@pytest.mark.parametrize("factor", [0.3, 0.72, 1.0, 1.5])
def test_wrong_predictions_never_overload_a_server(factor):
    jobs = random_jobs(500, seed=2)
    scheduler = PredictiveGreedy(CAPACITY, predict_duration=lambda job: job.duration * factor)
    for job in jobs:
        scheduler.add_job(job)
    for server in scheduler.all_servers:
        assert max_true_load(server) <= 1.0 + 1e-9


@pytest.mark.parametrize("factor", [0.3, 0.72, 1.0, 1.5])
def test_cost_is_never_below_the_lower_bound(factor):
    jobs = random_jobs(500, seed=3)
    scheduler = PredictiveGreedy(CAPACITY, predict_duration=lambda job: job.duration * factor)
    for job in jobs:
        scheduler.add_job(job)
    assert scheduler.cost() >= compute_opt(jobs) - 1e-9