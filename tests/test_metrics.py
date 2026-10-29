import math
import random

import pytest

from scheduler.job import Job
from evaluation.metrics import compute_opt

TOLERANCE = 1e-9


def brute_force_opt(jobs, capacity=1.0):
    # Slow reference version: for every time segment, add up the loads of all active jobs.
    event_times = sorted(set(t for job in jobs for t in (job.start, job.end)))
    num_dims = len(jobs[0].size)
    total = 0.0
    for i in range(len(event_times) - 1):
        t_start = event_times[i]
        load = [0.0] * num_dims
        for job in jobs:
            if job.start <= t_start < job.end:
                for d in range(num_dims):
                    load[d] += job.size[d]
        needed = max(math.ceil(load[d] / capacity - TOLERANCE) for d in range(num_dims))
        if needed > 0:
            total += (event_times[i + 1] - t_start) * needed
    return total


def test_opt_on_a_small_example_worked_out_by_hand():
    jobs = [
        Job(1, 0.0, 2.0, [0.5] * 4),   # runs 0.0 to 2.0
        Job(2, 0.0, 1.0, [0.5] * 4),   # runs 0.0 to 1.0
        Job(3, 0.5, 2.0, [0.5] * 4),   # runs 0.5 to 2.5
        Job(4, 1.5, 1.0, [0.5] * 4),   # runs 1.5 to 2.5
    ]
    # Servers needed per segment: 1, 2, 1, 2, 1, each segment lasting 0.5
    assert compute_opt(jobs) == pytest.approx(3.5)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_opt_matches_the_slow_reference(seed):
    rng = random.Random(seed)
    jobs, time = [], 0.0
    for job_id in range(200):
        time += rng.expovariate(50)
        duration = rng.lognormvariate(-2.0, 1.0)
        size = rng.choice([0.125, 0.25, 0.5])
        jobs.append(Job(job_id, time, duration, [size] * 4))
    assert compute_opt(jobs) == pytest.approx(brute_force_opt(jobs))