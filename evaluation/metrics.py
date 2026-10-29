import math

# Rounding a load up to a whole number of servers must ignore tiny floating point noise,
# e.g. 1.0000000000000002 should count as 1 server, not 2.
TOLERANCE = 1e-9


def compute_opt(jobs, capacity=1.0):
    # Lower bound on the cost of ANY schedule (it is a lower bound, not the true optimum):
    # in every moment, at least ceil(total load / capacity) servers must be running,
    # in the dimension where the load is highest.
    # Sweep line: walk through all start and end events in time order, keeping the running load.
    num_dims = len(jobs[0].size)

    events = []
    for job in jobs:
        events.append((job.start, 1, job.size))
        events.append((job.end, -1, job.size))
    events.sort(key=lambda event: event[0])

    load = [0.0] * num_dims
    opt_cost = 0.0
    previous_time = events[0][0]

    for time, sign, size in events:
        if time > previous_time:
            # The load has been constant since previous_time: price that stretch.
            servers_needed = max(math.ceil(load[d] / capacity - TOLERANCE)
                                 for d in range(num_dims))
            if servers_needed > 0:
                opt_cost += (time - previous_time) * servers_needed
            previous_time = time
        for d in range(num_dims):
            load[d] += sign * size[d]

    return opt_cost


def competitive_ratio(algorithm_cost, opt_cost):
    # ratio = 1.0 means optimal, ratio = 2.0 means twice the lower bound
    if opt_cost == 0:
        return float('inf')
    return algorithm_cost / opt_cost


def print_summary(results, opt_cost):
    print("\n" + "=" * 60)
    print(f"{'EVALUATION SUMMARY':^60}")
    print("=" * 60)
    print(f"OPT lower bound: {opt_cost:.4f}")
    print("-" * 60)
    print(f"{'Algorithm':<20} {'Cost':>10} {'Servers':>10} {'Ratio':>10}")
    print("-" * 60)
    for name, (cost, servers) in results.items():
        ratio = competitive_ratio(cost, opt_cost)
        print(f"{name:<20} {cost:>10.4f} {servers:>10} {ratio:>10.4f}")
    print("=" * 60)