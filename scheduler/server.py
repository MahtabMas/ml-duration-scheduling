from sortedcontainers import SortedDict


# One server in the scheduling simulation.
# Capacity and "is the server still in use" depend on TRUE job end times.
# Only believed_finishing_time() uses the scheduler's predicted end times.
class Server:
    def __init__(self, capacity, first_job):
        self.capacity   = capacity       # [1.0, 1.0, 1.0, 1.0]
        self.start_time = first_job.start
        self.is_active  = True
        self.cur_jobs   = SortedDict()   # true end time -> capacity held by jobs ending then
        self.running    = []             # (true end, believed end) of jobs running now
        self.placed_jobs = []            # every job ever placed here (kept for auditing)
        self.cur_size   = [0.0] * len(capacity)  # total resource usage now
        self._place_job(first_job)

    def _place_job(self, job):
        # if two jobs share the same true end time, aggregate their sizes
        if job.end in self.cur_jobs:
            for i in range(len(self.cur_size)):
                self.cur_jobs[job.end][i] += job.size[i]
        else:
            self.cur_jobs[job.end] = list(job.size)
        for i in range(len(self.cur_size)):
            self.cur_size[i] += job.size[i]
        self.running.append((job.end, job.believed_end))
        self.placed_jobs.append(job)

    def finishing_time(self):
        # TRUE time at which the last running job ends
        if not self.cur_jobs:
            return self.start_time
        return self.cur_jobs.keys()[-1]

    def believed_finishing_time(self):
        # Time at which the scheduler THINKS the server finishes (used for ranking only)
        if not self.running:
            return self.start_time
        return max(believed for _, believed in self.running)

    def remaining_capacity(self):
        return [self.capacity[i] - self.cur_size[i]
                for i in range(len(self.capacity))]

    def has_capacity(self, job):
        remaining = self.remaining_capacity()
        return all(job.size[i] <= remaining[i]
                   for i in range(len(self.capacity)))

    def duration(self):
        return self.finishing_time() - self.start_time

    def advance_time(self, to_time):
        # remove all jobs that really finished by to_time
        if not self.is_active:
            return False
        finished_keys = [k for k in self.cur_jobs.keys() if k <= to_time]
        for k in finished_keys:
            for i in range(len(self.cur_size)):
                self.cur_size[i] -= self.cur_jobs[k][i]
            del self.cur_jobs[k]
        self.running = [r for r in self.running if r[0] > to_time]
        if not self.cur_jobs:
            self.is_active = False
        return self.is_active

    def add_job(self, job):
        # advance time first to remove finished jobs, then check capacity
        self.advance_time(job.start)
        if not self.is_active:
            return False
        if self.has_capacity(job):
            self._place_job(job)
            return True
        return False

    def can_fit(self, job):
        # Like add_job, but only checks: nothing is placed.
        self.advance_time(job.start)
        return self.is_active and self.has_capacity(job)

    def is_active_at(self, time):
        if not self.is_active:
            return False
        return self.finishing_time() > time

    def __repr__(self):
        return (f"Server(start={self.start_time:.4f}, "
                f"finish={self.finishing_time():.4f}, "
                f"active={self.is_active}, "
                f"cur_size={[round(s, 3) for s in self.cur_size]})")