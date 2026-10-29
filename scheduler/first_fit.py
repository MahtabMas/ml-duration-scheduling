from scheduler.job import Job
from scheduler.server import Server


# Non-clairvoyant algorithm: scans servers in opening order, places job in the first server that fits.
class FirstFit:
    def __init__(self, capacity):
        self.capacity    = capacity
        self.all_servers = []    # all servers ever opened
        self.cur_time    = 0.0

    def add_job(self, job):
        assert job.start >= self.cur_time, "Jobs must arrive in non-decreasing order of start time"
        self.cur_time = job.start

        placed = False
        for server in self.all_servers:
            if not server.is_active_at(job.start):
                continue
            if server.add_job(job):
                placed = True
                break

        if not placed:
            new_server = Server(capacity=self.capacity, first_job=job)
            self.all_servers.append(new_server)

    def cost(self):
        return sum(s.duration() for s in self.all_servers)

    def num_servers(self):
        return len(self.all_servers)

    def __repr__(self):
        return (f"FirstFit("
                f"servers={self.num_servers()}, "
                f"cost={self.cost():.4f})")