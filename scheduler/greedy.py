from scheduler.server import Server


# Greedy:
# among the active servers with enough room, pick the one the job would extend
# the least (extension = how far the job pushes the server's finishing time later).
# Ties go to the server opened first.
#
# It works with believed end times: with exact durations they are the true ends
# (the published Greedy); with predicted durations they are the predictions.
# Capacity and cost always follow the true end times (see Server).
class Greedy:
    def __init__(self, capacity):
        self.capacity    = capacity
        self.all_servers = []    # all servers ever opened, in opening order
        self.cur_time    = 0.0

    def add_job(self, job):
        assert job.start >= self.cur_time, "Jobs must arrive in non-decreasing order of start time"
        self.cur_time = job.start

        best_server = None
        best_extension = float("inf")
        for server in self.all_servers:
            if not server.is_active_at(job.start):
                continue
            extension = max(0.0, job.believed_end - server.believed_finishing_time())
            if extension < best_extension and server.can_fit(job):
                best_extension = extension
                best_server = server

        if best_server is not None:
            best_server.add_job(job)
        else:
            self.all_servers.append(Server(capacity=self.capacity, first_job=job))

    def cost(self):
        return sum(s.duration() for s in self.all_servers)

    def num_servers(self):
        return len(self.all_servers)

    def __repr__(self):
        return (f"Greedy("
                f"servers={self.num_servers()}, "
                f"cost={self.cost():.4f})")