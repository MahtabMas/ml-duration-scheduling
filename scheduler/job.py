# Represents a single job (VM request) in the cloud scheduling simulation.
class Job:
    def __init__(self, job_id, start, duration, size, predicted_duration=None):
        self.job_id = job_id        # unique identifier (vmId from dataset)
        self.start = start          # arrival time in days
        self.size = size            # [core, memory, ssd, nic] each in [0, 1]

        # What really happens: the job holds its resources until this time.
        self.duration = duration    # TRUE duration
        self.end = start + duration # TRUE end time

        # What the scheduler believes. It is only used to RANK servers.
        # With no prediction (First Fit, Greedy) the scheduler knows the truth.
        if predicted_duration is None:
            predicted_duration = duration
        self.predicted_duration = predicted_duration
        self.believed_end = start + predicted_duration

    def __repr__(self):
        return (f"Job(id={self.job_id}, "
                f"start={self.start:.4f}, "
                f"end={self.end:.4f}, "
                f"size={[round(s, 3) for s in self.size]})")