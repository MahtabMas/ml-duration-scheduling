from scheduler.greedy import Greedy
from scheduler.job import Job


# Greedy that ranks servers using PREDICTED job durations.
#
# predict_duration is a function: it takes a job and returns the predicted duration in days.
# It must only use what is known when the job arrives (start time and size),
# never job.duration. Capacity and cost always follow the TRUE durations (see Server).
class PredictiveGreedy(Greedy):
    def __init__(self, capacity, predict_duration):
        super().__init__(capacity)
        self.predict_duration = predict_duration
        self.true_durations = []       # kept to measure prediction quality
        self.predicted_durations = []

    def add_job(self, job):
        predicted = max(self.predict_duration(job), 1e-6)
        self.true_durations.append(job.duration)
        self.predicted_durations.append(predicted)
        # Same job, but now carrying the prediction. Job.end stays the TRUE end.
        super().add_job(Job(job.job_id, job.start, job.duration, job.size, predicted))