"""Server model for the cloud resource allocation simulator."""

from dataclasses import dataclass, field

from environment.job import Job


@dataclass(slots=True)
class Server:
    """A simple server with finite CPU and memory capacity."""

    id: int | str
    capacity: dict[str, float]
    used: dict[str, float] = field(default_factory=lambda: {"cpu": 0.0, "memory": 0.0})
    queue: list[Job] = field(default_factory=list)
    running_jobs: list[Job] = field(default_factory=list)

    def __post_init__(self):
        # Initialize used with keys from capacity if not already present
        for key in self.capacity:
            if key not in self.used:
                self.used[key] = 0.0

    def reset(self) -> None:
        """Clear all usage and queued/running jobs."""
        for key in self.used:
            self.used[key] = 0.0
        self.queue.clear()
        self.running_jobs.clear()

    @property
    def cpu_utilization(self) -> float:
        """Return CPU utilization in [0, 1]."""
        return self.used.get("cpu", 0.0) / self.capacity.get("cpu", 1.0)

    @property
    def memory_utilization(self) -> float:
        """Return memory utilization in [0, 1]."""
        return self.used.get("memory", 0.0) / self.capacity.get("memory", 1.0)

    @property
    def load_score(self) -> float:
        """Combined load score used by heuristic schedulers."""
        return (
            self.cpu_utilization
            + self.memory_utilization
            + min(float(len(self.queue)) / 10.0, 1.0)
        ) / 3.0

    def can_run(self, job: Job) -> bool:
        """Return True when the job fits on currently available resources."""
        return (
            self.used.get("cpu", 0.0) + job.cpu_required <= self.capacity.get("cpu", 0.0)
            and self.used.get("memory", 0.0) + job.memory_required <= self.capacity.get("memory", 0.0)
        )

    def accept_job(self, job: Job, current_time: int) -> bool:
        """Start a job immediately when capacity is available."""
        if not self.can_run(job):
            return False

        job.start_time = current_time
        self.used["cpu"] = self.used.get("cpu", 0.0) + job.cpu_required
        self.used["memory"] = self.used.get("memory", 0.0) + job.memory_required
        self.running_jobs.append(job)
        return True

    def step(self, current_time: int) -> list[Job]:
        """Advance running jobs by one tick and release completed jobs."""
        completed_jobs: list[Job] = []
        still_running: list[Job] = []

        for job in self.running_jobs:
            job.remaining_time -= 1
            if job.remaining_time <= 0:
                job.completion_time = current_time
                self.used["cpu"] -= job.cpu_required
                self.used["memory"] -= job.memory_required
                completed_jobs.append(job)
            else:
                still_running.append(job)

        self.running_jobs = still_running
        return completed_jobs
