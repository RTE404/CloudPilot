"""Job model used by the cloud simulator."""

from dataclasses import dataclass, field

from environment.priority import PriorityTier


@dataclass(slots=True)
class Job:
    """A simulated cloud job with static resource requirements."""

    job_id: int
    cpu_required: float
    memory_required: float
    duration: int
    arrival_time: int
    priority_tier: PriorityTier
    deadline_ticks: int
    ticks_waited: int = 0
    remaining_time: int = field(init=False)
    start_time: int | None = None
    completion_time: int | None = None

    def __post_init__(self) -> None:
        self.remaining_time = self.duration

    @property
    def response_time(self) -> int | None:
        """Return completion time minus arrival time once complete."""
        if self.completion_time is None:
            return None
        return self.completion_time - self.arrival_time
