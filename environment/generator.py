"""Random job generator for the simulator."""

import numpy as np

from environment.job import Job


class JobGenerator:
    """Generate one random job per simulation step."""

    def __init__(
        self,
        min_cpu: float,
        max_cpu: float,
        min_memory: float,
        max_memory: float,
        min_duration: int,
        max_duration: int,
        seed: int | None = None,
    ) -> None:
        self.min_cpu = min_cpu
        self.max_cpu = max_cpu
        self.min_memory = min_memory
        self.max_memory = max_memory
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.rng = np.random.default_rng(seed)
        self.next_job_id = 0

    def reset(self, seed: int | None = None) -> None:
        """Reset sequence state and optionally reseed randomness."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.next_job_id = 0

    def generate(self, arrival_time: int) -> Job:
        """Create a new job for the current simulation time."""
        job = Job(
            job_id=self.next_job_id,
            cpu_required=float(self.rng.uniform(self.min_cpu, self.max_cpu)),
            memory_required=float(
                self.rng.uniform(self.min_memory, self.max_memory)
            ),
            duration=int(
                self.rng.integers(self.min_duration, self.max_duration + 1)
            ),
            arrival_time=arrival_time,
        )
        self.next_job_id += 1
        return job
