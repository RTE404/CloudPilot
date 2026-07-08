"""Random job generator for the simulator."""

import numpy as np

from environment.job import Job
from environment.priority import PriorityTier


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
        priority_distribution: dict[str, float] | None = None,
        deadline_ticks_range: dict[str, list[int]] | None = None,
    ) -> None:
        self.min_cpu = min_cpu
        self.max_cpu = max_cpu
        self.min_memory = min_memory
        self.max_memory = max_memory
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.priority_distribution = priority_distribution
        self.deadline_ticks_range = deadline_ticks_range
        
        self.rng = np.random.default_rng(seed)
        self.next_job_id = 0
        
        if self.priority_distribution:
            self._priority_choices = [PriorityTier(k) for k in self.priority_distribution.keys()]
            self._priority_probs = list(self.priority_distribution.values())
            # Ensure probabilities sum to 1.0 (with a small tolerance)
            assert np.isclose(sum(self._priority_probs), 1.0), "Priority distribution must sum to 1.0"
        else:
            self._priority_choices = [PriorityTier.BRONZE]
            self._priority_probs = [1.0]

    def reset(self, seed: int | None = None) -> None:
        """Reset sequence state and optionally reseed randomness."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.next_job_id = 0

    def generate(self, arrival_time: int) -> Job:
        """Create a new job for the current simulation time."""
        priority_tier = self.rng.choice(self._priority_choices, p=self._priority_probs)
        
        if self.deadline_ticks_range and priority_tier.value in self.deadline_ticks_range:
            min_deadline, max_deadline = self.deadline_ticks_range[priority_tier.value]
            deadline_ticks = int(self.rng.integers(min_deadline, max_deadline + 1))
        else:
            deadline_ticks = 1000

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
            priority_tier=priority_tier,
            deadline_ticks=deadline_ticks
        )
        self.next_job_id += 1
        return job
