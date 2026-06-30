"""Gymnasium environment for cloud resource allocation."""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.generator import JobGenerator
from environment.job import Job
from environment.server import Server
from utils.helpers import AppConfig


class CloudResourceEnv(gym.Env[np.ndarray, int]):
    """Cloud scheduling environment with discrete server-assignment actions."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.servers = [
            Server(
                id=server_id,
                cpu_capacity=config.cpu_capacity,
                memory_capacity=config.memory_capacity,
            )
            for server_id in range(config.servers)
        ]
        self.job_generator = JobGenerator(
            min_cpu=config.min_job_cpu,
            max_cpu=config.max_job_cpu,
            min_memory=config.min_job_memory,
            max_memory=config.max_job_memory,
            min_duration=config.min_job_duration,
            max_duration=config.max_job_duration,
            seed=config.seed,
        )

        observation_size = config.servers * 3 + 3
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(observation_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(config.servers)

        self.current_time = 0
        self.current_job: Job | None = None
        self.completed_jobs: list[Job] = []
        self.rejected_jobs = 0
        self.accepted_jobs = 0
        self.episode_reward = 0.0
        self.last_step_info: dict[str, Any] = {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset simulator state and return the initial observation."""
        super().reset(seed=seed)
        actual_seed = self.config.seed if seed is None else seed
        self.job_generator.reset(actual_seed)

        for server in self.servers:
            server.reset()

        self.current_time = 0
        self.current_job = self.job_generator.generate(self.current_time)
        self.completed_jobs = []
        self.rejected_jobs = 0
        self.accepted_jobs = 0
        self.episode_reward = 0.0
        self.last_step_info = {}
        return self._get_observation(), self._get_info()

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Assign the current job, advance time, and compute reward."""
        if self.current_job is None:
            raise RuntimeError("Environment must be reset before stepping.")

        selected_server = self.servers[int(action)]
        scheduled = selected_server.accept_job(self.current_job, self.current_time)
        completed_this_step: list[Job] = []
        reward = 0.0

        if scheduled:
            self.accepted_jobs += 1
            reward += 5.0
        else:
            self.rejected_jobs += 1
            selected_server.queue.append(self.current_job)
            reward -= 10.0

        self.current_time += 1
        for server in self.servers:
            completed_this_step.extend(server.step(self.current_time))

        self.completed_jobs.extend(completed_this_step)
        reward += 10.0 * len(completed_this_step)

        if self._is_balanced():
            reward += 2.0

        if self._has_overloaded_server():
            reward -= 15.0

        if any(len(server.queue) > 0 for server in self.servers):
            reward -= 5.0

        terminated = self.current_time >= self.config.episode_length
        truncated = False
        self.current_job = (
            None
            if terminated
            else self.job_generator.generate(self.current_time)
        )
        self.episode_reward += reward
        self.last_step_info = {
            "scheduled": scheduled,
            "completed_this_step": len(completed_this_step),
            "reward": reward,
        }
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def render(self) -> None:
        """Print a compact summary of the current cluster state."""
        utilization = [
            f"S{server.id}: CPU {server.cpu_utilization:.2f}, "
            f"MEM {server.memory_utilization:.2f}, Q {len(server.queue)}"
            for server in self.servers
        ]
        print(f"t={self.current_time} | " + " | ".join(utilization))

    def _get_observation(self) -> np.ndarray:
        server_state: list[float] = []
        for server in self.servers:
            server_state.extend(
                [
                    server.cpu_utilization,
                    server.memory_utilization,
                    min(float(len(server.queue)) / 10.0, 1.0),
                ]
            )

        if self.current_job is None:
            job_state = [0.0, 0.0, 0.0]
        else:
            job_state = [
                self.current_job.cpu_required / self.config.cpu_capacity,
                self.current_job.memory_required / self.config.memory_capacity,
                self.current_job.duration / self.config.max_job_duration,
            ]

        return np.asarray(server_state + job_state, dtype=np.float32)

    def _get_info(self) -> dict[str, Any]:
        avg_cpu = float(np.mean([server.cpu_utilization for server in self.servers]))
        avg_memory = float(
            np.mean([server.memory_utilization for server in self.servers])
        )
        avg_queue = float(np.mean([len(server.queue) for server in self.servers]))
        response_times = [
            job.response_time for job in self.completed_jobs if job.response_time is not None
        ]
        avg_response_time = float(np.mean(response_times)) if response_times else 0.0

        return {
            "time": self.current_time,
            "accepted_jobs": self.accepted_jobs,
            "completed_jobs": len(self.completed_jobs),
            "rejected_jobs": self.rejected_jobs,
            "average_cpu_utilization": avg_cpu,
            "average_memory_utilization": avg_memory,
            "average_queue_length": avg_queue,
            "average_response_time": avg_response_time,
            "episode_reward": self.episode_reward,
            **self.last_step_info,
        }

    def _is_balanced(self) -> bool:
        loads = np.asarray([server.load_score for server in self.servers])
        return bool(loads.max() - loads.min() <= 0.25)

    def _has_overloaded_server(self) -> bool:
        return any(
            server.cpu_utilization >= 0.95 or server.memory_utilization >= 0.95
            for server in self.servers
        )
