"""Live simulation runner used by REST and WebSocket endpoints."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend.api.schemas import (
    ChartPoint,
    DecisionState,
    JobState,
    MetricsState,
    ServerState,
    SimulationState,
    TrainingState,
)
from environment.cloud_env import CloudResourceEnv
from schedulers.least_loaded import LeastLoadedScheduler
from schedulers.random_scheduler import RandomScheduler
from schedulers.round_robin import RoundRobinScheduler
from utils.helpers import AppConfig

if TYPE_CHECKING:
    from stable_baselines3 import PPO


class LiveSimulation:
    """Manage one live CloudPilot simulation loop."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.env = CloudResourceEnv(config)
        self.observation, self.info = self.env.reset(seed=config.seed)
        self.status = "Paused"
        self.scheduler_name = "Least Loaded"
        self.speed = config.simulation_speed
        self.episode = 1
        self.history: list[ChartPoint] = []
        self.heatmap: list[list[float]] = [[] for _ in range(config.servers)]
        self.decision = DecisionState(scheduler=self.scheduler_name)
        self.training = TrainingState()
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._round_robin = RoundRobinScheduler()
        self._random = RandomScheduler(config.seed)
        self._ppo_model: PPO | None = self._load_ppo_model()

    async def start(self) -> None:
        """Start the background simulation loop."""
        self.status = "Running"
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def pause(self) -> None:
        """Pause simulation ticks."""
        self.status = "Paused"

    async def reset(self) -> None:
        """Reset simulation state."""
        async with self._lock:
            self.env = CloudResourceEnv(self.config)
            self.observation, self.info = self.env.reset(seed=self.config.seed)
            self.episode = 1
            self.history.clear()
            self.heatmap = [[] for _ in range(self.config.servers)]
            self.decision = DecisionState(scheduler=self.scheduler_name)
            self._round_robin.reset()

    async def configure(
        self,
        scheduler: str | None = None,
        speed: float | None = None,
    ) -> None:
        """Update active scheduler or tick speed."""
        async with self._lock:
            if scheduler is not None:
                self.scheduler_name = scheduler
                if scheduler == "Round Robin":
                    self._round_robin.reset()
            if speed is not None:
                self.speed = max(0.25, min(speed, 20.0))
            self.decision.scheduler = self.scheduler_name

    async def snapshot(self) -> SimulationState:
        """Return the current dashboard state."""
        async with self._lock:
            return self._snapshot_unlocked()

    async def _run(self) -> None:
        while True:
            if self.status == "Running":
                async with self._lock:
                    self._tick_unlocked()
            await asyncio.sleep(1.0 / self.speed)

    def _tick_unlocked(self) -> None:
        incoming = self.env.current_job
        action = self._select_action()
        self.observation, reward, terminated, truncated, self.info = self.env.step(
            action
        )
        self.decision = DecisionState(
            job_id=incoming.job_id if incoming else None,
            scheduler=self.scheduler_name,
            assigned_server=action,
            accepted=bool(self.info.get("scheduled", False)),
            reward=float(reward),
        )
        self._record_history(reward)

        if terminated or truncated:
            self.episode += 1
            self.observation, self.info = self.env.reset(
                seed=self.config.seed + self.episode
            )
            self._round_robin.reset()

    def _select_action(self) -> int:
        if self.scheduler_name == "Random":
            return self._random.select_action(self.env)
        if self.scheduler_name == "Round Robin":
            return self._round_robin.select_action(self.env)
        if self.scheduler_name == "PPO":
            if self._ppo_model is None:
                return LeastLoadedScheduler().select_action(self.env)
            action, _ = self._ppo_model.predict(self.observation, deterministic=True)
            return int(action)
        return LeastLoadedScheduler().select_action(self.env)

    def _record_history(self, reward: float) -> None:
        cpu = [server.cpu_utilization for server in self.env.servers]
        memory = [server.memory_utilization for server in self.env.servers]
        for index, value in enumerate(cpu):
            self.heatmap[index].append(value)
            self.heatmap[index] = self.heatmap[index][-60:]

        point = ChartPoint(
            step=self.env.current_time,
            episode=self.episode,
            reward=float(reward),
            average_queue_length=float(self.info["average_queue_length"]),
            jobs_completed=int(self.info["completed_jobs"]),
            jobs_rejected=int(self.info["rejected_jobs"]),
            cpu=cpu,
            memory=memory,
        )
        self.history.append(point)
        self.history = self.history[-80:]

    def _snapshot_unlocked(self) -> SimulationState:
        return SimulationState(
            status=self.status,
            scheduler=self.scheduler_name,
            episode=self.episode,
            step=self.env.current_time,
            speed=self.speed,
            incoming_job=self._job_state(self.env.current_job),
            servers=[self._server_state(server) for server in self.env.servers],
            decision=self.decision,
            metrics=self._metrics_state(),
            history=self.history,
            heatmap=self.heatmap,
        )

    def _server_state(self, server: Any) -> ServerState:
        if server.cpu_utilization >= 0.9 or server.memory_utilization >= 0.9:
            status = "Hot"
        elif server.running_jobs:
            status = "Active"
        else:
            status = "Idle"

        return ServerState(
            id=server.id,
            name=f"Server {chr(65 + server.id)}",
            cpu_utilization=server.cpu_utilization,
            memory_utilization=server.memory_utilization,
            queue_length=len(server.queue),
            running_jobs=[self._job_state(job) for job in server.running_jobs],
            status=status,
        )

    def _metrics_state(self) -> MetricsState:
        return MetricsState(
            jobs_processed=int(self.info.get("accepted_jobs", 0) + self.info.get("rejected_jobs", 0)),
            jobs_completed=int(self.info.get("completed_jobs", 0)),
            jobs_rejected=int(self.info.get("rejected_jobs", 0)),
            average_response_time=float(self.info.get("average_response_time", 0.0)),
            average_queue_length=float(self.info.get("average_queue_length", 0.0)),
            average_cpu_utilization=float(self.info.get("average_cpu_utilization", 0.0)),
            average_memory_utilization=float(self.info.get("average_memory_utilization", 0.0)),
            current_reward=float(self.info.get("reward", 0.0)),
            episode_reward=float(self.info.get("episode_reward", 0.0)),
            sla_violation_rate=float(self.info.get("sla_violation_rate", 0.0)),
            priority_weighted_completion_rate=float(self.info.get("priority_weighted_completion_rate", 0.0)),
        )

    @staticmethod
    def _job_state(job: Any | None) -> JobState | None:
        if job is None:
            return None
        return JobState(
            job_id=job.job_id,
            cpu_required=job.cpu_required,
            memory_required=job.memory_required,
            runtime=job.duration,
            arrival_time=job.arrival_time,
            priority_tier=job.priority_tier.value if hasattr(job, "priority_tier") else "bronze",
            deadline_ticks=job.deadline_ticks if hasattr(job, "deadline_ticks") else 0,
            remaining_time=getattr(job, "remaining_time", None),
        )

    @staticmethod
    def _load_ppo_model() -> "PPO | None":
        model_path = Path("models/ppo_final.zip")
        if not model_path.exists():
            return None
        from stable_baselines3 import PPO

        return PPO.load(model_path)
