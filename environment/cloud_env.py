"""Gymnasium environment for cloud resource allocation."""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.generator import JobGenerator
from environment.job import Job
from environment.server import Server
from utils.helpers import AppConfig, AppConfigV1_5
from environment.priority import PriorityTier


class TickEvents:
    def __init__(self):
        self.completions: list[Job] = []
        self.sla_violations: list[Job] = []
        self.rejections: list[Job] = []


class CloudResourceEnv(gym.Env[np.ndarray, int]):
    """Cloud scheduling environment with discrete server-assignment actions."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: AppConfigV1_5) -> None:
        super().__init__()
        self.config = config
        self.servers = [
            Server(
                id=s_config.id,
                capacity={"cpu": s_config.cpu_capacity, "memory": s_config.mem_capacity},
            )
            for s_config in config.cluster.servers
        ]
        self.job_generator = JobGenerator(
            min_cpu=config.min_job_cpu,
            max_cpu=config.max_job_cpu,
            min_memory=config.min_job_memory,
            max_memory=config.max_job_memory,
            min_duration=config.min_job_duration,
            max_duration=config.max_job_duration,
            seed=config.seed,
            priority_distribution=config.priority.distribution,
            deadline_ticks_range=config.priority.deadline_ticks
        )

        observation_size = config.servers * 4 + 1 + 3 + 3 + 1
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(observation_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(config.servers)

        self.current_time = 0
        self.global_queue: list[Job] = []
        self.current_job: Job | None = None
        
        self.completed_jobs: list[Job] = []
        self.rejected_jobs_count = 0
        self.accepted_jobs_count = 0
        self.episode_reward = 0.0
        self.last_step_info: dict[str, Any] = {}
        
        self.sla_violations_total = 0
        self.priority_weights_total = 0.0
        self.priority_weights_completed = 0.0
        self.max_wait_by_tier = {
            PriorityTier.GOLD.value: 0,
            PriorityTier.SILVER.value: 0,
            PriorityTier.BRONZE.value: 0,
        }
        self.avg_reward_components = {}

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
        self.global_queue.clear()
        self.current_job = self.job_generator.generate(self.current_time)
        self.completed_jobs = []
        self.rejected_jobs_count = 0
        self.accepted_jobs_count = 0
        self.episode_reward = 0.0
        self.last_step_info = {}
        
        self.sla_violations_total = 0
        self.priority_weights_total = 0.0
        self.priority_weights_completed = 0.0
        self.max_wait_by_tier = {
            PriorityTier.GOLD.value: 0,
            PriorityTier.SILVER.value: 0,
            PriorityTier.BRONZE.value: 0,
        }
        self.avg_reward_components = {}
        
        return self._get_observation(), self._get_info()

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Assign the current job, advance time, and compute reward."""
        if self.current_job is None:
            raise RuntimeError("Environment must be reset before stepping.")

        tick_events = TickEvents()

        # Step 0: Increment ticks_waited for queued jobs and update max wait
        for job in self.global_queue:
            job.ticks_waited += 1
            tier_val = job.priority_tier.value
            if job.ticks_waited > self.max_wait_by_tier[tier_val]:
                self.max_wait_by_tier[tier_val] = job.ticks_waited

        # Step 1: Check for SLA expiry
        still_in_queue = []
        for job in self.global_queue:
            if job.ticks_waited > job.deadline_ticks:
                tick_events.sla_violations.append(job)
                tick_events.rejections.append(job)
                self.rejected_jobs_count += 1
                self.sla_violations_total += 1
            else:
                still_in_queue.append(job)
        self.global_queue = still_in_queue

        # Step 2: Attempt to place remaining queued jobs in priority-then-FIFO order
        retry_order = getattr(self.config.priority, "queue_retry_order", "priority_fifo")
        if retry_order == "priority_fifo":
            self.global_queue.sort(
                key=lambda j: (
                    j.priority_tier == PriorityTier.BRONZE,
                    j.priority_tier == PriorityTier.SILVER,
                    j.arrival_time
                )
            )
        else:
            self.global_queue.sort(key=lambda j: j.arrival_time)
        
        remaining_queue = []
        for job in self.global_queue:
            placed = False
            for server in self.servers:
                if server.accept_job(job, self.current_time):
                    placed = True
                    self.accepted_jobs_count += 1
                    break
            if not placed:
                remaining_queue.append(job)
        self.global_queue = remaining_queue

        # Step 3: Process the new incoming job using the agent's action
        if isinstance(action, str):
            server_idx = next((i for i, s in enumerate(self.servers) if s.id == action), 0)
            selected_server = self.servers[server_idx]
        else:
            selected_server = self.servers[int(action)]
        
        scheduled = selected_server.accept_job(self.current_job, self.current_time)
        
        self.priority_weights_total += self.config.reward.tier_weights.get(self.current_job.priority_tier.value, 1.0)

        if scheduled:
            self.accepted_jobs_count += 1
        else:
            self.rejected_jobs_count += 1
            tick_events.rejections.append(self.current_job)
            self.global_queue.append(self.current_job)

        self.current_time += 1
        for server in self.servers:
            completed = server.step(self.current_time)
            tick_events.completions.extend(completed)

        self.completed_jobs.extend(tick_events.completions)
        for job in tick_events.completions:
            self.priority_weights_completed += self.config.reward.tier_weights.get(job.priority_tier.value, 1.0)

        # Step 4: Compute Reward
        reward, reward_components = self._compute_reward(tick_events)
        
        # Accumulate reward components for reporting average per episode
        for k, v in reward_components.items():
            self.avg_reward_components[k] = self.avg_reward_components.get(k, 0.0) + v

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
            "completed_this_step": len(tick_events.completions),
            "reward": reward,
            **reward_components
        }
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _compute_reward(self, tick_events: TickEvents) -> tuple[float, dict[str, float]]:
        cfg = self.config.reward
        r = 0.0
        components = {
            "completion": 0.0,
            "sla_violation": 0.0,
            "rejection": 0.0,
            "overload": 0.0,
            "queue": 0.0,
            "balance": 0.0,
            "starvation_bonus": 0.0
        }

        for job in tick_events.completions:
            w = cfg.tier_weights.get(job.priority_tier.value, 1.0)
            val = cfg.completion_base * w
            r += val
            components["completion"] += val

        for job in tick_events.sla_violations:
            w = cfg.tier_weights.get(job.priority_tier.value, 1.0)
            val = cfg.sla_violation_penalty * w
            r += val
            components["sla_violation"] += val

        for job in tick_events.rejections:
            w = cfg.tier_weights.get(job.priority_tier.value, 1.0)
            val = cfg.rejection_penalty_base * w
            r += val
            components["rejection"] += val

        total_overload_units = sum(
            max(0, s.used[res] - s.capacity[res]) for s in self.servers for res in s.capacity
        )
        val = cfg.overload_penalty_per_unit * total_overload_units
        r += val
        components["overload"] += val

        val = cfg.queue_penalty_per_job_per_tick * len(self.global_queue)
        r += val
        components["queue"] += val

        utils = [s.used[res] / s.capacity[res] for s in self.servers for res in s.capacity]
        std_dev = float(np.std(utils)) if len(utils) > 0 else 0.0
        val = cfg.balance_bonus_weight * (1 - std_dev)
        r += val
        components["balance"] += val

        for job in self.global_queue:
            if job.ticks_waited > cfg.starvation_guard.threshold_ticks:
                bonus = min(
                    cfg.starvation_guard.bonus_per_tick_over
                    * (job.ticks_waited - cfg.starvation_guard.threshold_ticks),
                    cfg.starvation_guard.cap,
                )
                r += bonus
                components["starvation_bonus"] += bonus

        return r, components

    def render(self) -> None:
        """Print a compact summary of the current cluster state."""
        utilization = [
            f"S{server.id}: CPU {server.cpu_utilization:.2f}, "
            f"MEM {server.memory_utilization:.2f}"
            for server in self.servers
        ]
        print(f"t={self.current_time} | Q={len(self.global_queue)} | " + " | ".join(utilization))

    def _get_observation(self) -> np.ndarray:
        norm = self.config.normalization
        cluster_cpu_cap = max(s.capacity["cpu"] for s in self.servers)
        cluster_mem_cap = max(s.capacity["memory"] for s in self.servers)
        max_deadline = max(max(v) for v in self.config.priority.deadline_ticks.values())

        server_state: list[float] = []
        for server in self.servers:
            server_state.extend([
                server.capacity["cpu"] / cluster_cpu_cap,
                server.capacity["memory"] / cluster_mem_cap,
                server.cpu_utilization,
                server.memory_utilization,
            ])

        queue_state = [min(float(len(self.global_queue)) / norm.max_queue_length, 1.0)]

        if self.current_job is None:
            job_state = [0.0, 0.0, 0.0]
            job_prio = [0.0, 0.0, 0.0]
            job_urgency = [0.0]
        else:
            job_state = [
                self.current_job.cpu_required / norm.max_job_cpu,
                self.current_job.memory_required / norm.max_job_mem,
                self.current_job.duration / norm.max_runtime,
            ]
            job_prio = [
                1.0 if self.current_job.priority_tier == PriorityTier.GOLD else 0.0,
                1.0 if self.current_job.priority_tier == PriorityTier.SILVER else 0.0,
                1.0 if self.current_job.priority_tier == PriorityTier.BRONZE else 0.0,
            ]
            job_urgency = [self.current_job.deadline_ticks / max_deadline]

        obs = server_state + queue_state + job_state + job_prio + job_urgency
        return np.asarray(obs, dtype=np.float32)

    def _get_info(self) -> dict[str, Any]:
        avg_cpu = float(np.mean([server.cpu_utilization for server in self.servers]))
        avg_memory = float(
            np.mean([server.memory_utilization for server in self.servers])
        )
        avg_queue = float(len(self.global_queue))
        response_times = [
            job.response_time for job in self.completed_jobs if job.response_time is not None
        ]
        avg_response_time = float(np.mean(response_times)) if response_times else 0.0

        total_jobs = self.accepted_jobs_count + self.rejected_jobs_count
        sla_violation_rate = self.sla_violations_total / max(1, total_jobs)
        pwc_rate = self.priority_weights_completed / max(1.0, self.priority_weights_total)
        
        # Calculate mean of reward components for the episode
        mean_components = {
            f"mean_{k}": v / max(1, self.current_time)
            for k, v in self.avg_reward_components.items()
        }

        return {
            "time": self.current_time,
            "accepted_jobs": self.accepted_jobs_count,
            "completed_jobs": len(self.completed_jobs),
            "rejected_jobs": self.rejected_jobs_count,
            "average_cpu_utilization": avg_cpu,
            "average_memory_utilization": avg_memory,
            "average_queue_length": avg_queue,
            "average_response_time": avg_response_time,
            "episode_reward": self.episode_reward,
            "sla_violations_total": self.sla_violations_total,
            "sla_violation_rate": sla_violation_rate,
            "priority_weighted_completion_rate": pwc_rate,
            "max_wait_by_tier": self.max_wait_by_tier,
            **self.last_step_info,
            **mean_components
        }
