"""Configuration and filesystem helpers."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    """Runtime configuration loaded from YAML."""

    servers: int = Field(default=4, ge=1)
    episode_length: int = Field(default=500, ge=1)
    cpu_capacity: float = Field(default=100.0, gt=0)
    memory_capacity: float = Field(default=32.0, gt=0)
    min_job_cpu: float = Field(default=10.0, ge=0)
    max_job_cpu: float = Field(default=60.0, gt=0)
    min_job_memory: float = Field(default=1.0, gt=0)
    max_job_memory: float = Field(default=8.0, gt=0)
    min_job_duration: int = Field(default=2, ge=1)
    max_job_duration: int = Field(default=20, ge=1)
    seed: int = 42
    training_timesteps: int = Field(default=100_000, ge=1)
    evaluation_episodes: int = Field(default=100, ge=1)
    simulation_speed: float = Field(default=2.0, gt=0)
    results_dir: str = "results"
    models_dir: str = "models"


def load_config(path: str | Path = "configs/config.yaml") -> AppConfig:
    """Load application configuration from a YAML file."""
    with Path(path).open("r", encoding="utf-8") as file:
        data: dict[str, Any] = yaml.safe_load(file) or {}
    return AppConfig(**data)


def ensure_directories(config: AppConfig | Any) -> None:
    """Create output directories required by training/evaluation."""
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)
    Path(config.models_dir).mkdir(parents=True, exist_ok=True)


class ServerConfig(BaseModel):
    id: str
    class_name: str = Field(alias="class")
    cpu_capacity: float
    mem_capacity: float

class ClusterConfig(BaseModel):
    servers: list[ServerConfig]

class PriorityConfig(BaseModel):
    distribution: dict[str, float]
    deadline_ticks: dict[str, list[int]]
    queue_retry_order: str

class StarvationGuardConfig(BaseModel):
    threshold_ticks: int
    bonus_per_tick_over: float
    cap: float

class RewardConfig(BaseModel):
    completion_base: float
    tier_weights: dict[str, float]
    sla_violation_penalty: float
    overload_penalty_per_unit: float
    rejection_penalty_base: float
    queue_penalty_per_job_per_tick: float
    balance_bonus_weight: float
    starvation_guard: StarvationGuardConfig

class NormalizationConfig(BaseModel):
    max_job_cpu: float
    max_job_mem: float
    max_runtime: float
    max_queue_length: float

class AppConfigV1_5(BaseModel):
    cluster: ClusterConfig
    priority: PriorityConfig
    reward: RewardConfig
    normalization: NormalizationConfig
    
    episode_length: int = 500
    seed: int = 42
    training_timesteps: int = 100_000
    evaluation_episodes: int = 100
    simulation_speed: float = 2.0
    results_dir: str = "results/v1.5"
    models_dir: str = "models"
    min_job_cpu: float = 1.0
    max_job_cpu: float = 8.0
    min_job_memory: float = 1.0
    max_job_memory: float = 32.0
    min_job_duration: int = 2
    max_job_duration: int = 50
    
    @property
    def servers(self) -> int:
        return len(self.cluster.servers)
    
    @property
    def cpu_capacity(self) -> float:
        return max(s.cpu_capacity for s in self.cluster.servers)
    
    @property
    def memory_capacity(self) -> float:
        return max(s.mem_capacity for s in self.cluster.servers)


def load_config_v1_5(path: str | Path = "configs/config_v1_5.yaml") -> AppConfigV1_5:
    """Load application configuration from a YAML file for v1.5."""
    with Path(path).open("r", encoding="utf-8") as file:
        data: dict[str, Any] = yaml.safe_load(file) or {}
    return AppConfigV1_5(**data)
