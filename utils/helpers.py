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


def ensure_directories(config: AppConfig) -> None:
    """Create output directories required by training/evaluation."""
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)
    Path(config.models_dir).mkdir(parents=True, exist_ok=True)
