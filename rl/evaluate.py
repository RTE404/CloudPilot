"""Evaluation helpers for PPO and baseline schedulers."""

from collections.abc import Protocol
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table
from stable_baselines3 import PPO

from environment.cloud_env import CloudResourceEnv
from metrics.metrics import aggregate_metrics, summarize_episode
from schedulers.least_loaded import LeastLoadedScheduler
from schedulers.random_scheduler import RandomScheduler
from schedulers.round_robin import RoundRobinScheduler
from utils.helpers import AppConfig
from visualization.plots import plot_evaluation_metrics


class Scheduler(Protocol):
    """Protocol implemented by baseline schedulers."""

    name: str

    def select_action(self, env: CloudResourceEnv) -> int:
        """Return a valid action for the current environment."""


def evaluate_all(config: AppConfig, model_path: str | Path, console: Console) -> pd.DataFrame:
    """Evaluate baselines and PPO, save CSVs and plots, and return a summary."""
    rows: list[dict[str, float | int | str]] = []
    schedulers: list[Scheduler] = [
        RandomScheduler(config.seed),
        RoundRobinScheduler(),
        LeastLoadedScheduler(),
    ]

    for scheduler in schedulers:
        rows.extend(_evaluate_scheduler(config, scheduler))

    ppo_model = PPO.load(model_path)
    rows.extend(_evaluate_ppo(config, ppo_model))

    episodes = pd.DataFrame(rows)
    summary = aggregate_metrics(rows)
    results_dir = Path(config.results_dir)
    episodes.to_csv(results_dir / "evaluation_episodes.csv", index=False)
    summary.to_csv(results_dir / "evaluation_summary.csv", index=False)
    plot_evaluation_metrics(summary, results_dir)
    _print_summary(summary, console)
    return summary


def _evaluate_scheduler(
    config: AppConfig,
    scheduler: Scheduler,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for episode in range(config.evaluation_episodes):
        if hasattr(scheduler, "reset"):
            scheduler.reset()

        env = CloudResourceEnv(config)
        _, info = env.reset(seed=config.seed + episode)
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = scheduler.select_action(env)
            _, _, terminated, truncated, info = env.step(action)

        rows.append(summarize_episode(scheduler.name, episode, info))
    return rows


def _evaluate_ppo(
    config: AppConfig,
    model: PPO,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for episode in range(config.evaluation_episodes):
        env = CloudResourceEnv(config)
        observation, info = env.reset(seed=config.seed + episode)
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(int(action))

        rows.append(summarize_episode("PPO Agent", episode, info))
    return rows


def _print_summary(summary: pd.DataFrame, console: Console) -> None:
    table = Table(title="Evaluation Summary")
    for column in summary.columns:
        table.add_column(column.replace("_", " ").title())

    for _, row in summary.iterrows():
        table.add_row(
            str(row["scheduler"]),
            f"{row['average_response_time']:.2f}",
            f"{row['average_queue_length']:.2f}",
            f"{row['successful_jobs']:.2f}",
            f"{row['rejected_jobs']:.2f}",
            f"{row['average_cpu_utilization']:.3f}",
            f"{row['average_memory_utilization']:.3f}",
            f"{row['episode_reward']:.2f}",
        )
    console.print(table)
