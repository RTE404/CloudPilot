"""Metric collection and aggregation utilities."""

from collections.abc import Iterable

import pandas as pd


EPISODE_METRIC_COLUMNS = [
    "scheduler",
    "episode",
    "average_response_time",
    "average_queue_length",
    "successful_jobs",
    "rejected_jobs",
    "average_cpu_utilization",
    "average_memory_utilization",
    "episode_reward",
]


def summarize_episode(
    scheduler: str,
    episode: int,
    info: dict,
) -> dict[str, float | int | str]:
    """Convert final environment info into one episode metrics row."""
    return {
        "scheduler": scheduler,
        "episode": episode,
        "average_response_time": float(info["average_response_time"]),
        "average_queue_length": float(info["average_queue_length"]),
        "successful_jobs": int(info["completed_jobs"]),
        "rejected_jobs": int(info["rejected_jobs"]),
        "average_cpu_utilization": float(info["average_cpu_utilization"]),
        "average_memory_utilization": float(info["average_memory_utilization"]),
        "episode_reward": float(info["episode_reward"]),
    }


def aggregate_metrics(rows: Iterable[dict[str, float | int | str]]) -> pd.DataFrame:
    """Aggregate per-episode rows by scheduler."""
    frame = pd.DataFrame(rows, columns=EPISODE_METRIC_COLUMNS)
    return (
        frame.groupby("scheduler", as_index=False)
        .agg(
            average_response_time=("average_response_time", "mean"),
            average_queue_length=("average_queue_length", "mean"),
            successful_jobs=("successful_jobs", "mean"),
            rejected_jobs=("rejected_jobs", "mean"),
            average_cpu_utilization=("average_cpu_utilization", "mean"),
            average_memory_utilization=("average_memory_utilization", "mean"),
            episode_reward=("episode_reward", "mean"),
        )
        .sort_values("episode_reward", ascending=False)
    )
