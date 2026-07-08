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
    "sla_violation_rate",
    "priority_weighted_completion_rate",
    "max_wait_gold",
    "max_wait_silver",
    "max_wait_bronze",
    "mean_completion_reward",
    "mean_sla_violation_reward",
    "mean_rejection_reward",
    "mean_overload_reward",
    "mean_queue_reward",
    "mean_balance_reward",
    "mean_starvation_bonus_reward"
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
        "average_response_time": float(info.get("average_response_time", 0.0)),
        "average_queue_length": float(info.get("average_queue_length", 0.0)),
        "successful_jobs": int(info.get("completed_jobs", 0)),
        "rejected_jobs": int(info.get("rejected_jobs", 0)),
        "average_cpu_utilization": float(info.get("average_cpu_utilization", 0.0)),
        "average_memory_utilization": float(info.get("average_memory_utilization", 0.0)),
        "episode_reward": float(info.get("episode_reward", 0.0)),
        "sla_violation_rate": float(info.get("sla_violation_rate", 0.0)),
        "priority_weighted_completion_rate": float(info.get("priority_weighted_completion_rate", 0.0)),
        "max_wait_gold": int(info.get("max_wait_by_tier", {}).get(1, 0)),
        "max_wait_silver": int(info.get("max_wait_by_tier", {}).get(2, 0)),
        "max_wait_bronze": int(info.get("max_wait_by_tier", {}).get(3, 0)),
        "mean_completion_reward": float(info.get("mean_completion", 0.0)),
        "mean_sla_violation_reward": float(info.get("mean_sla_violation", 0.0)),
        "mean_rejection_reward": float(info.get("mean_rejection", 0.0)),
        "mean_overload_reward": float(info.get("mean_overload", 0.0)),
        "mean_queue_reward": float(info.get("mean_queue", 0.0)),
        "mean_balance_reward": float(info.get("mean_balance", 0.0)),
        "mean_starvation_bonus_reward": float(info.get("mean_starvation_bonus", 0.0)),
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
            sla_violation_rate=("sla_violation_rate", "mean"),
            priority_weighted_completion_rate=("priority_weighted_completion_rate", "mean"),
            max_wait_gold=("max_wait_gold", "mean"),
            max_wait_silver=("max_wait_silver", "mean"),
            max_wait_bronze=("max_wait_bronze", "mean"),
            mean_completion_reward=("mean_completion_reward", "mean"),
            mean_sla_violation_reward=("mean_sla_violation_reward", "mean"),
            mean_rejection_reward=("mean_rejection_reward", "mean"),
            mean_overload_reward=("mean_overload_reward", "mean"),
            mean_queue_reward=("mean_queue_reward", "mean"),
            mean_balance_reward=("mean_balance_reward", "mean"),
            mean_starvation_bonus_reward=("mean_starvation_bonus_reward", "mean"),
        )
        .sort_values("episode_reward", ascending=False)
    )
