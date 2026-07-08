"""Plot generation for training and scheduler evaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_reward(log_dir: str | Path, results_dir: str | Path) -> None:
    """Plot monitor rewards when Stable-Baselines3 logs are available."""
    log_path = Path(log_dir) / "monitor.csv"
    if not log_path.exists():
        return

    rewards = pd.read_csv(log_path, skiprows=1)
    if rewards.empty or "r" not in rewards:
        return

    output = Path(results_dir) / "training_reward.png"
    plt.figure(figsize=(9, 5))
    plt.plot(rewards["r"], label="Episode reward")
    plt.title("Training Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_evaluation_metrics(summary: pd.DataFrame, results_dir: str | Path) -> None:
    """Save comparison plots for all required evaluation metrics."""
    metric_titles = {
        "average_response_time": "Response Time Comparison",
        "average_queue_length": "Queue Length Comparison",
        "average_cpu_utilization": "CPU Utilization",
        "average_memory_utilization": "Memory Utilization",
        "episode_reward": "Scheduler Comparison",
        "sla_violation_rate": "SLA Violation Rate",
        "priority_weighted_completion_rate": "Priority-Weighted Completion Rate",
        "max_wait_gold": "Max Wait (Gold)",
        "max_wait_silver": "Max Wait (Silver)",
        "max_wait_bronze": "Max Wait (Bronze)",
    }

    for metric, title in metric_titles.items():
        if metric not in summary.columns:
            continue
        output = Path(results_dir) / f"{metric}.png"
        plt.figure(figsize=(9, 5))
        plt.bar(summary["scheduler"], summary[metric])
        plt.title(title)
        plt.xlabel("Scheduler")
        plt.ylabel(metric.replace("_", " ").title())
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(output)
        plt.close()
