"""Training pipeline for the PPO scheduler."""

from pathlib import Path

from rich.console import Console
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from environment.cloud_env import CloudResourceEnv
from rl.model import build_ppo_model
from utils.helpers import AppConfigV1_5, load_config_v1_5
from visualization.plots import plot_training_reward


class TensorboardCallback(BaseCallback):
    """Log separate reward components to TensorBoard."""
    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "completion" in info:
                self.logger.record("reward_components/completion", info["completion"])
                self.logger.record("reward_components/sla_violation", info["sla_violation"])
                self.logger.record("reward_components/rejection", info["rejection"])
                self.logger.record("reward_components/overload", info["overload"])
                self.logger.record("reward_components/queue", info["queue"])
                self.logger.record("reward_components/balance", info["balance"])
                self.logger.record("reward_components/starvation_bonus", info.get("starvation_bonus", 0.0))
        return True


class RichTrainingCallback(BaseCallback):
    """Print compact episode progress using Rich."""

    def __init__(self, console: Console) -> None:
        super().__init__()
        self.console = console
        self.episode_count = 0
        self.recent_rewards: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is None:
                continue

            self.episode_count += 1
            reward = float(episode["r"])
            self.recent_rewards.append(reward)
            self.recent_rewards = self.recent_rewards[-20:]
            average_reward = sum(self.recent_rewards) / len(self.recent_rewards)

            self.console.print(
                "Episode "
                f"{self.episode_count} | reward {reward:.2f} | "
                f"avg reward {average_reward:.2f} | "
                f"completed {info.get('completed_jobs', 0)} | "
                f"rejected {info.get('rejected_jobs', 0)} | "
                f"cpu {info.get('average_cpu_utilization', 0.0):.2f} | "
                f"mem {info.get('average_memory_utilization', 0.0):.2f}"
            )
        return True


def train_agent(config: AppConfigV1_5, console: Console) -> Path:
    """Train PPO and save final/best models and monitor logs."""
    models_dir = Path(config.models_dir)
    results_dir = Path(config.results_dir)
    log_dir = results_dir / "training_logs"
    best_dir = models_dir / "best"
    log_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    train_env = Monitor(CloudResourceEnv(config), filename=str(log_dir / "monitor.csv"))
    eval_env = Monitor(CloudResourceEnv(config))
    model = build_ppo_model(train_env, config.seed, tensorboard_log=str(log_dir / "tensorboard"))
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(log_dir),
        eval_freq=max(config.episode_length, 1),
        deterministic=True,
        render=False,
    )
    rich_callback = RichTrainingCallback(console)
    tb_callback = TensorboardCallback()

    console.print(
        f"[bold]Training PPO[/bold] for {config.training_timesteps:,} timesteps"
    )
    model.learn(
        total_timesteps=config.training_timesteps,
        callback=[tb_callback, rich_callback, eval_callback],
    )

    final_model_path = models_dir / "ppo_final"
    model.save(final_model_path)
    plot_training_reward(log_dir, results_dir)
    console.print(f"Saved final model to [green]{final_model_path}.zip[/green]")
    return final_model_path.with_suffix(".zip")
