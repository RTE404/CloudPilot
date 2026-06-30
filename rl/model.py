"""Model factory for PPO scheduling agents."""

from stable_baselines3 import PPO

from environment.cloud_env import CloudResourceEnv


def build_ppo_model(env: CloudResourceEnv, seed: int) -> PPO:
    """Create a PPO model with stable-baselines3 defaults."""
    return PPO("MlpPolicy", env, verbose=0, seed=seed)
