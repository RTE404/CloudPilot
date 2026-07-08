"""Unit test for starvation guard logic."""

from environment.cloud_env import CloudResourceEnv
from utils.helpers import load_config_v1_5
from environment.priority import PriorityTier
from environment.job import Job

def test_starvation_guard():
    config = load_config_v1_5("configs/config_v1_5.yaml")
    env = CloudResourceEnv(config)
    env.reset()
    
    # Create a job that has waited beyond threshold
    threshold = config.reward.starvation_guard.threshold_ticks
    job = Job(1, 1.0, 1.0, 10, 0, priority_tier=PriorityTier.BRONZE, deadline_ticks=100)
    job.ticks_waited = threshold + 1
    
    env.global_queue = [job]
    env.current_job = Job(2, 1.0, 1.0, 10, 0, priority_tier=PriorityTier.BRONZE, deadline_ticks=100)
    
    # Fill all servers so the queued job cannot be placed
    for server in env.servers:
        server.used["cpu"] = server.capacity["cpu"]
        server.used["memory"] = server.capacity["memory"]
        
    obs, reward, terminated, truncated, info = env.step(0)
    
    # Bonus per tick over is 0.05, it waited 1 tick over (after increment it becomes threshold + 2)
    expected_bonus = 0.05 * 2
    assert info.get("starvation_bonus", 0.0) == expected_bonus
