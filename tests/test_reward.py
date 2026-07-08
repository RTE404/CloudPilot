"""Unit tests for the reward calculation in v1.5."""

from environment.cloud_env import CloudResourceEnv, TickEvents
from environment.job import Job
from environment.priority import PriorityTier
from utils.helpers import load_config_v1_5

def test_compute_reward():
    config = load_config_v1_5("configs/config_v1_5.yaml")
    env = CloudResourceEnv(config)
    env.reset()

    tick_events = TickEvents()
    
    job_c = Job(1, 1.0, 1.0, 10, 0, priority_tier=PriorityTier.GOLD, deadline_ticks=5)
    tick_events.completions.append(job_c)
    
    job_v = Job(2, 1.0, 1.0, 10, 0, priority_tier=PriorityTier.SILVER, deadline_ticks=5)
    tick_events.sla_violations.append(job_v)
    
    job_r = Job(3, 1.0, 1.0, 10, 0, priority_tier=PriorityTier.BRONZE, deadline_ticks=5)
    tick_events.rejections.append(job_r)

    # Force an overload of 2 CPU units on server 0
    env.servers[0].used["cpu"] = env.servers[0].capacity["cpu"] + 2.0
    
    # Force queue length to 3
    env.global_queue = [job_r] * 3
    
    reward, components = env._compute_reward(tick_events)
    
    assert components["completion"] == 3.0
    assert components["sla_violation"] == -4.0
    assert components["rejection"] == -1.0
    assert components["overload"] == -1.0
    assert components["queue"] == -0.03
