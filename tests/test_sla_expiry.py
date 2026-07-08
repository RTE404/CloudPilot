"""Unit test for SLA expiry logic."""

from environment.cloud_env import CloudResourceEnv
from utils.helpers import load_config_v1_5
from environment.priority import PriorityTier
from environment.job import Job

def test_sla_expiry():
    config = load_config_v1_5("configs/config_v1_5.yaml")
    env = CloudResourceEnv(config)
    env.reset()
    
    # Create a job with deadline 3, waited 4
    job = Job(1, 1.0, 1.0, 10, 0, priority_tier=PriorityTier.GOLD, deadline_ticks=3)
    job.ticks_waited = 3 # will be incremented to 4 in step()
    
    env.global_queue = [job]
    env.current_job = Job(2, 1.0, 1.0, 10, 0, priority_tier=PriorityTier.GOLD, deadline_ticks=3)
    
    # Use action 0 to place the current job
    obs, reward, terminated, truncated, info = env.step(0)
    
    # SLA violation should be counted for the queued job
    assert env.rejected_jobs_count == 1  # Or more if action failed
    assert len(env.global_queue) == 0 or env.global_queue[0].job_id != 1
    
    assert info.get("sla_violation", 0.0) < 0.0
