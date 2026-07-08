"""Unit test for priority sampling distribution."""

from environment.generator import JobGenerator
from environment.priority import PriorityTier
from utils.helpers import load_config_v1_5

def test_priority_sampling():
    config = load_config_v1_5("configs/config_v1_5.yaml")
    gen = JobGenerator(
        min_cpu=1.0, max_cpu=2.0, min_memory=1.0, max_memory=2.0,
        min_duration=1, max_duration=2, seed=42,
        priority_distribution=config.priority.distribution,
        deadline_ticks_range=config.priority.deadline_ticks
    )
    
    counts = {PriorityTier.GOLD: 0, PriorityTier.SILVER: 0, PriorityTier.BRONZE: 0}
    n = 10000
    for _ in range(n):
        job = gen.generate(0)
        counts[job.priority_tier] += 1
        
    gold_prop = counts[PriorityTier.GOLD] / n
    silver_prop = counts[PriorityTier.SILVER] / n
    bronze_prop = counts[PriorityTier.BRONZE] / n
    
    assert abs(gold_prop - 0.15) < 0.02
    assert abs(silver_prop - 0.35) < 0.02
    assert abs(bronze_prop - 0.50) < 0.02
