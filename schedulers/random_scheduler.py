"""Random baseline scheduler."""

import numpy as np

from environment.cloud_env import CloudResourceEnv


class RandomScheduler:
    """Select a server uniformly at random."""

    name = "Random"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def select_action(self, env: CloudResourceEnv) -> int:
        """Return a random valid server index."""
        return int(self.rng.integers(0, env.config.servers))
