"""Least-loaded baseline scheduler."""

from environment.cloud_env import CloudResourceEnv


class LeastLoadedScheduler:
    """Select the server with the lowest current load score."""

    name = "Least Loaded"

    def select_action(self, env: CloudResourceEnv) -> int:
        """Return the index of the least-loaded server."""
        return min(env.servers, key=lambda server: server.load_score).id
