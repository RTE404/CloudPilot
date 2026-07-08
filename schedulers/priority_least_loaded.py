"""Priority-aware Least-loaded baseline scheduler."""

from environment.cloud_env import CloudResourceEnv


class PriorityLeastLoadedScheduler:
    """Select the server with the lowest current load score, with priority-aware queue handling."""

    name = "Priority Least Loaded"

    def select_action(self, env: CloudResourceEnv) -> int:
        """Return the index of the least-loaded server."""
        # The environment handles the queue retry order based on its configuration.
        # Ensure the environment is configured for priority_fifo.
        if hasattr(env.config, "priority"):
            env.config.priority.queue_retry_order = "priority_fifo"
        return min(env.servers, key=lambda server: server.load_score).id
