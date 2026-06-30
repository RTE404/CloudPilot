"""Round-robin baseline scheduler."""

from environment.cloud_env import CloudResourceEnv


class RoundRobinScheduler:
    """Cycle through servers in order."""

    name = "Round Robin"

    def __init__(self) -> None:
        self.next_server = 0

    def reset(self) -> None:
        """Reset the next selected server to the first server."""
        self.next_server = 0

    def select_action(self, env: CloudResourceEnv) -> int:
        """Return the next server index in cyclic order."""
        action = self.next_server
        self.next_server = (self.next_server + 1) % env.config.servers
        return action
