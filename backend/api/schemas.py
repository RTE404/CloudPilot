"""Pydantic schemas returned by the CloudPilot API."""

from pydantic import BaseModel


class JobState(BaseModel):
    """Current incoming or running job state."""

    job_id: int
    cpu_required: float
    memory_required: float
    runtime: int
    arrival_time: int
    priority_tier: str = "bronze"
    deadline_ticks: int = 0
    remaining_time: int | None = None


class ServerState(BaseModel):
    """Serializable server state for the dashboard."""

    id: int
    name: str
    cpu_utilization: float
    memory_utilization: float
    queue_length: int
    running_jobs: list[JobState]
    status: str


class MetricsState(BaseModel):
    """Live KPI snapshot."""

    jobs_processed: int
    jobs_completed: int
    jobs_rejected: int
    average_response_time: float
    average_queue_length: float
    average_cpu_utilization: float
    average_memory_utilization: float
    current_reward: float
    episode_reward: float
    sla_violation_rate: float
    priority_weighted_completion_rate: float


class DecisionState(BaseModel):
    """Most recent scheduling decision."""

    job_id: int | None = None
    scheduler: str
    assigned_server: int | None = None
    accepted: bool = False
    reward: float = 0.0


class ChartPoint(BaseModel):
    """One time-series point for live charts."""

    step: int
    episode: int
    reward: float
    average_queue_length: float
    jobs_completed: int
    jobs_rejected: int
    cpu: list[float]
    memory: list[float]


class SimulationState(BaseModel):
    """Complete state streamed to dashboard clients."""

    project_name: str = "CloudPilot"
    status: str
    scheduler: str
    episode: int
    step: int
    speed: float
    incoming_job: JobState | None
    servers: list[ServerState]
    decision: DecisionState
    metrics: MetricsState
    history: list[ChartPoint]
    heatmap: list[list[float]]


class ControlRequest(BaseModel):
    """Simulation control request."""

    scheduler: str | None = None
    speed: float | None = None


class TrainingState(BaseModel):
    """Training progress fields displayed by the dashboard."""

    current_episode: int = 0
    current_timestep: int = 0
    average_reward: float = 0.0
    best_reward: float = 0.0
    loss: float | None = None
    status: str = "idle"
