export interface JobState {
  job_id: number;
  cpu_required: number;
  memory_required: number;
  runtime: number;
  arrival_time: number;
  remaining_time: number | null;
}

export interface ServerState {
  id: number;
  name: string;
  cpu_utilization: number;
  memory_utilization: number;
  queue_length: number;
  running_jobs: JobState[];
  status: string;
}

export interface MetricsState {
  jobs_processed: number;
  jobs_completed: number;
  jobs_rejected: number;
  average_response_time: number;
  average_queue_length: number;
  average_cpu_utilization: number;
  average_memory_utilization: number;
  current_reward: number;
  episode_reward: number;
}

export interface DecisionState {
  job_id: number | null;
  scheduler: string;
  assigned_server: number | null;
  accepted: boolean;
  reward: number;
}

export interface ChartPoint {
  step: number;
  episode: number;
  reward: number;
  average_queue_length: number;
  jobs_completed: number;
  jobs_rejected: number;
  cpu: number[];
  memory: number[];
}

export interface SimulationState {
  project_name: string;
  status: string;
  scheduler: string;
  episode: number;
  step: number;
  speed: number;
  incoming_job: JobState | null;
  servers: ServerState[];
  decision: DecisionState;
  metrics: MetricsState;
  history: ChartPoint[];
  heatmap: number[][];
}

export interface TrainingState {
  current_episode: number;
  current_timestep: number;
  average_reward: number;
  best_reward: number;
  loss: number | null;
  status: string;
}
