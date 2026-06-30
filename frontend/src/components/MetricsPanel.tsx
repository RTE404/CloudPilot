import type { MetricsState } from "../types";

interface MetricsPanelProps {
  metrics: MetricsState;
}

const metricMap: Array<[keyof MetricsState, string, (value: number) => string]> = [
  ["jobs_processed", "Processed", (value) => value.toFixed(0)],
  ["jobs_completed", "Completed", (value) => value.toFixed(0)],
  ["jobs_rejected", "Rejected", (value) => value.toFixed(0)],
  ["average_response_time", "Avg Response", (value) => value.toFixed(2)],
  ["average_queue_length", "Avg Queue", (value) => value.toFixed(2)],
  ["average_cpu_utilization", "Avg CPU", (value) => `${Math.round(value * 100)}%`],
  ["average_memory_utilization", "Avg Memory", (value) => `${Math.round(value * 100)}%`],
  ["current_reward", "Reward", (value) => value.toFixed(1)],
  ["episode_reward", "Episode Reward", (value) => value.toFixed(1)]
];

export function MetricsPanel({ metrics }: MetricsPanelProps) {
  return (
    <section className="grid gap-3 sm:grid-cols-3 xl:grid-cols-9">
      {metricMap.map(([key, label, format]) => (
        <div key={key} className="metric-tile min-h-20">
          <span>{label}</span>
          <strong>{format(metrics[key])}</strong>
        </div>
      ))}
    </section>
  );
}
