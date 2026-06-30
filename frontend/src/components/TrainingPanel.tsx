import { Save } from "lucide-react";

import type { TrainingState } from "../types";

interface TrainingPanelProps {
  training: TrainingState;
}

export function TrainingPanel({ training }: TrainingPanelProps) {
  return (
    <section className="panel">
      <div className="flex items-center justify-between">
        <div className="section-title">Training</div>
        <button className="icon-button" title="Save checkpoint">
          <Save size={18} />
        </button>
      </div>
      <div className="mt-4 grid gap-3 sm:grid-cols-5">
        <TrainingMetric label="Episode" value={training.current_episode.toFixed(0)} />
        <TrainingMetric label="Timestep" value={training.current_timestep.toFixed(0)} />
        <TrainingMetric label="Avg Reward" value={training.average_reward.toFixed(2)} />
        <TrainingMetric label="Best Reward" value={training.best_reward.toFixed(2)} />
        <TrainingMetric label="Loss" value={training.loss === null ? "--" : training.loss.toFixed(4)} />
      </div>
    </section>
  );
}

function TrainingMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-tile">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
