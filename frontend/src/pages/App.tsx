import { useEffect, useState } from "react";

import { DecisionFlow } from "../components/DecisionFlow";
import { Header } from "../components/Header";
import { Heatmap } from "../components/Heatmap";
import { IncomingJobPanel } from "../components/IncomingJobPanel";
import { JobFlow } from "../components/JobFlow";
import { LiveCharts } from "../components/LiveCharts";
import { MetricsPanel } from "../components/MetricsPanel";
import { ServerGrid } from "../components/ServerGrid";
import { TrainingPanel } from "../components/TrainingPanel";
import { useSimulation } from "../hooks/useSimulation";
import { fetchTraining } from "../services/api";
import type { TrainingState } from "../types";

const emptyTraining: TrainingState = {
  current_episode: 0,
  current_timestep: 0,
  average_reward: 0,
  best_reward: 0,
  loss: null,
  status: "idle"
};

export default function App() {
  const { state, connected } = useSimulation();
  const [training, setTraining] = useState<TrainingState>(emptyTraining);

  useEffect(() => {
    const interval = window.setInterval(() => {
      fetchTraining()
        .then(setTraining)
        .catch(() => undefined);
    }, 2000);

    return () => window.clearInterval(interval);
  }, []);

  if (!state) {
    return (
      <main className="grid min-h-screen place-items-center bg-cloud text-ink">
        <div className="text-lg font-semibold">CloudPilot</div>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-cloud text-ink">
      <Header state={state} connected={connected} />
      <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-4">
        <MetricsPanel metrics={state.metrics} />

        <div className="grid gap-4 lg:grid-cols-[1fr_1fr_1.2fr]">
          <IncomingJobPanel job={state.incoming_job} />
          <DecisionFlow decision={state.decision} />
          <JobFlow job={state.incoming_job} decision={state.decision} />
        </div>

        <ServerGrid servers={state.servers} selectedServer={state.decision.assigned_server} />
        <LiveCharts history={state.history} />
        <Heatmap heatmap={state.heatmap} />
        <TrainingPanel training={training} />
      </div>
    </main>
  );
}
