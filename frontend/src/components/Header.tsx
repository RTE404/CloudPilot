import { Gauge, Pause, Play, RotateCcw } from "lucide-react";

import { pauseSimulation, resetSimulation, startSimulation, updateControl } from "../services/api";
import type { SimulationState } from "../types";

const schedulers = ["Random", "Round Robin", "Least Loaded", "PPO"];

interface HeaderProps {
  state: SimulationState;
  connected: boolean;
}

export function Header({ state, connected }: HeaderProps) {
  return (
    <header className="border-b border-line bg-panel">
      <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-ink">CloudPilot</h1>
          <div className="mt-1 flex flex-wrap items-center gap-3 text-sm text-slate-600">
            <span className="inline-flex items-center gap-2">
              <span className={`h-2 w-2 rounded-full ${connected ? "bg-teal" : "bg-rose"}`} />
              {state.status}
            </span>
            <span>{state.scheduler}</span>
            <span>Episode {state.episode}</span>
            <span>{state.speed.toFixed(1)}x</span>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <div className="flex rounded border border-line bg-cloud p-1">
            {schedulers.map((scheduler) => (
              <button
                key={scheduler}
                className={`h-9 rounded px-3 text-sm font-medium transition ${
                  state.scheduler === scheduler
                    ? "bg-ink text-white"
                    : "text-slate-600 hover:bg-white"
                }`}
                onClick={() => updateControl({ scheduler })}
              >
                {scheduler}
              </button>
            ))}
          </div>

          <label className="flex h-10 items-center gap-2 rounded border border-line bg-panel px-3 text-sm text-slate-600">
            <Gauge size={16} />
            <input
              aria-label="Simulation speed"
              className="w-28 accent-teal"
              type="range"
              min="0.25"
              max="10"
              step="0.25"
              value={state.speed}
              onChange={(event) => updateControl({ speed: Number(event.target.value) })}
            />
          </label>

          <button className="icon-button" title="Start" onClick={() => startSimulation()}>
            <Play size={18} />
          </button>
          <button className="icon-button" title="Pause" onClick={() => pauseSimulation()}>
            <Pause size={18} />
          </button>
          <button className="icon-button" title="Reset" onClick={() => resetSimulation()}>
            <RotateCcw size={18} />
          </button>
        </div>
      </div>
    </header>
  );
}
