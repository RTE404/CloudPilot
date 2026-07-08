import { Server } from "lucide-react";

import type { ServerState } from "../types";
import { UtilizationBar } from "./UtilizationBar";

interface ServerGridProps {
  servers: ServerState[];
  selectedServer: number | null;
}

export function ServerGrid({ servers, selectedServer }: ServerGridProps) {
  return (
    <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      {servers.map((server) => (
        <article
          key={server.id}
          className={`panel transition duration-300 ${
            selectedServer === server.id ? "ring-2 ring-teal" : ""
          }`}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Server size={18} className="text-cobalt" />
              <h2 className="text-base font-semibold text-ink">{server.name}</h2>
            </div>
            <span className={statusClass(server.status)}>{server.status}</span>
          </div>

          <div className="mt-5 space-y-4">
            <UtilizationBar label="CPU" value={server.cpu_utilization} tone="cpu" />
            <UtilizationBar label="Memory" value={server.memory_utilization} tone="memory" />
          </div>

          <div className="mt-5 grid grid-cols-2 gap-3">
            <div className="metric-tile">
              <span>Queue</span>
              <strong>{server.queue_length}</strong>
            </div>
            <div className="metric-tile">
              <span>Running</span>
              <strong>{server.running_jobs.length}</strong>
            </div>
          </div>

          <div className="mt-4 flex min-h-9 flex-wrap gap-1">
            {server.running_jobs.slice(0, 8).map((job) => {
              let tierClass = "bg-slate-100 text-slate-700";
              if (job.priority_tier === "gold") tierClass = "bg-yellow-100 text-yellow-800 border-yellow-300 border";
              if (job.priority_tier === "silver") tierClass = "bg-slate-200 text-slate-800 border-slate-400 border";
              if (job.priority_tier === "bronze") tierClass = "bg-orange-100 text-orange-800 border-orange-300 border";
              
              return (
                <span key={job.job_id} className={`job-chip ${tierClass}`}>
                  #{job.job_id}
                </span>
              );
            })}
          </div>
        </article>
      ))}
    </section>
  );
}

function statusClass(status: string) {
  if (status === "Hot") {
    return "rounded bg-rose/10 px-2 py-1 text-xs font-semibold text-rose";
  }
  if (status === "Active") {
    return "rounded bg-teal/10 px-2 py-1 text-xs font-semibold text-teal";
  }
  return "rounded bg-slate-100 px-2 py-1 text-xs font-semibold text-slate-500";
}
