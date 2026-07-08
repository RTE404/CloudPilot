import { Cpu, MemoryStick, Timer } from "lucide-react";
import type React from "react";

import type { JobState } from "../types";

interface IncomingJobPanelProps {
  job: JobState | null;
}

export function IncomingJobPanel({ job }: IncomingJobPanelProps) {
  const getTierColor = (tier: string) => {
    switch (tier.toLowerCase()) {
      case "gold": return "border-yellow-400 bg-yellow-50 text-yellow-900";
      case "silver": return "border-slate-400 bg-slate-50 text-slate-900";
      case "bronze": return "border-orange-300 bg-orange-50 text-orange-900";
      default: return "border-line bg-cloud";
    }
  };

  return (
    <section className="panel min-h-[180px]">
      <div className="section-title">Incoming Job</div>
      {job ? (
        <div key={job.job_id} className={`mt-5 animate-pop rounded border p-4 ${getTierColor(job.priority_tier)}`}>
          <div className="flex justify-between items-center">
            <div className="text-xl font-semibold">Job #{job.job_id}</div>
            <div className="text-xs uppercase font-bold tracking-wider px-2 py-1 rounded bg-white/50">{job.priority_tier}</div>
          </div>
          <div className="mt-4 grid grid-cols-4 gap-3">
            <MetricIcon icon={<Cpu size={16} />} label="CPU" value={`${job.cpu_required.toFixed(1)}%`} />
            <MetricIcon icon={<MemoryStick size={16} />} label="Memory" value={`${job.memory_required.toFixed(1)} GB`} />
            <MetricIcon icon={<Timer size={16} />} label="Runtime" value={`${job.runtime}`} />
            <MetricIcon icon={<Timer size={16} />} label="Deadline" value={`${job.deadline_ticks}`} />
          </div>
        </div>
      ) : (
        <div className="mt-8 text-sm text-slate-500">No active job</div>
      )}
    </section>
  );
}

function MetricIcon({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="rounded border border-line bg-panel p-3">
      <div className="flex items-center gap-2 text-xs text-slate-500">
        {icon}
        {label}
      </div>
      <div className="mt-1 text-sm font-semibold text-ink">{value}</div>
    </div>
  );
}
