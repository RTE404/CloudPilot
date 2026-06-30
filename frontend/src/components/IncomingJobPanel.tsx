import { Cpu, MemoryStick, Timer } from "lucide-react";
import type React from "react";

import type { JobState } from "../types";

interface IncomingJobPanelProps {
  job: JobState | null;
}

export function IncomingJobPanel({ job }: IncomingJobPanelProps) {
  return (
    <section className="panel min-h-[180px]">
      <div className="section-title">Incoming Job</div>
      {job ? (
        <div key={job.job_id} className="mt-5 animate-pop rounded border border-line bg-cloud p-4">
          <div className="text-xl font-semibold text-ink">Job #{job.job_id}</div>
          <div className="mt-4 grid grid-cols-3 gap-3">
            <MetricIcon icon={<Cpu size={16} />} label="CPU" value={`${job.cpu_required.toFixed(1)}%`} />
            <MetricIcon icon={<MemoryStick size={16} />} label="Memory" value={`${job.memory_required.toFixed(1)} GB`} />
            <MetricIcon icon={<Timer size={16} />} label="Runtime" value={`${job.runtime}`} />
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
