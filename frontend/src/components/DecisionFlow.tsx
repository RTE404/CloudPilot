import { ArrowDown, CheckCircle2, XCircle } from "lucide-react";

import type { DecisionState } from "../types";

interface DecisionFlowProps {
  decision: DecisionState;
}

export function DecisionFlow({ decision }: DecisionFlowProps) {
  const serverName =
    decision.assigned_server === null
      ? "Pending"
      : `Server ${String.fromCharCode(65 + decision.assigned_server)}`;

  return (
    <section className="panel min-h-[180px]">
      <div className="section-title">Live Decision</div>
      <div className="mt-5 grid place-items-center gap-2 text-center">
        <div className="decision-node">Job #{decision.job_id ?? "--"}</div>
        <ArrowDown size={18} className="text-slate-400" />
        <div className="decision-node border-teal/40 bg-teal/10 text-teal">
          {decision.scheduler}
        </div>
        <ArrowDown size={18} className="text-slate-400" />
        <div className="flex items-center gap-2 rounded border border-line bg-cloud px-4 py-2 font-semibold text-ink">
          {decision.accepted ? (
            <CheckCircle2 size={18} className="text-teal" />
          ) : (
            <XCircle size={18} className="text-rose" />
          )}
          {serverName}
        </div>
      </div>
    </section>
  );
}
