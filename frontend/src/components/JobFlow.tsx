import type { DecisionState, JobState } from "../types";

interface JobFlowProps {
  job: JobState | null;
  decision: DecisionState;
}

export function JobFlow({ job, decision }: JobFlowProps) {
  const target =
    decision.assigned_server === null
      ? "Waiting"
      : `Server ${String.fromCharCode(65 + decision.assigned_server)}`;

  return (
    <section className="panel">
      <div className="section-title">Job Flow</div>
      <div className="flow-track mt-5">
        <span>Incoming</span>
        <span>CloudPilot</span>
        <span>{target}</span>
        {job && <div key={job.job_id} className="flow-dot" />}
      </div>
    </section>
  );
}
