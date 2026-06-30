import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import type React from "react";

import type { ChartPoint } from "../types";

interface LiveChartsProps {
  history: ChartPoint[];
}

export function LiveCharts({ history }: LiveChartsProps) {
  const utilization = history.map((point) => ({
    step: point.step,
    sA: Math.round((point.cpu[0] ?? 0) * 100),
    sB: Math.round((point.cpu[1] ?? 0) * 100),
    sC: Math.round((point.cpu[2] ?? 0) * 100),
    sD: Math.round((point.cpu[3] ?? 0) * 100),
    mA: Math.round((point.memory[0] ?? 0) * 100),
    mB: Math.round((point.memory[1] ?? 0) * 100),
    mC: Math.round((point.memory[2] ?? 0) * 100),
    mD: Math.round((point.memory[3] ?? 0) * 100)
  }));

  return (
    <section className="grid gap-4 xl:grid-cols-2">
      <ChartShell title="Reward vs Episode">
        <ResponsiveContainer>
          <AreaChart data={history}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="step" />
            <YAxis />
            <Tooltip />
            <Area type="monotone" dataKey="reward" stroke="#0f9f9a" fill="#0f9f9a" fillOpacity={0.18} />
          </AreaChart>
        </ResponsiveContainer>
      </ChartShell>

      <ChartShell title="Average Queue Length">
        <ResponsiveContainer>
          <LineChart data={history}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="step" />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="average_queue_length" stroke="#d99025" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </ChartShell>

      <ChartShell title="CPU Utilization per Server">
        <ResponsiveContainer>
          <LineChart data={utilization}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="step" />
            <YAxis domain={[0, 100]} />
            <Tooltip />
            <Line type="monotone" dataKey="sA" stroke="#0f9f9a" dot={false} />
            <Line type="monotone" dataKey="sB" stroke="#2d6cdf" dot={false} />
            <Line type="monotone" dataKey="sC" stroke="#d99025" dot={false} />
            <Line type="monotone" dataKey="sD" stroke="#d1495b" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </ChartShell>

      <ChartShell title="Jobs Completed and Rejected">
        <ResponsiveContainer>
          <BarChart data={history}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="step" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="jobs_completed" fill="#0f9f9a" />
            <Bar dataKey="jobs_rejected" fill="#d1495b" />
          </BarChart>
        </ResponsiveContainer>
      </ChartShell>
    </section>
  );
}

function ChartShell({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="panel">
      <div className="section-title">{title}</div>
      <div className="mt-4 h-64">{children}</div>
    </section>
  );
}
