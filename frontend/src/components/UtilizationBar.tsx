interface UtilizationBarProps {
  label: string;
  value: number;
  tone: "cpu" | "memory";
}

export function UtilizationBar({ label, value, tone }: UtilizationBarProps) {
  const percent = Math.round(value * 100);
  const color = tone === "cpu" ? "bg-teal" : "bg-cobalt";

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs font-medium text-slate-600">
        <span>{label}</span>
        <span>{percent}%</span>
      </div>
      <div className="h-2 overflow-hidden rounded bg-slate-200">
        <div
          className={`h-full ${color} transition-all duration-500`}
          style={{ width: `${Math.min(percent, 100)}%` }}
        />
      </div>
    </div>
  );
}
