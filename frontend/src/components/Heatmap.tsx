interface HeatmapProps {
  heatmap: number[][];
}

export function Heatmap({ heatmap }: HeatmapProps) {
  return (
    <section className="panel">
      <div className="section-title">Utilization Heatmap</div>
      <div className="mt-4 space-y-2">
        {heatmap.map((row, rowIndex) => (
          <div key={rowIndex} className="grid grid-cols-[72px_1fr] items-center gap-3">
            <span className="text-xs font-semibold text-slate-500">
              Server {String.fromCharCode(65 + rowIndex)}
            </span>
            <div className="grid grid-flow-col auto-cols-fr gap-1">
              {Array.from({ length: 60 }).map((_, columnIndex) => {
                const value = row[columnIndex] ?? 0;
                return (
                  <span
                    key={columnIndex}
                    className="h-4 rounded-sm"
                    style={{ backgroundColor: heatColor(value) }}
                  />
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function heatColor(value: number) {
  if (value > 0.85) {
    return "#d1495b";
  }
  if (value > 0.6) {
    return "#d99025";
  }
  if (value > 0.25) {
    return "#0f9f9a";
  }
  return "#d8dee9";
}
