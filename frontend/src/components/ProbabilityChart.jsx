const W = 640;
const H = 260;
const PAD = { top: 16, right: 20, bottom: 36, left: 44 };

/**
 * Biểu đồ đường SVG thuần.
 * points: [{ x: ngày, y: xác suất 0-1 }]; threshold: đường ngưỡng nguy cơ (tuỳ chọn).
 */
export default function ProbabilityChart({ points, threshold, yLabel = "Xác suất" }) {
  if (!points || points.length === 0) {
    return <p className="muted">Chưa có dữ liệu để vẽ biểu đồ.</p>;
  }
  const xs = points.map((p) => p.x);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;
  const sx = (x) => PAD.left + (maxX === minX ? innerW / 2 : ((x - minX) / (maxX - minX)) * innerW);
  const sy = (y) => PAD.top + (1 - y) * innerH;
  const path = points.map((p, i) => `${i ? "L" : "M"}${sx(p.x)},${sy(p.y)}`).join(" ");
  const ticks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label={yLabel}>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={PAD.left} x2={W - PAD.right} y1={sy(t)} y2={sy(t)} className="chart-grid" />
          <text x={PAD.left - 8} y={sy(t) + 4} textAnchor="end" className="chart-text">
            {Math.round(t * 100)}%
          </text>
        </g>
      ))}
      {threshold != null && (
        <line x1={PAD.left} x2={W - PAD.right} y1={sy(threshold)} y2={sy(threshold)} className="chart-threshold" />
      )}
      <path d={path} className="chart-line" fill="none" />
      {points.map((p) => (
        <g key={p.x}>
          <circle cx={sx(p.x)} cy={sy(p.y)} r="4" className="chart-dot">
            <title>{`Ngày ${p.x}: ${Math.round(p.y * 100)}%`}</title>
          </circle>
          <text x={sx(p.x)} y={H - 12} textAnchor="middle" className="chart-text">
            {p.x}
          </text>
        </g>
      ))}
    </svg>
  );
}
