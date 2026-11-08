const LABELS = { LOW: "Nguy cơ thấp", MEDIUM: "Nguy cơ trung bình", HIGH: "Nguy cơ cao" };

export default function RiskBadge({ level, probability }) {
  if (!level) return <span className="badge badge-none">Chưa có dự báo</span>;
  const percent = probability != null ? ` · ${Math.round(probability * 100)}%` : "";
  return (
    <span className={`badge badge-${level.toLowerCase()}`}>
      {LABELS[level] ?? level}
      {percent}
    </span>
  );
}
