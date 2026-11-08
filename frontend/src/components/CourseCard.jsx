import { Link } from "react-router-dom";

export default function CourseCard({ title, subtitle, status, to, children }) {
  return (
    <Link to={to} className="course-card">
      <div className="course-card-head">
        <h3>{title}</h3>
        {status && <span className={`status status-${status.tone}`}>{status.label}</span>}
      </div>
      {subtitle && <p className="muted">{subtitle}</p>}
      {children}
    </Link>
  );
}
