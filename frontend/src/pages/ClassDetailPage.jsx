import { Link, useParams } from "react-router-dom";
import { instructorApi } from "../api/instructorApi.js";
import { predictionApi } from "../api/predictionApi.js";
import AsyncState from "../components/AsyncState.jsx";
import ProbabilityChart from "../components/ProbabilityChart.jsx";
import RiskBadge from "../components/RiskBadge.jsx";
import { useSession } from "../context/SessionContext.jsx";
import { useApi } from "../hooks/useApi.js";

export default function ClassDetailPage() {
  const { code, presentation, classId } = useParams();
  const { session } = useSession();
  const id = session.userId;
  const students = useApi(() => instructorApi.getClassStudents(id, code, presentation, classId), [id, code, presentation, classId]);
  const trend = useApi(() => predictionApi.getRiskHistory(id, code, presentation, classId), [id, code, presentation, classId]);

  return (
    <section>
      <Link to="/instructor" className="back">Quay lại danh sách lớp</Link>
      <h1>Lớp {classId}</h1>
      <p className="muted">{code} · {presentation}</p>

      <div className="panel">
        <h2>Tỷ lệ sinh viên có nguy cơ theo thời gian</h2>
        <AsyncState state={trend}>
          {(points) => (
            <ProbabilityChart
              points={points.map((p) => ({ x: p.cutoff_date, y: p.at_risk_rate }))}
              yLabel="Tỷ lệ sinh viên có nguy cơ"
            />
          )}
        </AsyncState>
      </div>

      <div className="panel">
        <h2>Sinh viên (nguy cơ cao nhất ở trên)</h2>
        <AsyncState state={students}>
          {(rows) => (
            <table className="table">
              <thead>
                <tr><th>Mã sinh viên</th><th>Hoạt động có điểm</th><th>Xác suất điểm liệt</th><th>Mức nguy cơ</th></tr>
              </thead>
              <tbody>
                {rows.map((s) => (
                  <tr key={s.student_id}>
                    <td>{s.student_id}</td>
                    <td>{s.n_assessments}</td>
                    <td>{s.fail_probability == null ? "—" : `${Math.round(s.fail_probability * 100)}%`}</td>
                    <td><RiskBadge level={s.risk_level} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </AsyncState>
      </div>
    </section>
  );
}
