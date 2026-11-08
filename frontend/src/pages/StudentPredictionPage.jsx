import { Link, useParams } from "react-router-dom";
import { predictionApi } from "../api/predictionApi.js";
import AsyncState from "../components/AsyncState.jsx";
import ProbabilityChart from "../components/ProbabilityChart.jsx";
import RiskBadge from "../components/RiskBadge.jsx";
import { useSession } from "../context/SessionContext.jsx";
import { useApi } from "../hooks/useApi.js";

export default function StudentPredictionPage() {
  const { code, presentation } = useParams();
  const { session } = useSession();
  const studentId = session.userId;
  const state = useApi(
    () => predictionApi.getStudentPredictionHistory(studentId, code, presentation),
    [studentId, code, presentation]
  );

  return (
    <section>
      <Link to={`/student/courses/${code}/${presentation}`} className="back">Quay lại môn học</Link>
      <h1>Dự báo nguy cơ điểm liệt · {code} {presentation}</h1>
      <AsyncState state={state}>
        {({ history }) => {
          const latest = history[history.length - 1];
          return (
            <>
              {latest && (
                <div className="panel summary">
                  <div>
                    <div className="big">{Math.round(latest.fail_probability * 100)}%</div>
                    <div className="muted">xác suất bị điểm liệt sau {latest.n_assessments} hoạt động tính điểm</div>
                  </div>
                  <RiskBadge level={latest.risk_level} />
                </div>
              )}
              <p className="muted">
                Đây là xác suất do mô hình ước lượng, không phải điểm cuối cùng.
              </p>
              <div className="panel">
                <h2>Diễn biến theo từng mốc</h2>
                <ProbabilityChart points={history.map((p) => ({ x: p.cutoff_date, y: p.fail_probability }))} />
              </div>
              <div className="panel">
                <table className="table">
                  <thead>
                    <tr><th>Ngày chốt</th><th>Hoạt động có điểm</th><th>Xác suất</th><th>Mức nguy cơ</th></tr>
                  </thead>
                  <tbody>
                    {history.map((p) => (
                      <tr key={p.cutoff_date}>
                        <td>Ngày {p.cutoff_date}</td>
                        <td>{p.n_assessments}</td>
                        <td>{Math.round(p.fail_probability * 100)}%</td>
                        <td><RiskBadge level={p.risk_level} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          );
        }}
      </AsyncState>
    </section>
  );
}
