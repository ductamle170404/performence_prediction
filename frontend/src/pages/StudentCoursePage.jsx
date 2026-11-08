import { Link, useParams } from "react-router-dom";
import { courseApi } from "../api/courseApi.js";
import AssessmentTimeline from "../components/AssessmentTimeline.jsx";
import AsyncState from "../components/AsyncState.jsx";
import { useApi } from "../hooks/useApi.js";

export default function StudentCoursePage() {
  const { code, presentation } = useParams();
  const state = useApi(() => courseApi.getAssessments(code, presentation), [code, presentation]);

  return (
    <section>
      <Link to="/student/courses" className="back">Quay lại danh sách môn</Link>
      <h1>{code} · {presentation}</h1>
      <AsyncState state={state}>
        {(timeline) => (
          <>
            <p className="muted">
              {timeline.is_ongoing
                ? `Học kỳ đang diễn ra, hiển thị hoạt động đến ngày ${timeline.current_day}.`
                : "Học kỳ đã kết thúc, hiển thị toàn bộ hoạt động."}
            </p>
            <div className="panel">
              <AssessmentTimeline
                assessments={timeline.assessments}
                currentDay={timeline.current_day}
                length={timeline.length}
              />
            </div>
            <Link className="btn" to={`/student/courses/${code}/${presentation}/prediction`}>
              Xem dự báo điểm liệt
            </Link>
          </>
        )}
      </AsyncState>
    </section>
  );
}
