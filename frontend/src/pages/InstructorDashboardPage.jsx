import { instructorApi } from "../api/instructorApi.js";
import { predictionApi } from "../api/predictionApi.js";
import AsyncState from "../components/AsyncState.jsx";
import CourseCard from "../components/CourseCard.jsx";
import { useSession } from "../context/SessionContext.jsx";
import { useApi } from "../hooks/useApi.js";

const percent = (v) => `${Math.round(v * 100)}%`;

function CourseOverview({ instructorId, code, presentation }) {
  const state = useApi(() => predictionApi.getCourseRisk(instructorId, code, presentation), [instructorId, code, presentation]);
  return (
    <AsyncState state={state}>
      {(risk) => (
        <p className="muted">
          Toàn môn: <strong>{risk.at_risk_students}</strong>/{risk.total_students} sinh viên có nguy cơ ({percent(risk.at_risk_rate)})
        </p>
      )}
    </AsyncState>
  );
}

function ClassRiskLine({ instructorId, item }) {
  const state = useApi(
    () => predictionApi.getClassRisk(instructorId, item.code_module, item.code_presentation, item.class_id),
    [instructorId, item.class_id]
  );
  if (state.loading) return <p className="muted">Đang tính nguy cơ…</p>;
  if (state.error) return <p className="error">{state.error.message}</p>;
  const r = state.data;
  return <p>{r.at_risk_students}/{r.total_students} sinh viên có nguy cơ ({percent(r.at_risk_rate)})</p>;
}

export default function InstructorDashboardPage() {
  const { session } = useSession();
  const state = useApi(() => instructorApi.getTeachingCourses(session.userId), [session.userId]);

  return (
    <section>
      <h1>Lớp bạn phụ trách</h1>
      <AsyncState state={state}>
        {(items) => {
          if (!items.length) return <p className="muted">Chưa có lớp nào được phân công cho mã này.</p>;
          const courses = [...new Map(items.map((i) => [`${i.code_module}|${i.code_presentation}`, i])).values()];
          return courses.map((course) => (
            <div key={`${course.code_module}-${course.code_presentation}`} className="panel">
              <h2>{course.code_module} · {course.code_presentation}</h2>
              <CourseOverview instructorId={session.userId} code={course.code_module} presentation={course.code_presentation} />
              <div className="grid">
                {items
                  .filter((i) => i.code_module === course.code_module && i.code_presentation === course.code_presentation)
                  .map((i) => (
                    <CourseCard
                      key={i.class_id}
                      title={`Lớp ${i.class_id}`}
                      subtitle={`${i.total_students} sinh viên`}
                      to={`/instructor/courses/${i.code_module}/${i.code_presentation}/classes/${i.class_id}`}
                    >
                      <ClassRiskLine instructorId={session.userId} item={i} />
                    </CourseCard>
                  ))}
              </div>
            </div>
          ));
        }}
      </AsyncState>
    </section>
  );
}
