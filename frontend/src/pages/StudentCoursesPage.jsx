import { courseApi } from "../api/courseApi.js";
import AsyncState from "../components/AsyncState.jsx";
import CourseCard from "../components/CourseCard.jsx";
import { useSession } from "../context/SessionContext.jsx";
import { useApi } from "../hooks/useApi.js";

export default function StudentCoursesPage() {
  const { session } = useSession();
  const state = useApi(() => courseApi.getStudentCourses(session.userId), [session.userId]);

  return (
    <section>
      <h1>Môn học của bạn</h1>
      <AsyncState state={state}>
        {(courses) => (
          <div className="grid">
            {courses.map((c) => (
              <CourseCard
                key={`${c.code_module}-${c.code_presentation}`}
                title={`${c.code_module} · ${c.code_presentation}`}
                subtitle={`Lớp ${c.class_id} · ${c.length} ngày`}
                status={c.is_ongoing ? { label: "Đang học", tone: "live" } : { label: "Đã kết thúc", tone: "done" }}
                to={`/student/courses/${c.code_module}/${c.code_presentation}`}
              />
            ))}
          </div>
        )}
      </AsyncState>
    </section>
  );
}
