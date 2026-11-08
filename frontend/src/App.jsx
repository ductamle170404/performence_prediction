import { Navigate, Route, Routes } from "react-router-dom";
import { useSession } from "./context/SessionContext.jsx";
import MainLayout from "./layouts/MainLayout.jsx";
import LoginPage from "./pages/LoginPage.jsx";
import StudentCoursesPage from "./pages/StudentCoursesPage.jsx";
import StudentCoursePage from "./pages/StudentCoursePage.jsx";
import StudentPredictionPage from "./pages/StudentPredictionPage.jsx";
import InstructorDashboardPage from "./pages/InstructorDashboardPage.jsx";
import ClassDetailPage from "./pages/ClassDetailPage.jsx";

function RequireRole({ role, children }) {
  const { session } = useSession();
  if (!session) return <Navigate to="/login" replace />;
  if (session.role !== role) {
    return <Navigate to={session.role === "instructor" ? "/instructor" : "/student/courses"} replace />;
  }
  return children;
}

export default function App() {
  const { session } = useSession();
  const home = session?.role === "instructor" ? "/instructor" : "/student/courses";

  return (
    <Routes>
      <Route element={<MainLayout />}>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/" element={<Navigate to={session ? home : "/login"} replace />} />

        <Route path="/student/courses" element={<RequireRole role="student"><StudentCoursesPage /></RequireRole>} />
        <Route path="/student/courses/:code/:presentation" element={<RequireRole role="student"><StudentCoursePage /></RequireRole>} />
        <Route path="/student/courses/:code/:presentation/prediction" element={<RequireRole role="student"><StudentPredictionPage /></RequireRole>} />

        <Route path="/instructor" element={<RequireRole role="instructor"><InstructorDashboardPage /></RequireRole>} />
        <Route path="/instructor/courses/:code/:presentation/classes/:classId" element={<RequireRole role="instructor"><ClassDetailPage /></RequireRole>} />

        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
