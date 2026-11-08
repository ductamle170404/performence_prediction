import { Link, Outlet, useNavigate } from "react-router-dom";
import { useSession } from "../context/SessionContext.jsx";

export default function MainLayout() {
  const { session, signOut } = useSession();
  const navigate = useNavigate();
  const home = session?.role === "instructor" ? "/instructor" : "/student/courses";

  return (
    <div className="shell">
      <header className="topbar">
        <Link to={home} className="brand">Dự báo điểm liệt</Link>
        <nav className="topbar-nav">
          {session && (
            <>
              <span className="muted">
                {session.role === "instructor" ? "Giảng viên" : "Sinh viên"} {session.userId}
              </span>
              <button
                className="btn btn-ghost"
                onClick={() => {
                  signOut();
                  navigate("/login");
                }}
              >
                Đổi tài khoản
              </button>
            </>
          )}
        </nav>
      </header>
      <main className="content">
        <Outlet />
      </main>
    </div>
  );
}
