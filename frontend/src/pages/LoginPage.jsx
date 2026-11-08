import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { instructorApi } from "../api/instructorApi.js";
import { useApi } from "../hooks/useApi.js";
import { useSession } from "../context/SessionContext.jsx";

export default function LoginPage() {
  const { signIn } = useSession();
  const navigate = useNavigate();
  const [role, setRole] = useState("student");
  const [userId, setUserId] = useState("");
  const demo = useApi(() => instructorApi.getDemoUsers(), []);
  const suggestions = demo.data ? (role === "student" ? demo.data.students : demo.data.instructors) : [];

  function submit(event) {
    event.preventDefault();
    if (!userId.trim()) return;
    signIn(role, userId);
    navigate(role === "instructor" ? "/instructor" : "/student/courses");
  }

  return (
    <section className="panel narrow">
      <h1>Đăng nhập demo</h1>
      <p className="muted">
        Chỉ dùng cho prototype. Khi tích hợp LMS, danh tính lấy từ phiên đăng nhập của LMS.
      </p>
      <form onSubmit={submit} className="form">
        <fieldset className="segmented">
          <legend className="visually-hidden">Vai trò</legend>
          {[["student", "Sinh viên"], ["instructor", "Giảng viên"]].map(([value, label]) => (
            <label key={value} className={role === value ? "active" : ""}>
              <input type="radio" name="role" value={value} checked={role === value}
                     onChange={() => { setRole(value); setUserId(""); }} />
              {label}
            </label>
          ))}
        </fieldset>
        <label>
          {role === "student" ? "Mã sinh viên" : "Mã giảng viên"}
          <input value={userId} onChange={(e) => setUserId(e.target.value)} autoFocus required />
        </label>
        {suggestions.length > 0 && (
          <div className="chips">
            <span className="muted">Gợi ý từ dữ liệu:</span>
            {suggestions.map((id) => (
              <button type="button" key={id} className="chip" onClick={() => setUserId(String(id))}>{id}</button>
            ))}
          </div>
        )}
        {demo.error && <p className="error">{demo.error.message}</p>}
        <button className="btn" type="submit">Vào hệ thống</button>
      </form>
    </section>
  );
}
