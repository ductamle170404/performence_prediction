import { createContext, useContext, useMemo, useState } from "react";

// Demo identity. Khi tích hợp LMS thật, lấy danh tính từ phiên đăng nhập của LMS.
const SessionContext = createContext(null);
const STORAGE_KEY = "pp_session";

function readStored() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) || null;
  } catch {
    return null;
  }
}

export function SessionProvider({ children }) {
  const [session, setSession] = useState(readStored);

  const value = useMemo(
    () => ({
      session,
      signIn(role, userId) {
        const next = { role, userId: String(userId).trim() };
        localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
        setSession(next);
      },
      signOut() {
        localStorage.removeItem(STORAGE_KEY);
        setSession(null);
      },
    }),
    [session]
  );

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession() {
  const ctx = useContext(SessionContext);
  if (!ctx) throw new Error("useSession phải nằm trong SessionProvider");
  return ctx;
}
