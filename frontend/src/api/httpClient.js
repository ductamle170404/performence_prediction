const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

export async function get(path, params = {}) {
  const url = new URL(path, BASE_URL);
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      url.searchParams.set(key, value);
    }
  });

  let response;
  try {
    response = await fetch(url);
  } catch {
    throw new ApiError("Không kết nối được tới backend. Hãy kiểm tra server đang chạy.", 0);
  }
  if (!response.ok) {
    let detail = `Lỗi ${response.status}`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* giữ thông báo mặc định */
    }
    throw new ApiError(detail, response.status);
  }
  return response.json();
}
