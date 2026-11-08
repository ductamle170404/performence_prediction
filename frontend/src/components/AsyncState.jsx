/** Hiển thị trạng thái tải/lỗi, hoặc render children khi đã có dữ liệu. */
export default function AsyncState({ state, children }) {
  if (state.loading) return <p className="muted">Đang tải dữ liệu…</p>;
  if (state.error) return <p className="error" role="alert">{state.error.message}</p>;
  return children(state.data);
}
