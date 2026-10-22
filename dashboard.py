"""Dashboard cục bộ cho các kết quả huấn luyện.

Chạy:
    .\\venv\\Scripts\\python.exe dashboard.py
    .\\venv\\Scripts\\python.exe dashboard.py --export

Dashboard chỉ đọc các file CSV trong results_cls/ và results/, không chạy lại
huấn luyện và không thay đổi bất kỳ file logic nào của project.
"""
from __future__ import annotations

import argparse
import csv
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_PORT = 8765


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows: list[dict[str, Any]] = []
        for row in csv.DictReader(file):
            converted: dict[str, Any] = {}
            for key, value in row.items():
                if key is None or not key.strip():
                    key = "feature"
                if value is None:
                    converted[key] = ""
                    continue
                try:
                    converted[key] = float(value)
                    if converted[key].is_integer():
                        converted[key] = int(converted[key])
                except ValueError:
                    converted[key] = value
            rows.append(converted)
        return rows


def find_first(*paths: Path) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def collect_data() -> dict[str, Any]:
    cls = find_first(
        ROOT / "results_cls" / "classification_at_risk.csv",
        ROOT / "results" / "classification.csv",
    )
    module = find_first(
        ROOT / "results_cls" / "by_module_at_risk.csv",
        ROOT / "results" / "by_module.csv",
    )
    importance_files = sorted(
        list((ROOT / "results_cls").glob("importance_*.csv"))
        + list((ROOT / "results").glob("importance_*.csv"))
    )
    importance: dict[str, list[dict[str, Any]]] = {}
    for path in importance_files:
        importance[path.stem] = read_csv(path)
    return {
        "classification": read_csv(cls) if cls else [],
        "module": read_csv(module) if module else [],
        "importance": importance,
        "source": str(cls.relative_to(ROOT)) if cls else "",
    }


def render(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Performance Prediction Dashboard</title>
  <style>
    :root {{
      --bg: #f4f7fb; --card: #fff; --ink: #172033; --muted: #64748b;
      --primary: #2563eb; --primary-light: #dbeafe; --line: #e5e7eb;
      --green: #059669; --orange: #d97706; --shadow: 0 8px 24px #17203312;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink);
      font: 14px/1.5 Segoe UI, Arial, sans-serif; }}
    .wrap {{ max-width: 1320px; margin: auto; padding: 28px; }}
    header {{ display:flex; justify-content:space-between; gap:18px; align-items:flex-start;
      margin-bottom:24px; }}
    h1 {{ margin:0 0 5px; font-size:30px; letter-spacing:-.5px; }}
    h2 {{ margin:0 0 16px; font-size:18px; }}
    .subtitle, .muted {{ color:var(--muted); }}
    .badge {{ background:var(--primary-light); color:var(--primary); border-radius:999px;
      padding:7px 12px; white-space:nowrap; font-weight:600; }}
    .grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-bottom:18px; }}
    .card {{ background:var(--card); border:1px solid var(--line); border-radius:14px;
      padding:20px; box-shadow:var(--shadow); }}
    .kpi-label {{ color:var(--muted); font-size:12px; text-transform:uppercase;
      letter-spacing:.6px; }}
    .kpi-value {{ margin-top:5px; font-size:28px; font-weight:700; color:var(--primary); }}
    .layout {{ display:grid; grid-template-columns:1.1fr .9fr; gap:18px; margin-bottom:18px; }}
    .table-wrap {{ overflow:auto; max-height:510px; }}
    table {{ width:100%; border-collapse:collapse; white-space:nowrap; }}
    th, td {{ padding:10px 11px; border-bottom:1px solid var(--line); text-align:right; }}
    th {{ position:sticky; top:0; background:#f8fafc; color:var(--muted); font-size:12px; }}
    th:first-child, td:first-child {{ text-align:left; }}
    tr:hover td {{ background:#f8fbff; }}
    .bar-row {{ display:grid; grid-template-columns:180px 1fr 55px; align-items:center;
      gap:10px; margin:13px 0; }}
    .bar-label {{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
    .bar {{ height:12px; background:var(--primary-light); border-radius:8px; overflow:hidden; }}
    .bar > i {{ display:block; height:100%; background:linear-gradient(90deg,#2563eb,#60a5fa);
      border-radius:8px; }}
    select {{ border:1px solid var(--line); border-radius:8px; padding:8px 10px;
      background:#fff; color:var(--ink); margin-bottom:12px; }}
    .empty {{ padding:35px; text-align:center; color:var(--muted); }}
    @media (max-width: 900px) {{ .grid, .layout {{ grid-template-columns:1fr 1fr; }} }}
    @media (max-width: 620px) {{ .wrap {{ padding:16px; }} header, .grid, .layout {{ display:block; }}
      .card {{ margin-bottom:16px; }} .badge {{ display:inline-block; margin-top:12px; }} }}
  </style>
</head>
<body>
<main class="wrap">
  <header>
    <div><h1>Performance Prediction</h1>
      <div class="subtitle">Bảng điều khiển kết quả mô hình cảnh báo sớm</div></div>
    <div class="badge" id="source"></div>
  </header>
  <section class="grid" id="kpis"></section>
  <section class="layout">
    <div class="card"><h2>So sánh mô hình</h2><div class="table-wrap" id="model-table"></div></div>
    <div class="card"><h2>AUC XGBoost theo môn</h2><div id="module-chart"></div></div>
  </section>
  <section class="card"><h2>Feature importance</h2>
    <select id="importance-select"></select><div id="importance-chart"></div>
  </section>
</main>
<script>
const DATA = {payload};
const cls = DATA.classification || [], modules = DATA.module || [];
const fmt = (v, digits=3) => typeof v === "number" ? v.toFixed(digits) : (v ?? "—");
const pct = v => typeof v === "number" ? (v * 100).toFixed(1) + "%" : "—";
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}}[c]));
document.getElementById("source").textContent = DATA.source ? "Nguồn: " + DATA.source : "Chưa có kết quả";

function card(label, value) {{
  return `<div class="card"><div class="kpi-label">${{label}}</div><div class="kpi-value">${{value}}</div></div>`;
}}
if (cls.length) {{
  const best = [...cls].sort((a,b) => (b.auc || 0) - (a.auc || 0))[0];
  const avgRisk = cls.reduce((sum, row) => sum + (row.recall || 0), 0) / cls.length;
  document.getElementById("kpis").innerHTML =
    card("Số dòng đánh giá", cls.length) +
    card("AUC tốt nhất", fmt(best.auc)) +
    card("Mô hình nổi bật", esc(best.model)) +
    card("Recall trung bình", pct(avgRisk));
  const cols = ["cutoff","model","auc","pr_auc","f1","recall","precision","accuracy"];
  let table = "<table><thead><tr>" + cols.map(c => `<th>${{esc(c)}}</th>`).join("") +
    "</tr></thead><tbody>";
  table += cls.map(row => "<tr>" + cols.map(c => `<td>${{c === "model" ? esc(row[c]) :
    c === "cutoff" ? "Ngày " + esc(row[c]) : fmt(row[c])}}</td>`).join("") + "</tr>").join("");
  document.getElementById("model-table").innerHTML = table + "</tbody></table>";
}} else {{
  document.getElementById("kpis").innerHTML =
    '<div class="card empty" style="grid-column:1/-1">Chưa tìm thấy CSV kết quả. Hãy chạy train_cls.py trước.</div>';
  document.getElementById("model-table").innerHTML = '<div class="empty">Chưa có dữ liệu.</div>';
}}

function bars(container, rows, labelKey, valueKey, maxValue=1) {{
  if (!rows.length) {{ container.innerHTML = '<div class="empty">Chưa có dữ liệu.</div>'; return; }}
  container.innerHTML = rows.map(row => {{
    const value = Math.max(0, Number(row[valueKey]) || 0);
    return `<div class="bar-row"><span class="bar-label">${{esc(row[labelKey])}}</span>
      <span class="bar"><i style="width:${{Math.min(value/maxValue*100,100)}}%"></i></span>
      <strong>${{fmt(value)}}</strong></div>`;
  }}).join("");
}}
const moduleRows = modules.filter(r => r.module).sort((a,b) => (b.auc||0)-(a.auc||0));
bars(document.getElementById("module-chart"), moduleRows, "module", "auc");

const select = document.getElementById("importance-select");
const importanceKeys = Object.keys(DATA.importance || {{}});
if (!importanceKeys.length) {{
  select.remove(); document.getElementById("importance-chart").innerHTML =
    '<div class="empty">Chưa có file feature importance.</div>';
}} else {{
  select.innerHTML = importanceKeys.map(k => `<option value="${{esc(k)}}">${{esc(k)}}</option>`).join("");
  function showImportance() {{
    const rows = (DATA.importance[select.value] || []).slice(0, 20)
      .sort((a,b) => (b.importance||0)-(a.importance||0));
    bars(document.getElementById("importance-chart"), rows, "feature", "importance",
      Math.max(...rows.map(r => Number(r.importance)||0), 1));
  }}
  select.addEventListener("change", showImportance); showImportance();
}}
</script>
</body>
</html>"""


def write_dashboard(path: Path) -> None:
    path.write_text(render(collect_data()), encoding="utf-8")
    print(f"Dashboard exported to: {path}")


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path not in ("/", "/index.html"):
            self.send_error(404)
            return
        content = render(collect_data()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Mở dashboard kết quả huấn luyện.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--export", action="store_true",
                        help="Xuất dashboard.html thay vì mở server.")
    args = parser.parse_args()
    if args.export:
        write_dashboard(ROOT / "dashboard.html")
        return

    server = ThreadingHTTPServer(("127.0.0.1", args.port), DashboardHandler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Dashboard running at {url} (Ctrl+C to stop)")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
