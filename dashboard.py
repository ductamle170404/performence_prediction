r"""Local web dashboard for student predictions and model comparisons.

Run with:
    .\venv\Scripts\python.exe dashboard.py
    .\venv\Scripts\python.exe dashboard.py --export

The dashboard reads prediction artifacts from ./predict/ and does not rerun
training or modify the source datasets.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parent
PREDICT_DIR = ROOT / "predict"
DEFAULT_PORT = 8765
DEFAULT_PREDICTIONS = "predictions_all_models.csv"
MODELS = ("XGBoost", "RandomForest", "LogisticRegression")
PREDICTION_FIELDS = (
    "id_student",
    "code_module",
    "code_presentation",
    "progress_pct",
    "cutoff_day",
    "final_result",
    "actual_fail",
    "model",
    "probability_fail",
    "threshold",
    "predicted_positive",
)


def read_csv(path: Path, fields: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            converted: dict[str, Any] = {}
            for key, value in row.items():
                if key is None or (fields is not None and key not in fields):
                    continue
                if value is None:
                    converted[key] = ""
                    continue
                try:
                    number = float(value)
                    converted[key] = (
                        int(number) if number.is_integer() else number
                    ) if math.isfinite(number) else None
                except ValueError:
                    converted[key] = value
            rows.append(converted)
    return rows


def prediction_files() -> list[str]:
    return sorted(path.name for path in PREDICT_DIR.glob("predictions_*.csv"))


def collect_bootstrap() -> dict[str, Any]:
    evaluation_path = PREDICT_DIR / "evaluation.csv"
    module_path = PREDICT_DIR / "by_module_XGBoost.csv"
    manifest_path = PREDICT_DIR / "manifest.json"
    evaluation = read_csv(evaluation_path) if evaluation_path.exists() else []
    module_rows = read_csv(module_path) if module_path.exists() else []
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
    if not manifest.get("run_id"):
        latest = PREDICT_DIR / DEFAULT_PREDICTIONS
        if not latest.exists():
            latest = None
        if latest:
            stat = latest.stat()
            manifest = {
                "run_id": f"{stat.st_mtime_ns}-{stat.st_size}",
                "generated_at": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(
                    timespec="seconds"
                ),
            }
    return {
        "files": prediction_files(),
        "models": list(MODELS),
        "evaluation": evaluation,
        "modules": sorted({row["module"] for row in module_rows if row.get("module")}),
        "data_version": manifest.get("run_id", ""),
        "generated_at": manifest.get("generated_at", ""),
        "progress": sorted(
            {row["progress_pct"] for row in evaluation if row.get("progress_pct") is not None}
        ),
    }


def json_bytes(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/").encode("utf-8")


def read_explanations(model: str, progress: float) -> dict[str, Any]:
    if model not in MODELS or not 0 <= progress <= 100:
        raise ValueError("Invalid explanation model or progress.")
    suffix = f"{model}_p{progress:g}.csv"
    shap_path = PREDICT_DIR / "shap" / f"shap_importance_{suffix}"
    lime_path = PREDICT_DIR / "lime" / f"lime_{suffix}"
    return {
        "model": model,
        "progress": progress,
        "shap": read_csv(shap_path) if shap_path.exists() else [],
        "lime": read_csv(lime_path) if lime_path.exists() else [],
        "shap_available": shap_path.exists(),
        "lime_available": lime_path.exists(),
    }


def render_export() -> None:
    page_path = ROOT / "dashboard.html"
    if not page_path.exists():
        raise FileNotFoundError(f"Dashboard HTML not found: {page_path}")
    files = prediction_files()
    if DEFAULT_PREDICTIONS not in files:
        raise FileNotFoundError(
            f"Missing {PREDICT_DIR / DEFAULT_PREDICTIONS}; generate prediction CSVs first."
        )
    bootstrap = collect_bootstrap()
    exported_data = {
        "bootstrap": bootstrap,
        "predictions": read_csv(PREDICT_DIR / DEFAULT_PREDICTIONS, PREDICTION_FIELDS),
        "explanations": {
            f"{model}|{float(progress):g}": read_explanations(model, float(progress))
            for model in MODELS
            for progress in bootstrap["progress"]
        },
    }
    page = page_path.read_text(encoding="utf-8")
    payload = json.dumps(exported_data, ensure_ascii=False).replace("</", "<\\/")
    page = page.replace(
        "</head>", f"<script>window.EXPORTED_DATA={payload};</script>\n</head>", 1
    )
    page_path.write_text(page, encoding="utf-8")
    print(f"Standalone dashboard exported to: {page_path}")


class DashboardHandler(BaseHTTPRequestHandler):
    def respond(self, status: int, content: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        request = urlparse(self.path)
        if request.path in ("/", "/index.html"):
            page = (ROOT / "dashboard.html").read_bytes()
            return self.respond(200, page, "text/html; charset=utf-8")

        try:
            if request.path == "/api/bootstrap":
                return self.respond(200, json_bytes(collect_bootstrap()), "application/json")
            if request.path == "/api/predictions":
                file_name = parse_qs(request.query).get("file", [DEFAULT_PREDICTIONS])[0]
                if file_name not in prediction_files():
                    return self.respond(
                        404, json_bytes({"error": "Không tìm thấy file dự đoán."}),
                        "application/json",
                    )
                rows = read_csv(PREDICT_DIR / file_name, PREDICTION_FIELDS)
                return self.respond(200, json_bytes(rows), "application/json")
            if request.path == "/api/explanations":
                query = parse_qs(request.query)
                model = query.get("model", [""])[0]
                progress = float(query.get("progress", ["-1"])[0])
                return self.respond(
                    200, json_bytes(read_explanations(model, progress)), "application/json"
                )
        except ValueError as error:
            return self.respond(
                400, json_bytes({"error": f"Tham số giải thích không hợp lệ: {error}"}),
                "application/json",
            )
        except OSError as error:
            return self.respond(
                500, json_bytes({"error": f"Không đọc được dữ liệu: {error}"}),
                "application/json",
            )

        self.send_error(404)

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Mở dashboard kết quả dự đoán.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--export", action="store_true",
        help="Xuất dashboard.html độc lập với dữ liệu predictions_all_models.csv.",
    )
    args = parser.parse_args()
    if args.export:
        render_export()
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
