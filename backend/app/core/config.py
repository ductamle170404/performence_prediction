"""Đọc cấu hình từ biến môi trường (.env ở thư mục gốc dự án)."""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[2]  # .../backend
PROJECT_DIR = BACKEND_DIR.parent

load_dotenv(PROJECT_DIR / ".env")
load_dotenv(BACKEND_DIR / ".env")


def _resolve_dir(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (BACKEND_DIR / path).resolve()


def _optional_int(name: str) -> Optional[int]:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else None


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    prediction_output_dir: Path
    cors_origins: List[str]
    # Ngưỡng điểm liệt: phải cấu hình theo quy chế môn/trường.
    fail_score_threshold: float
    # Xác suất tối thiểu để coi sinh viên là "có nguy cơ".
    at_risk_threshold: float
    medium_risk_threshold: float
    high_risk_threshold: float
    # Ngày hiện tại (tính theo ngày tương đối so với đầu kỳ) dùng để giả lập
    # học kỳ đang diễn ra. Để trống = mọi học kỳ đều đã kết thúc.
    simulated_current_day: Optional[int]


def load_settings() -> Settings:
    origins = os.getenv("CORS_ORIGINS", "http://localhost:5173")
    return Settings(
        data_dir=_resolve_dir(os.getenv("DATA_DIR", "../database/raw")),
        prediction_output_dir=_resolve_dir(
            os.getenv("PREDICTION_OUTPUT_DIR", "../database/predictions")
        ),
        cors_origins=[o.strip() for o in origins.split(",") if o.strip()],
        fail_score_threshold=float(os.getenv("FAIL_SCORE_THRESHOLD", "40")),
        at_risk_threshold=float(os.getenv("AT_RISK_THRESHOLD", "0.5")),
        medium_risk_threshold=float(os.getenv("MEDIUM_RISK_THRESHOLD", "0.3")),
        high_risk_threshold=float(os.getenv("HIGH_RISK_THRESHOLD", "0.7")),
        simulated_current_day=_optional_int("SIMULATED_CURRENT_DAY"),
    )
