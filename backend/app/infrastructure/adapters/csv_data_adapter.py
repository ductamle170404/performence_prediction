"""Đọc CSV (prototype). Thay bằng LMS/DB adapter khi triển khai thật."""
from pathlib import Path
from typing import Dict, List

import pandas as pd

from app.core.exceptions import DataSchemaError


class CsvDataAdapter:
    # Tên logic -> các tên file chấp nhận (không phân biệt hoa thường).
    FILE_ALIASES: Dict[str, List[str]] = {
        "assignments": ["student_class_assignment.csv"],
        "student_assessment": ["student_assessment.csv", "studentassetment.csv", "studentassessment.csv"],
        "class_instructor": ["class_instructor.csv"],
        "courses": ["courses.csv"],
        "assessments": ["assessments.csv"],
    }
    REQUIRED_COLUMNS: Dict[str, List[str]] = {
        "assignments": ["code_module", "code_presentation", "id_student", "class_id"],
        "student_assessment": ["id_assessment", "id_student", "date_submitted", "is_banked", "score"],
        "class_instructor": ["code_module", "code_presentation", "class_id", "instructor_id"],
        "courses": ["code_module", "code_presentation", "module_presentation_length"],
        "assessments": ["code_module", "code_presentation", "id_assessment", "assessment_type", "date", "weight"],
    }
    NUMERIC_COLUMNS = [
        "id_student", "id_assessment", "date_submitted", "score", "date",
        "weight", "module_presentation_length", "date_registration", "registration_rank",
    ]
    ID_COLUMNS = {"id_student", "id_assessment"}

    def __init__(self, data_dir: Path):
        self._data_dir = Path(data_dir)
        self._cache: Dict[str, pd.DataFrame] = {}

    # --- file resolving -------------------------------------------------
    def _resolve_path(self, key: str) -> Path:
        if not self._data_dir.exists():
            raise DataSchemaError(f"Không tìm thấy thư mục dữ liệu: {self._data_dir}")
        by_lower = {p.name.lower(): p for p in self._data_dir.iterdir() if p.is_file()}
        for alias in self.FILE_ALIASES[key]:
            if alias in by_lower:
                return by_lower[alias]
        raise DataSchemaError(
            f"Thiếu file CSV '{self.FILE_ALIASES[key][0]}' trong {self._data_dir}"
        )

    # --- public API -----------------------------------------------------
    def read_csv(self, key: str) -> pd.DataFrame:
        """Trả về DataFrame (cache). Không chỉnh sửa DataFrame trả về."""
        if key not in self._cache:
            df = pd.read_csv(self._resolve_path(key), skipinitialspace=True)
            df.columns = [c.strip() for c in df.columns]
            self._check_columns(key, df)
            self._cache[key] = self._normalize(df)
        return self._cache[key]

    def validate_schema(self) -> List[str]:
        """Kiểm tra mọi file; ném DataSchemaError nếu có lỗi."""
        for key in self.FILE_ALIASES:
            self.read_csv(key)
        return list(self.FILE_ALIASES)

    def clear_cache(self) -> None:
        self._cache.clear()

    # --- helpers ----------------------------------------------------------
    def _check_columns(self, key: str, df: pd.DataFrame) -> None:
        missing = [c for c in self.REQUIRED_COLUMNS[key] if c not in df.columns]
        if missing:
            raise DataSchemaError(f"File '{key}' thiếu cột: {', '.join(missing)}")

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in ("code_module", "code_presentation", "class_id", "instructor_id", "assessment_type"):
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        for col in self.NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in self.ID_COLUMNS & set(df.columns):
            df = df.dropna(subset=[col])
            df[col] = df[col].astype(int)
        if "is_banked" in df.columns:
            df["is_banked"] = pd.to_numeric(df["is_banked"], errors="coerce").fillna(0).astype(int)
        return df.reset_index(drop=True)
