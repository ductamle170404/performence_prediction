"""Lưu/đọc kết quả dự báo trong một file CSV (prototype)."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from app.domain.models.prediction import Prediction
from app.infrastructure.adapters.csv_data_adapter import CsvDataAdapter

Key = Tuple[int, str, str, int]


class PredictionRepository:
    FILE_NAME = "predictions.csv"

    def __init__(self, adapter: CsvDataAdapter, output_dir: Path):
        self._adapter = adapter  # giữ để thay bằng DB repository sau này
        self._path = Path(output_dir) / self.FILE_NAME
        self._records: Dict[Key, Prediction] = {}
        self._load()

    def find_prediction(self, student_id: int, code_module: str, code_presentation: str, cutoff: int) -> Optional[Prediction]:
        return self._records.get((student_id, code_module, code_presentation, cutoff))

    def find_latest_prediction(self, student_id: int, code_module: str, code_presentation: str) -> Optional[Prediction]:
        history = self.find_student_history(student_id, code_module, code_presentation)
        return history[-1] if history else None

    def find_student_history(self, student_id: int, code_module: str, code_presentation: str) -> List[Prediction]:
        items = [p for p in self._records.values()
                 if p.student_id == student_id and p.code_module == code_module and p.code_presentation == code_presentation]
        return sorted(items, key=lambda p: p.cutoff_date)

    def find_prediction_history(self, code_module: str, code_presentation: str, class_id: str) -> List[Prediction]:
        items = [p for p in self._records.values()
                 if p.code_module == code_module and p.code_presentation == code_presentation and p.class_id == class_id]
        return sorted(items, key=lambda p: (p.cutoff_date, p.student_id))

    def save_predictions(self, records: List[Prediction]) -> None:
        for p in records:
            self._records[(p.student_id, p.code_module, p.code_presentation, p.cutoff_date)] = p
        self._flush()

    # --- internals --------------------------------------------------------
    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            df = pd.read_csv(self._path)
        except Exception:
            return
        for r in df.to_dict("records"):
            p = Prediction(
                student_id=int(r["student_id"]), code_module=str(r["code_module"]),
                code_presentation=str(r["code_presentation"]), class_id=str(r["class_id"]),
                cutoff_date=int(r["cutoff_date"]), n_assessments=int(r["n_assessments"]),
                fail_probability=float(r["fail_probability"]), risk_level=str(r["risk_level"]),
                model_version=str(r["model_version"]), generated_at=str(r["generated_at"]),
            )
            self._records[(p.student_id, p.code_module, p.code_presentation, p.cutoff_date)] = p

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([p.to_dict() for p in self._records.values()]).to_csv(self._path, index=False)
