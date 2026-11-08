from typing import Dict, List

from app.domain.policies.risk_policy import RiskPolicy
from app.infrastructure.repositories.prediction_repository import PredictionRepository
from app.infrastructure.repositories.student_repository import StudentRepository
from app.services.prediction_service import PredictionService


class RiskAggregationService:
    def __init__(self, prediction_service: PredictionService, prediction_repo: PredictionRepository,
                 student_repo: StudentRepository, risk_policy: RiskPolicy):
        self._predictions = prediction_service
        self._prediction_repo = prediction_repo
        self._students = student_repo
        self._policy = risk_policy

    def _summary(self, predictions, cutoff: int) -> Dict:
        total = len(predictions)
        at_risk = sum(1 for p in predictions if self._policy.is_at_risk(p.fail_probability))
        avg = sum(p.fail_probability for p in predictions) / total if total else 0.0
        return {
            "total_students": total,
            "at_risk_students": at_risk,
            "at_risk_rate": round(at_risk / total, 4) if total else 0.0,  # tránh chia cho 0
            "avg_fail_probability": round(avg, 4),
            "cutoff_date": cutoff,
        }

    def calculate_class_risk(self, code_module: str, code_presentation: str, class_id: str, cutoff: int) -> Dict:
        preds = self._predictions.predict_class(code_module, code_presentation, class_id, cutoff)
        return {"code_module": code_module, "code_presentation": code_presentation,
                "class_id": class_id, **self._summary(preds, cutoff)}

    def calculate_course_risk(self, code_module: str, code_presentation: str, cutoff: int) -> Dict:
        all_preds, classes = [], []
        for class_id in self._students.find_classes(code_module, code_presentation):
            preds = self._predictions.predict_class(code_module, code_presentation, class_id, cutoff)
            all_preds.extend(preds)
            classes.append({"class_id": class_id, **self._summary(preds, cutoff)})
        return {"code_module": code_module, "code_presentation": code_presentation,
                **self._summary(all_preds, cutoff), "classes": classes}

    def build_risk_trend(self, code_module: str, code_presentation: str, class_id: str, max_day: int) -> List[Dict]:
        trend = []
        for day in self._predictions.get_milestones(code_module, code_presentation, max_day):
            preds = self._predictions.predict_class(code_module, code_presentation, class_id, day)
            trend.append(self._summary(preds, day))
        return trend
