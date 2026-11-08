"""PredictionEngine: tạo feature theo cutoff và tính xác suất điểm liệt.

HeuristicPredictionEngine là mock inference để demo end-to-end. Khi tích hợp
pipeline thật (thư mục `prediction/`), tạo lớp mới kế thừa PredictionEngine,
nạp model artifact đã train rồi thay trong dependencies.py. Không train model
trong request HTTP.
"""
import math
from abc import ABC, abstractmethod
from typing import Dict, List

from app.domain.models.assessment import Assessment, StudentScore


class PredictionEngine(ABC):
    model_version: str

    @abstractmethod
    def build_features(
        self, scores: List[StudentScore], assessments: List[Assessment], cutoff: int
    ) -> Dict[str, float]:
        ...

    @abstractmethod
    def predict_probability(self, features: Dict[str, float]) -> float:
        ...

    @abstractmethod
    def explain_prediction(self, features: Dict[str, float]) -> List[Dict[str, float]]:
        ...


class HeuristicPredictionEngine(PredictionEngine):
    model_version = "heuristic-v0"

    def __init__(self, fail_score_threshold: float, prior_probability: float = 0.3):
        self.fail_score_threshold = fail_score_threshold
        self.prior_probability = prior_probability

    def build_features(self, scores, assessments, cutoff):
        # Chỉ dùng dữ liệu có mặt tại cutoff để tránh data leakage.
        due = [a for a in assessments if a.date <= cutoff]
        known = {
            s.id_assessment: s
            for s in scores
            if s.score is not None and s.date_submitted is not None and s.date_submitted <= cutoff
        }
        total_weight = sum(a.weight or 1.0 for a in due)
        if not due or total_weight == 0:
            return {"n_due": 0, "n_scored": len(known), "weighted_mean": -1.0, "missing_ratio": 0.0}
        weighted = sum((a.weight or 1.0) * known[a.id_assessment].score for a in due if a.id_assessment in known)
        missing = [a for a in due if a.id_assessment not in known]
        return {
            "n_due": len(due),
            "n_scored": len(known),
            # Assessment đến hạn nhưng chưa có điểm được tính là 0.
            "weighted_mean": weighted / total_weight,
            "missing_ratio": len(missing) / len(due),
        }

    def predict_probability(self, features):
        if features["n_due"] == 0:
            return self.prior_probability
        margin = features["weighted_mean"] - (self.fail_score_threshold + 15)
        return 1.0 / (1.0 + math.exp(margin / 10.0))

    def explain_prediction(self, features):
        return [
            {"feature": "weighted_mean", "value": features["weighted_mean"]},
            {"feature": "missing_ratio", "value": features["missing_ratio"]},
        ]
