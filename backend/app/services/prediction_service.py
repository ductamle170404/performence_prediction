from datetime import datetime
from typing import List, Optional

from app.core.exceptions import NotFoundError, ValidationAppError
from app.domain.models.prediction import Prediction
from app.domain.policies.risk_policy import RiskPolicy
from app.infrastructure.adapters.prediction_engine import PredictionEngine
from app.infrastructure.repositories.assessment_repository import AssessmentRepository
from app.infrastructure.repositories.course_repository import CourseRepository
from app.infrastructure.repositories.prediction_repository import PredictionRepository
from app.infrastructure.repositories.student_repository import StudentRepository


class PredictionService:
    def __init__(
        self,
        course_repo: CourseRepository,
        student_repo: StudentRepository,
        assessment_repo: AssessmentRepository,
        prediction_repo: PredictionRepository,
        engine: PredictionEngine,
        risk_policy: RiskPolicy,
        default_current_day: Optional[int],
    ):
        self._courses = course_repo
        self._students = student_repo
        self._assessments = assessment_repo
        self._predictions = prediction_repo
        self._engine = engine
        self._policy = risk_policy
        self._default_day = default_current_day

    # --- cutoff helpers -----------------------------------------------------
    def resolve_cutoff(self, code_module: str, code_presentation: str,
                       cutoff_date: Optional[int] = None, current_day: Optional[int] = None) -> int:
        """cutoff_date > current_day (hoặc ngày giả lập) > cuối học kỳ."""
        course = self._courses.find_course(code_module, code_presentation)
        if course is None:
            raise NotFoundError(f"Không tìm thấy course {code_module}/{code_presentation}")
        if cutoff_date is not None:
            if cutoff_date < 0:
                raise ValidationAppError("cutoff_date phải >= 0")
            return min(cutoff_date, course.length)
        day = current_day if current_day is not None else self._default_day
        return min(day, course.length) if course.is_ongoing(day) else course.length

    def get_milestones(self, code_module: str, code_presentation: str, max_day: int) -> List[int]:
        """Các mốc dự báo: ngày của từng assessment đến max_day."""
        days = sorted({a.date for a in self._assessments.find_assessments(code_module, code_presentation) if 0 <= a.date <= max_day})
        return days or [max_day]

    # --- prediction -----------------------------------------------------------
    def predict_student_risk(self, student_id: int, code_module: str, code_presentation: str, cutoff: int) -> Prediction:
        student = self._students.find_student(student_id, code_module, code_presentation)
        if student is None:
            raise NotFoundError(f"Sinh viên {student_id} không học {code_module}/{code_presentation}")
        cached = self._predictions.find_prediction(student_id, code_module, code_presentation, cutoff)
        if cached and cached.model_version == self._engine.model_version:
            return cached
        prediction = self._compute(student_id, student.class_id, code_module, code_presentation, cutoff)
        self._predictions.save_predictions([prediction])
        return prediction

    def predict_class(self, code_module: str, code_presentation: str, class_id: str, cutoff: int) -> List[Prediction]:
        students = self._students.find_students_by_class(code_module, code_presentation, class_id)
        results, fresh = [], []
        for s in students:
            cached = self._predictions.find_prediction(s.id_student, code_module, code_presentation, cutoff)
            if cached and cached.model_version == self._engine.model_version:
                results.append(cached)
                continue
            p = self._compute(s.id_student, class_id, code_module, code_presentation, cutoff)
            results.append(p)
            fresh.append(p)
        if fresh:
            self._predictions.save_predictions(fresh)
        return results

    def get_prediction_history(self, student_id: int, code_module: str, code_presentation: str,
                               max_day: int) -> List[Prediction]:
        student = self._students.find_student(student_id, code_module, code_presentation)
        if student is None:
            raise NotFoundError(f"Sinh viên {student_id} không học {code_module}/{code_presentation}")
        return [self.predict_student_risk(student_id, code_module, code_presentation, day)
                for day in self.get_milestones(code_module, code_presentation, max_day)]

    def explain(self, student_id: int, code_module: str, code_presentation: str, cutoff: int):
        features = self._features(student_id, code_module, code_presentation, cutoff)
        return self._engine.explain_prediction(features)

    # --- internals --------------------------------------------------------------
    def _features(self, student_id, code_module, code_presentation, cutoff):
        scores = self._assessments.find_student_scores(student_id, code_module, code_presentation)
        assessments = self._assessments.find_assessments(code_module, code_presentation)
        return self._engine.build_features(scores, assessments, cutoff)

    def _compute(self, student_id, class_id, code_module, code_presentation, cutoff) -> Prediction:
        features = self._features(student_id, code_module, code_presentation, cutoff)
        probability = round(float(self._engine.predict_probability(features)), 4)
        return Prediction(
            student_id=student_id,
            code_module=code_module,
            code_presentation=code_presentation,
            class_id=class_id,
            cutoff_date=cutoff,
            n_assessments=int(features["n_scored"]),
            fail_probability=probability,
            risk_level=self._policy.classify(probability),
            model_version=self._engine.model_version,
            generated_at=datetime.now().isoformat(timespec="seconds"),
        )
