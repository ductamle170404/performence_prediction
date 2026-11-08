"""Dependency wiring: khởi tạo singleton và cung cấp cho FastAPI Depends."""
from functools import lru_cache

from app.core.config import Settings, load_settings
from app.domain.policies.risk_policy import RiskPolicy
from app.infrastructure.adapters.csv_data_adapter import CsvDataAdapter
from app.infrastructure.adapters.prediction_engine import HeuristicPredictionEngine, PredictionEngine
from app.infrastructure.repositories.assessment_repository import AssessmentRepository
from app.infrastructure.repositories.course_repository import CourseRepository
from app.infrastructure.repositories.prediction_repository import PredictionRepository
from app.infrastructure.repositories.student_repository import StudentRepository
from app.services.assessment_service import AssessmentService
from app.services.course_service import CourseService
from app.services.instructor_service import InstructorService
from app.services.prediction_service import PredictionService
from app.services.risk_aggregation_service import RiskAggregationService


@lru_cache
def get_settings() -> Settings:
    return load_settings()


@lru_cache
def get_adapter() -> CsvDataAdapter:
    return CsvDataAdapter(get_settings().data_dir)


@lru_cache
def get_course_repository() -> CourseRepository:
    return CourseRepository(get_adapter())


@lru_cache
def get_student_repository() -> StudentRepository:
    return StudentRepository(get_adapter())


@lru_cache
def get_assessment_repository() -> AssessmentRepository:
    return AssessmentRepository(get_adapter())


@lru_cache
def get_prediction_repository() -> PredictionRepository:
    return PredictionRepository(get_adapter(), get_settings().prediction_output_dir)


@lru_cache
def get_risk_policy() -> RiskPolicy:
    s = get_settings()
    return RiskPolicy(s.medium_risk_threshold, s.high_risk_threshold, s.at_risk_threshold)


@lru_cache
def get_prediction_engine() -> PredictionEngine:
    # Thay bằng engine dùng model thật khi tích hợp thư mục `prediction/`.
    return HeuristicPredictionEngine(get_settings().fail_score_threshold)


@lru_cache
def get_assessment_service() -> AssessmentService:
    return AssessmentService(get_assessment_repository(), get_course_repository())


@lru_cache
def get_course_service() -> CourseService:
    return CourseService(get_course_repository(), get_assessment_service(), get_settings().simulated_current_day)


@lru_cache
def get_prediction_service() -> PredictionService:
    return PredictionService(
        get_course_repository(), get_student_repository(), get_assessment_repository(),
        get_prediction_repository(), get_prediction_engine(), get_risk_policy(),
        get_settings().simulated_current_day,
    )


@lru_cache
def get_risk_aggregation_service() -> RiskAggregationService:
    return RiskAggregationService(
        get_prediction_service(), get_prediction_repository(), get_student_repository(), get_risk_policy()
    )


@lru_cache
def get_instructor_service() -> InstructorService:
    return InstructorService(get_student_repository(), get_course_repository(), get_prediction_service())
