from typing import Optional

from fastapi import APIRouter, Depends

from app.dependencies import (get_instructor_service, get_prediction_service,
                              get_risk_aggregation_service)
from app.presentation.schemas.prediction_schema import (ClassRiskOut, CourseRiskOut,
                                                        PredictionHistoryOut, PredictionOut,
                                                        RiskTrendPointOut)
from app.services.instructor_service import InstructorService
from app.services.prediction_service import PredictionService
from app.services.risk_aggregation_service import RiskAggregationService

router = APIRouter(prefix="/api", tags=["predictions"])


@router.get("/students/{student_id}/courses/{code_module}/{presentation}/prediction", response_model=PredictionOut)
def get_student_prediction(student_id: int, code_module: str, presentation: str,
                           cutoff_date: Optional[int] = None, current_day: Optional[int] = None,
                           service: PredictionService = Depends(get_prediction_service)):
    cutoff = service.resolve_cutoff(code_module, presentation, cutoff_date, current_day)
    return service.predict_student_risk(student_id, code_module, presentation, cutoff).to_dict()


@router.get("/students/{student_id}/courses/{code_module}/{presentation}/prediction-history",
            response_model=PredictionHistoryOut)
def get_student_prediction_history(student_id: int, code_module: str, presentation: str,
                                   current_day: Optional[int] = None,
                                   service: PredictionService = Depends(get_prediction_service)):
    max_day = service.resolve_cutoff(code_module, presentation, None, current_day)
    history = service.get_prediction_history(student_id, code_module, presentation, max_day)
    return {"student_id": student_id, "code_module": code_module, "code_presentation": presentation,
            "history": [p.to_dict() for p in history]}


@router.get("/courses/{code_module}/{presentation}/risk", response_model=CourseRiskOut)
def get_course_risk(code_module: str, presentation: str, instructor_id: str,
                    cutoff_date: Optional[int] = None, current_day: Optional[int] = None,
                    predictions: PredictionService = Depends(get_prediction_service),
                    aggregation: RiskAggregationService = Depends(get_risk_aggregation_service),
                    instructors: InstructorService = Depends(get_instructor_service)):
    instructors.assert_can_view_course(instructor_id, code_module, presentation)
    cutoff = predictions.resolve_cutoff(code_module, presentation, cutoff_date, current_day)
    return aggregation.calculate_course_risk(code_module, presentation, cutoff)


@router.get("/courses/{code_module}/{presentation}/classes/{class_id}/risk", response_model=ClassRiskOut)
def get_class_risk(code_module: str, presentation: str, class_id: str, instructor_id: str,
                   cutoff_date: Optional[int] = None, current_day: Optional[int] = None,
                   predictions: PredictionService = Depends(get_prediction_service),
                   aggregation: RiskAggregationService = Depends(get_risk_aggregation_service),
                   instructors: InstructorService = Depends(get_instructor_service)):
    instructors.assert_can_view_class(instructor_id, code_module, presentation, class_id)
    cutoff = predictions.resolve_cutoff(code_module, presentation, cutoff_date, current_day)
    return aggregation.calculate_class_risk(code_module, presentation, class_id, cutoff)


@router.get("/courses/{code_module}/{presentation}/classes/{class_id}/risk-history",
            response_model=list[RiskTrendPointOut])
def get_risk_history(code_module: str, presentation: str, class_id: str, instructor_id: str,
                     current_day: Optional[int] = None,
                     predictions: PredictionService = Depends(get_prediction_service),
                     aggregation: RiskAggregationService = Depends(get_risk_aggregation_service),
                     instructors: InstructorService = Depends(get_instructor_service)):
    instructors.assert_can_view_class(instructor_id, code_module, presentation, class_id)
    max_day = predictions.resolve_cutoff(code_module, presentation, None, current_day)
    return aggregation.build_risk_trend(code_module, presentation, class_id, max_day)
