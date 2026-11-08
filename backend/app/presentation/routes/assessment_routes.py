from fastapi import APIRouter, Depends

from app.dependencies import get_assessment_service
from app.services.assessment_service import AssessmentService

router = APIRouter(prefix="/api", tags=["assessments"])


@router.get("/students/{student_id}/courses/{code_module}/{presentation}/completed-count")
def get_completed_count(student_id: int, code_module: str, presentation: str, cutoff_date: int,
                        service: AssessmentService = Depends(get_assessment_service)):
    return {
        "student_id": student_id,
        "cutoff_date": cutoff_date,
        "n_assessments": service.get_completed_assessment_count(student_id, code_module, presentation, cutoff_date),
    }
