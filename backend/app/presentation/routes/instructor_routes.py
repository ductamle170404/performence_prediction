from typing import List, Optional

from fastapi import APIRouter, Depends

from app.dependencies import get_instructor_service, get_prediction_service, get_student_repository
from app.infrastructure.repositories.student_repository import StudentRepository
from app.presentation.schemas.course_schema import TeachingClassOut
from app.presentation.schemas.prediction_schema import ClassStudentOut
from app.services.instructor_service import InstructorService
from app.services.prediction_service import PredictionService

router = APIRouter(prefix="/api", tags=["instructors"])


@router.get("/instructors/{instructor_id}/courses", response_model=List[TeachingClassOut])
def get_teaching_courses(instructor_id: str, service: InstructorService = Depends(get_instructor_service)):
    return service.get_teaching_courses(instructor_id)


@router.get("/courses/{code_module}/{presentation}/classes/{class_id}/students", response_model=List[ClassStudentOut])
def get_class_students(code_module: str, presentation: str, class_id: str, instructor_id: str,
                       cutoff_date: Optional[int] = None, current_day: Optional[int] = None,
                       service: InstructorService = Depends(get_instructor_service),
                       predictions: PredictionService = Depends(get_prediction_service)):
    cutoff = predictions.resolve_cutoff(code_module, presentation, cutoff_date, current_day)
    return service.get_class_students(instructor_id, code_module, presentation, class_id, cutoff)


@router.get("/demo/users", tags=["demo"])
def list_demo_users(repo: StudentRepository = Depends(get_student_repository)):
    """Gợi ý ID để đăng nhập demo. Không phải cơ chế xác thực production."""
    return repo.list_demo_users()
