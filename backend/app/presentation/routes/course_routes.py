from typing import List, Optional

from fastapi import APIRouter, Depends

from app.dependencies import get_course_service
from app.presentation.schemas.course_schema import StudentCourseOut, TimelineOut
from app.services.course_service import CourseService

router = APIRouter(prefix="/api", tags=["courses"])


@router.get("/students/{student_id}/courses", response_model=List[StudentCourseOut])
def get_student_courses(student_id: int, current_day: Optional[int] = None,
                        service: CourseService = Depends(get_course_service)):
    return service.get_student_courses(student_id, current_day)


@router.get("/courses/{code_module}/{presentation}/assessments", response_model=TimelineOut)
def get_assessments(code_module: str, presentation: str, current_day: Optional[int] = None,
                    service: CourseService = Depends(get_course_service)):
    return service.get_course_timeline(code_module, presentation, current_day)
