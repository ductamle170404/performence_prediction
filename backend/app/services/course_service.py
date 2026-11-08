from typing import Dict, List, Optional

from app.core.exceptions import NotFoundError
from app.infrastructure.repositories.course_repository import CourseRepository
from app.services.assessment_service import AssessmentService


class CourseService:
    def __init__(self, course_repo: CourseRepository, assessment_service: AssessmentService, default_current_day: Optional[int]):
        self._courses = course_repo
        self._assessment_service = assessment_service
        self._default_day = default_current_day

    def _day(self, current_day: Optional[int]) -> Optional[int]:
        return current_day if current_day is not None else self._default_day

    def get_student_courses(self, student_id: int, current_day: Optional[int] = None) -> List[Dict]:
        day = self._day(current_day)
        pairs = self._courses.find_student_courses(student_id)
        if not pairs:
            raise NotFoundError(f"Sinh viên {student_id} chưa đăng ký course nào")
        return [
            {
                "code_module": course.code_module,
                "code_presentation": course.code_presentation,
                "class_id": student.class_id,
                "length": course.length,
                "is_ongoing": course.is_ongoing(day),
            }
            for student, course in pairs
        ]

    def get_course_timeline(self, code_module: str, code_presentation: str, current_day: Optional[int] = None) -> Dict:
        day = self._day(current_day)
        course = self._courses.find_course(code_module, code_presentation)
        if course is None:
            raise NotFoundError(f"Không tìm thấy course {code_module}/{code_presentation}")
        assessments = self._assessment_service.get_assessments(code_module, code_presentation, day)
        return {
            "code_module": code_module,
            "code_presentation": code_presentation,
            "length": course.length,
            "is_ongoing": course.is_ongoing(day),
            "current_day": day if course.is_ongoing(day) else None,
            "assessments": assessments,
        }
