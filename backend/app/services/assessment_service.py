from typing import List, Optional

from app.core.exceptions import NotFoundError
from app.domain.models.assessment import Assessment
from app.infrastructure.repositories.assessment_repository import AssessmentRepository
from app.infrastructure.repositories.course_repository import CourseRepository


class AssessmentService:
    def __init__(self, assessment_repo: AssessmentRepository, course_repo: CourseRepository):
        self._assessments = assessment_repo
        self._courses = course_repo

    def get_assessments(self, code_module: str, code_presentation: str, current_day: Optional[int] = None) -> List[Assessment]:
        """Học kỳ đang diễn ra: chỉ assessment đến current_day. Đã kết thúc: tất cả."""
        course = self._courses.find_course(code_module, code_presentation)
        if course is None:
            raise NotFoundError(f"Không tìm thấy course {code_module}/{code_presentation}")
        items = self._assessments.find_assessments(code_module, code_presentation)
        if course.is_ongoing(current_day):
            return [a for a in items if a.date <= current_day]
        return items

    def get_completed_assessment_count(self, student_id: int, code_module: str, code_presentation: str, cutoff: int) -> int:
        scores = self._assessments.find_student_scores(student_id, code_module, code_presentation)
        return sum(1 for s in scores if s.score is not None and s.date_submitted is not None and s.date_submitted <= cutoff)
