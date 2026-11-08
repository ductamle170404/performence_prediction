from typing import Dict, List

from app.core.exceptions import ForbiddenError
from app.infrastructure.repositories.course_repository import CourseRepository
from app.infrastructure.repositories.student_repository import StudentRepository
from app.services.prediction_service import PredictionService


class InstructorService:
    def __init__(self, student_repo: StudentRepository, course_repo: CourseRepository,
                 prediction_service: PredictionService):
        self._students = student_repo
        self._courses = course_repo
        self._predictions = prediction_service

    def get_teaching_courses(self, instructor_id: str) -> List[Dict]:
        result = []
        for a in self._students.find_instructor_assignments(instructor_id):
            course = self._courses.find_course(a["code_module"], a["code_presentation"])
            students = self._students.find_students_by_class(a["code_module"], a["code_presentation"], a["class_id"])
            result.append({**a, "length": course.length if course else 0, "total_students": len(students)})
        return result

    def assert_can_view_class(self, instructor_id: str, code_module: str, code_presentation: str, class_id: str) -> None:
        allowed = any(
            a["code_module"] == code_module and a["code_presentation"] == code_presentation and a["class_id"] == class_id
            for a in self._students.find_instructor_assignments(instructor_id)
        )
        if not allowed:
            raise ForbiddenError("Giảng viên không được phân công lớp này")

    def assert_can_view_course(self, instructor_id: str, code_module: str, code_presentation: str) -> None:
        allowed = any(
            a["code_module"] == code_module and a["code_presentation"] == code_presentation
            for a in self._students.find_instructor_assignments(instructor_id)
        )
        if not allowed:
            raise ForbiddenError("Giảng viên không được phân công môn/học kỳ này")

    def get_class_students(self, instructor_id: str, code_module: str, code_presentation: str,
                           class_id: str, cutoff: int) -> List[Dict]:
        self.assert_can_view_class(instructor_id, code_module, code_presentation, class_id)
        predictions = {p.student_id: p for p in
                       self._predictions.predict_class(code_module, code_presentation, class_id, cutoff)}
        rows = []
        for s in self._students.find_students_by_class(code_module, code_presentation, class_id):
            p = predictions.get(s.id_student)
            rows.append({
                "student_id": s.id_student,
                "class_id": s.class_id,
                "n_assessments": p.n_assessments if p else 0,
                "fail_probability": p.fail_probability if p else None,
                "risk_level": p.risk_level if p else None,
                "cutoff_date": cutoff,
            })
        return sorted(rows, key=lambda r: -(r["fail_probability"] or 0))
