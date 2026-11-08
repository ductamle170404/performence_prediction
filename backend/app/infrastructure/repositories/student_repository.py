from typing import Dict, List, Optional

from app.domain.models.student import Student
from app.infrastructure.adapters.csv_data_adapter import CsvDataAdapter
from app.infrastructure.repositories.course_repository import CourseRepository


class StudentRepository:
    def __init__(self, adapter: CsvDataAdapter):
        self._adapter = adapter

    def find_students_by_class(self, code_module: str, code_presentation: str, class_id: str) -> List[Student]:
        df = self._adapter.read_csv("assignments")
        rows = df[
            (df.code_module == code_module)
            & (df.code_presentation == code_presentation)
            & (df.class_id == class_id)
        ]
        return [CourseRepository._to_student(r) for r in rows.itertuples(index=False)]

    def find_student(self, student_id: int, code_module: str, code_presentation: str) -> Optional[Student]:
        df = self._adapter.read_csv("assignments")
        rows = df[
            (df.id_student == student_id)
            & (df.code_module == code_module)
            & (df.code_presentation == code_presentation)
        ]
        return CourseRepository._to_student(next(rows.itertuples(index=False))) if not rows.empty else None

    def find_classes(self, code_module: str, code_presentation: str) -> List[str]:
        df = self._adapter.read_csv("assignments")
        rows = df[(df.code_module == code_module) & (df.code_presentation == code_presentation)]
        return sorted(rows.class_id.unique().tolist())

    def find_instructor_assignments(self, instructor_id: str) -> List[Dict[str, str]]:
        df = self._adapter.read_csv("class_instructor")
        rows = df[df.instructor_id == str(instructor_id)]
        return [
            {"code_module": r.code_module, "code_presentation": r.code_presentation, "class_id": r.class_id}
            for r in rows.itertuples(index=False)
        ]

    def list_demo_users(self, limit: int = 10) -> Dict[str, list]:
        students = self._adapter.read_csv("assignments").id_student.drop_duplicates().head(limit)
        instructors = self._adapter.read_csv("class_instructor").instructor_id.drop_duplicates().head(limit)
        return {"students": [int(s) for s in students], "instructors": [str(i) for i in instructors]}
