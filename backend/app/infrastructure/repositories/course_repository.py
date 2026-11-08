from typing import List, Optional, Tuple

from app.domain.models.course import Course
from app.domain.models.student import Student
from app.infrastructure.adapters.csv_data_adapter import CsvDataAdapter


class CourseRepository:
    def __init__(self, adapter: CsvDataAdapter):
        self._adapter = adapter

    def find_course(self, code_module: str, code_presentation: str) -> Optional[Course]:
        df = self._adapter.read_csv("courses")
        row = df[(df.code_module == code_module) & (df.code_presentation == code_presentation)]
        if row.empty:
            return None
        r = row.iloc[0]
        return Course(code_module, code_presentation, int(r.module_presentation_length))

    def find_student_courses(self, student_id: int) -> List[Tuple[Student, Course]]:
        df = self._adapter.read_csv("assignments")
        rows = df[df.id_student == student_id]
        result = []
        for r in rows.itertuples(index=False):
            course = self.find_course(r.code_module, r.code_presentation)
            if course is None:
                continue
            result.append((self._to_student(r), course))
        return result

    @staticmethod
    def _to_student(r) -> Student:
        def opt(name):
            v = getattr(r, name, None)
            return None if v is None or v != v else float(v)

        return Student(
            id_student=int(r.id_student),
            code_module=r.code_module,
            code_presentation=r.code_presentation,
            class_id=r.class_id,
            date_registration=opt("date_registration"),
            registration_rank=opt("registration_rank"),
        )
