from typing import List, Optional

from pydantic import BaseModel


class StudentCourseOut(BaseModel):
    code_module: str
    code_presentation: str
    class_id: str
    length: int
    is_ongoing: bool


class AssessmentOut(BaseModel):
    id_assessment: int
    assessment_type: str
    date: int
    weight: float


class TimelineOut(BaseModel):
    code_module: str
    code_presentation: str
    length: int
    is_ongoing: bool
    current_day: Optional[int] = None
    assessments: List[AssessmentOut]


class TeachingClassOut(BaseModel):
    code_module: str
    code_presentation: str
    class_id: str
    length: int
    total_students: int
