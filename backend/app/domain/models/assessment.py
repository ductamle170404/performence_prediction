from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Assessment:
    id_assessment: int
    code_module: str
    code_presentation: str
    assessment_type: str
    date: int  # ngày tương đối so với đầu kỳ
    weight: float


@dataclass(frozen=True)
class StudentScore:
    """Điểm đã ghi nhận của sinh viên cho một assessment."""

    id_assessment: int
    assessment_type: str
    assessment_date: int
    weight: float
    date_submitted: Optional[int]
    is_banked: bool
    score: Optional[float]
