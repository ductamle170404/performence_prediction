from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Student:
    """Bản ghi đăng ký của một sinh viên trong một lớp/học kỳ."""

    id_student: int
    code_module: str
    code_presentation: str
    class_id: str
    date_registration: Optional[float] = None
    registration_rank: Optional[float] = None
