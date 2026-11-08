from typing import List, Optional

from pydantic import BaseModel, Field


class PredictionOut(BaseModel):
    student_id: int
    code_module: str
    code_presentation: str
    class_id: str
    cutoff_date: int
    n_assessments: int
    fail_probability: float = Field(ge=0, le=1)
    risk_level: str
    model_version: str
    generated_at: str


class PredictionHistoryOut(BaseModel):
    student_id: int
    code_module: str
    code_presentation: str
    history: List[PredictionOut]


class ClassRiskOut(BaseModel):
    code_module: str
    code_presentation: str
    class_id: str
    total_students: int
    at_risk_students: int
    at_risk_rate: float
    avg_fail_probability: float
    cutoff_date: int


class ClassSummaryOut(BaseModel):
    class_id: str
    total_students: int
    at_risk_students: int
    at_risk_rate: float
    avg_fail_probability: float
    cutoff_date: int


class CourseRiskOut(BaseModel):
    code_module: str
    code_presentation: str
    total_students: int
    at_risk_students: int
    at_risk_rate: float
    avg_fail_probability: float
    cutoff_date: int
    classes: List[ClassSummaryOut]


class RiskTrendPointOut(BaseModel):
    cutoff_date: int
    total_students: int
    at_risk_students: int
    at_risk_rate: float
    avg_fail_probability: float


class ClassStudentOut(BaseModel):
    student_id: int
    class_id: str
    n_assessments: int
    fail_probability: Optional[float] = None
    risk_level: Optional[str] = None
    cutoff_date: int
