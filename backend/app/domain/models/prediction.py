from dataclasses import asdict, dataclass


@dataclass
class Prediction:
    student_id: int
    code_module: str
    code_presentation: str
    class_id: str
    cutoff_date: int  # ngày tương đối so với đầu kỳ
    n_assessments: int
    fail_probability: float
    risk_level: str
    model_version: str
    generated_at: str

    def to_dict(self) -> dict:
        return asdict(self)
