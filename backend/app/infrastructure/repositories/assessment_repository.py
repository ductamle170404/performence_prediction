from typing import Dict, List

from app.domain.models.assessment import Assessment, StudentScore
from app.infrastructure.adapters.csv_data_adapter import CsvDataAdapter


class AssessmentRepository:
    def __init__(self, adapter: CsvDataAdapter):
        self._adapter = adapter
        self._assessment_cache: Dict[tuple, List[Assessment]] = {}
        self._score_cache: Dict[tuple, Dict[int, List[StudentScore]]] = {}

    def find_assessments(self, code_module: str, code_presentation: str) -> List[Assessment]:
        key = (code_module, code_presentation)
        if key not in self._assessment_cache:
            self._assessment_cache[key] = self._load_assessments(*key)
        return self._assessment_cache[key]

    def find_student_scores(self, student_id: int, code_module: str, code_presentation: str) -> List[StudentScore]:
        return self._scores_by_student(code_module, code_presentation).get(student_id, [])

    # --- internals --------------------------------------------------------
    def _load_assessments(self, code_module: str, code_presentation: str) -> List[Assessment]:
        df = self._adapter.read_csv("assessments")
        rows = df[(df.code_module == code_module) & (df.code_presentation == code_presentation)]
        courses = self._adapter.read_csv("courses")
        c = courses[(courses.code_module == code_module) & (courses.code_presentation == code_presentation)]
        length = int(c.iloc[0].module_presentation_length) if not c.empty else 0
        result = []
        for r in rows.itertuples(index=False):
            # Assessment không có ngày (thường là exam cuối kỳ) -> cuối học kỳ.
            date = length if r.date != r.date else int(r.date)
            weight = 0.0 if r.weight != r.weight else float(r.weight)
            result.append(Assessment(int(r.id_assessment), code_module, code_presentation, r.assessment_type, date, weight))
        return sorted(result, key=lambda a: (a.date, a.id_assessment))

    def _scores_by_student(self, code_module: str, code_presentation: str) -> Dict[int, List[StudentScore]]:
        key = (code_module, code_presentation)
        if key in self._score_cache:
            return self._score_cache[key]
        assessments = {a.id_assessment: a for a in self.find_assessments(*key)}
        df = self._adapter.read_csv("student_assessment")
        df = df[df.id_assessment.isin(assessments.keys())]
        grouped: Dict[int, List[StudentScore]] = {}
        for r in df.itertuples(index=False):
            a = assessments[int(r.id_assessment)]
            grouped.setdefault(int(r.id_student), []).append(
                StudentScore(
                    id_assessment=a.id_assessment,
                    assessment_type=a.assessment_type,
                    assessment_date=a.date,
                    weight=a.weight,
                    date_submitted=None if r.date_submitted != r.date_submitted else int(r.date_submitted),
                    is_banked=bool(r.is_banked),
                    score=None if r.score != r.score else float(r.score),
                )
            )
        self._score_cache[key] = grouped
        return grouped
