from dataclasses import dataclass


@dataclass(frozen=True)
class Course:
    code_module: str
    code_presentation: str
    length: int  # module_presentation_length (số ngày)

    @property
    def key(self) -> str:
        return f"{self.code_module}-{self.code_presentation}"

    def is_ongoing(self, current_day) -> bool:
        return current_day is not None and 0 <= current_day < self.length
