class RiskPolicy:
    """Quy tắc nghiệp vụ chuyển xác suất thành mức nguy cơ."""

    LOW, MEDIUM, HIGH = "LOW", "MEDIUM", "HIGH"

    def __init__(self, medium_threshold: float, high_threshold: float, at_risk_threshold: float):
        if not 0 <= medium_threshold <= high_threshold <= 1:
            raise ValueError("Cần 0 <= medium <= high <= 1")
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold
        self.at_risk_threshold = at_risk_threshold

    def classify(self, fail_probability: float) -> str:
        if fail_probability >= self.high_threshold:
            return self.HIGH
        if fail_probability >= self.medium_threshold:
            return self.MEDIUM
        return self.LOW

    def is_at_risk(self, fail_probability: float) -> bool:
        return fail_probability >= self.at_risk_threshold
