import math
from typing import Literal, Protocol


class SemanticEvaluationResultLike(Protocol):
    recommendation: Literal["pass", "fail"]
    overall_score: float
    groundedness: "SemanticDimensionLike"
    correctness: "SemanticDimensionLike"
    distractor_quality: "SemanticDimensionLike"
    explanation_quality: "SemanticDimensionLike"
    difficulty_alignment: "SemanticDimensionLike"
    bloom_alignment: "SemanticDimensionLike"


class SemanticDimensionLike(Protocol):
    score: float


class SemanticQualityPolicy:
    """Interprets a non-authoritative evaluator result with an explicit threshold."""

    def __init__(self, pass_threshold: float = 0.75) -> None:
        if not math.isfinite(pass_threshold) or not 0.0 <= pass_threshold <= 1.0:
            raise ValueError("semantic pass threshold must be between 0 and 1")
        self.pass_threshold = pass_threshold

    def passes(self, result: SemanticEvaluationResultLike) -> bool:
        dimensions = (
            result.groundedness,
            result.correctness,
            result.distractor_quality,
            result.explanation_quality,
            result.difficulty_alignment,
            result.bloom_alignment,
        )
        return (
            result.recommendation == "pass"
            and result.overall_score >= self.pass_threshold
            and all(item.score >= self.pass_threshold for item in dimensions)
        )
