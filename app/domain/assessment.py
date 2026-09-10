from dataclasses import dataclass
from enum import Enum


class QuestionType(str, Enum):
    MCQ = "mcq"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class BloomLevel(str, Enum):
    REMEMBER = "remember"
    UNDERSTAND = "understand"
    APPLY = "apply"
    ANALYZE = "analyze"
    EVALUATE = "evaluate"
    CREATE = "create"


def score_mcq_answer(submitted_key: str, correct_key: str) -> bool:
    """Return True if the submitted option key matches the correct option key.

    Comparison is case-sensitive and exact, matching how option keys are stored
    (always uppercase single letters: A, B, C, D).
    """
    return submitted_key == correct_key


# ---------------------------------------------------------------------------
# M4C: Weak-topic detection — deterministic, no I/O
# ---------------------------------------------------------------------------

# Configurable thresholds.  These are the authoritative constants for the
# weak-topic detection policy; they may be referenced by services and tests.
WEAK_TOPIC_MIN_ATTEMPTS: int = 3
WEAK_TOPIC_ACCURACY_THRESHOLD: float = 0.60


@dataclass(frozen=True)
class AreaPerformance:
    """Aggregated performance for a single topic or skill label."""

    label: str
    attempts: int
    correct: int
    accuracy: float  # correct / attempts, rounded to 4 dp; 0.0 when attempts == 0


@dataclass(frozen=True)
class WeakArea:
    """A topic or skill identified as weak by the detection policy."""

    label: str
    attempts: int
    correct: int
    accuracy: float


def compute_accuracy(attempts: int, correct: int) -> float:
    """Return accuracy as a float in [0.0, 1.0], rounded to 4 decimal places."""
    if attempts == 0:
        return 0.0
    return round(correct / attempts, 4)


def detect_weak_areas(
    performances: list[AreaPerformance],
    min_attempts: int = WEAK_TOPIC_MIN_ATTEMPTS,
    accuracy_threshold: float = WEAK_TOPIC_ACCURACY_THRESHOLD,
) -> list[WeakArea]:
    """Return the subset of areas that are considered weak.

    An area is weak when:
      - attempts >= min_attempts  (sufficient evidence)
      - accuracy <= accuracy_threshold  (poor performance)

    The result is sorted by accuracy ascending (weakest first), then label
    ascending for determinism.
    """
    weak = [
        WeakArea(
            label=p.label,
            attempts=p.attempts,
            correct=p.correct,
            accuracy=p.accuracy,
        )
        for p in performances
        if p.attempts >= min_attempts and p.accuracy <= accuracy_threshold
    ]
    return sorted(weak, key=lambda w: (w.accuracy, w.label))
