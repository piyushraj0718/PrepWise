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
