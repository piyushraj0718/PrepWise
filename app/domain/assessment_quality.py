from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
import re
from typing import Any, Sequence

from app.domain.assessment import BloomLevel, Difficulty


class QualityDecision(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    REGENERATE = "regenerate"


@dataclass(frozen=True)
class QualityFinding:
    code: str
    message: str
    question_index: int | None = None


@dataclass(frozen=True)
class QuestionQualityResult:
    accepted: bool
    hard_failures: list[QualityFinding] = field(default_factory=list)
    warnings: list[QualityFinding] = field(default_factory=list)
    deterministic_score: float = 0.0
    normalized_stem: str = ""
    duplicate_of_question_index: int | None = None
    near_duplicate_of_question_index: int | None = None
    decision: QualityDecision = QualityDecision.REJECT


@dataclass(frozen=True)
class AssessmentQualityResult:
    accepted: bool
    question_results: list[QuestionQualityResult] = field(default_factory=list)
    hard_failures: list[QualityFinding] = field(default_factory=list)
    warnings: list[QualityFinding] = field(default_factory=list)
    deterministic_score: float = 0.0
    decision: QualityDecision = QualityDecision.REJECT


@dataclass(frozen=True)
class AssessmentQualityConfig:
    near_duplicate_threshold: float = 0.92
    minimum_accepted_score: float = 0.8
    warning_stem_length: int = 20
    warning_explanation_length: int = 20


class RegenerationDecisionPolicy:
    def __init__(self, max_attempts: int = 3) -> None:
        if isinstance(max_attempts, bool) or max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.max_attempts = max_attempts

    def decide(
        self, result: QuestionQualityResult | AssessmentQualityResult, attempt_number: int = 1
    ) -> QualityDecision:
        if result.accepted:
            return QualityDecision.ACCEPT
        if attempt_number < self.max_attempts:
            return QualityDecision.REGENERATE
        return QualityDecision.REJECT


class AssessmentQualityValidator:
    """Authoritative deterministic validation for generated MCQ content.

    The score measures structural and provenance completeness only. It does not
    measure factual correctness, distractor quality, or semantic difficulty.
    """

    _CITATION_MARKER_PATTERN = re.compile(
        r"(?:\[\s*\d+\s*\]|\[\s*(?:chunk[_ -]?id|citation|source)\s*[:=#]|"
        r"\b(?:chunk[_ -]?id|citation|source)\s*[:=])",
        re.IGNORECASE,
    )
    _FORBIDDEN_OPTION_PATTERN = re.compile(
        r"^(?:all|none)\s+of\s+the\s+above$", re.IGNORECASE)
    _SCORE_WEIGHTS = {
        "structure": 0.25,
        "citations": 0.20,
        "options": 0.20,
        "explanation": 0.15,
        "uniqueness": 0.20,
    }

    def __init__(self, config: AssessmentQualityConfig | None = None) -> None:
        self.config = config or AssessmentQualityConfig()
        if not 0 < self.config.near_duplicate_threshold <= 1:
            raise ValueError(
                "near_duplicate_threshold must be between 0 and 1")
        if not 0 <= self.config.minimum_accepted_score <= 1:
            raise ValueError("minimum_accepted_score must be between 0 and 1")

    def validate(
        self,
        questions: Sequence[Any],
        retrieved_chunk_ids: set[str] | frozenset[str],
    ) -> AssessmentQualityResult:
        results = [
            self._validate_question(question, index, retrieved_chunk_ids)
            for index, question in enumerate(questions)
        ]
        results = self._apply_batch_uniqueness(results)
        hard_failures = [
            finding for result in results for finding in result.hard_failures]
        warnings = [
            finding for result in results for finding in result.warnings]
        score = self._average_score(results)
        accepted = not hard_failures and score >= self.config.minimum_accepted_score
        decision = QualityDecision.ACCEPT if accepted else QualityDecision.REGENERATE
        return AssessmentQualityResult(
            accepted=accepted,
            question_results=results,
            hard_failures=hard_failures,
            warnings=warnings,
            deterministic_score=score,
            decision=decision,
        )

    def _validate_question(
        self,
        question: Any,
        question_index: int,
        retrieved_chunk_ids: set[str] | frozenset[str],
    ) -> QuestionQualityResult:
        hard_failures: list[QualityFinding] = []
        warnings: list[QualityFinding] = []
        stem = self._text(question, "stem")
        explanation = self._text(question, "explanation")
        options = self._sequence(question, "options")
        cited_chunk_ids = self._sequence(question, "cited_chunk_ids")
        correct_option_key = self._value(question, "correct_option_key")
        difficulty = self._value(question, "difficulty")
        bloom_level = self._value(question, "bloom_level")
        normalized_stem = normalize_quality_text(stem)

        if not stem:
            self._fail(hard_failures, "empty_stem",
                       "The question stem is empty", question_index)
        if len(options) != 4:
            self._fail(hard_failures, "invalid_option_count",
                       "The question must have exactly four options", question_index)
        option_keys = [self._text(option, "option_key") for option in options]
        option_texts = [self._text(option, "option_text")
                        for option in options]
        normalized_option_keys = [
            normalize_quality_text(key) for key in option_keys]
        normalized_option_texts = [
            normalize_quality_text(text) for text in option_texts]
        if any(not text for text in option_texts):
            self._fail(hard_failures, "empty_option_text",
                       "Option text must not be empty", question_index)
        if len(set(normalized_option_keys)) != len(normalized_option_keys):
            self._fail(hard_failures, "duplicate_option_keys",
                       "Option keys must be unique", question_index)
        if len(set(normalized_option_texts)) != len(normalized_option_texts):
            self._fail(hard_failures, "duplicate_option_text",
                       "Option text must be unique", question_index)
        if any(self._FORBIDDEN_OPTION_PATTERN.fullmatch(text.strip()) for text in option_texts):
            self._fail(hard_failures, "forbidden_option_pattern",
                       "All/none of the above options are not allowed", question_index)
        if isinstance(correct_option_key, str):
            normalized_correct_key = normalize_quality_text(correct_option_key)
            if normalized_correct_key not in normalized_option_keys:
                self._fail(hard_failures, "invalid_correct_option",
                           "The correct option key must exist", question_index)
        else:
            self._fail(hard_failures, "missing_correct_option",
                       "Exactly one correct option key is required", question_index)
        if not explanation:
            self._fail(hard_failures, "empty_explanation",
                       "The explanation is empty", question_index)
        if not self._valid_enum(difficulty, Difficulty):
            self._fail(hard_failures, "invalid_difficulty",
                       "Difficulty is invalid", question_index)
        if not self._valid_enum(bloom_level, BloomLevel):
            self._fail(hard_failures, "invalid_bloom_level",
                       "Bloom level is invalid", question_index)
        if not cited_chunk_ids:
            self._fail(hard_failures, "empty_citations",
                       "At least one citation is required", question_index)
        normalized_citations = [self._text_value(
            value) for value in cited_chunk_ids]
        if len(set(normalized_citations)) != len(normalized_citations):
            self._fail(hard_failures, "duplicate_citations",
                       "Citation IDs must be unique", question_index)
        if any(citation_id not in retrieved_chunk_ids for citation_id in normalized_citations):
            self._fail(hard_failures, "invalid_citation",
                       "Every citation must be a retrieved chunk ID", question_index)
        generated_text = [stem, explanation, *option_texts]
        if any(self._CITATION_MARKER_PATTERN.search(text) for text in generated_text):
            self._fail(hard_failures, "citation_markup",
                       "Generated text must not contain citation markup", question_index)
        if stem and len(stem) < self.config.warning_stem_length:
            self._warn(warnings, "short_stem",
                       "The stem is unusually short", question_index)
        if explanation and len(explanation) < self.config.warning_explanation_length:
            self._warn(warnings, "short_explanation",
                       "The explanation is unusually short", question_index)

        components = {
            "structure": not any(finding.code in {"empty_stem", "invalid_option_count", "empty_option_text", "missing_correct_option", "invalid_correct_option", "invalid_difficulty", "invalid_bloom_level"} for finding in hard_failures),
            "citations": not any(finding.code in {"empty_citations", "duplicate_citations", "invalid_citation", "citation_markup"} for finding in hard_failures),
            "options": not any(finding.code in {"duplicate_option_keys", "duplicate_option_text", "forbidden_option_pattern"} for finding in hard_failures),
            "explanation": bool(explanation),
            "uniqueness": True,
        }
        score = self._score(components)
        accepted = not hard_failures and score >= self.config.minimum_accepted_score
        return QuestionQualityResult(
            accepted=accepted,
            hard_failures=hard_failures,
            warnings=warnings,
            deterministic_score=score,
            normalized_stem=normalized_stem,
            decision=QualityDecision.ACCEPT if accepted else QualityDecision.REGENERATE,
        )

    def _apply_batch_uniqueness(
        self, results: list[QuestionQualityResult]
    ) -> list[QuestionQualityResult]:
        updated = list(results)
        for index, result in enumerate(results):
            duplicate_index: int | None = None
            near_duplicate_index: int | None = None
            for previous_index, previous in enumerate(results[:index]):
                if result.normalized_stem == previous.normalized_stem:
                    duplicate_index = previous_index
                    break
                if self._similarity(result.normalized_stem, previous.normalized_stem) >= self.config.near_duplicate_threshold:
                    near_duplicate_index = previous_index
                    break
            if duplicate_index is None and near_duplicate_index is None:
                continue
            failures = list(result.hard_failures)
            if duplicate_index is not None:
                failures.append(QualityFinding(
                    "duplicate_stem", "The question stem duplicates an earlier question", index))
            else:
                failures.append(QualityFinding(
                    "near_duplicate_stem", "The question stem is too similar to an earlier question", index))
            score = result.deterministic_score - \
                self._SCORE_WEIGHTS["uniqueness"]
            updated[index] = QuestionQualityResult(
                accepted=False,
                hard_failures=failures,
                warnings=result.warnings,
                deterministic_score=max(0.0, score),
                normalized_stem=result.normalized_stem,
                duplicate_of_question_index=duplicate_index,
                near_duplicate_of_question_index=near_duplicate_index,
                decision=QualityDecision.REGENERATE,
            )
        return updated

    @staticmethod
    def _similarity(left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        return SequenceMatcher(None, left, right).ratio()

    def stems_conflict(self, left: str, right: str) -> bool:
        normalized_left = normalize_quality_text(left)
        normalized_right = normalize_quality_text(right)
        return normalized_left == normalized_right or (
            self._similarity(normalized_left, normalized_right)
            >= self.config.near_duplicate_threshold
        )

    def _average_score(self, results: Sequence[QuestionQualityResult]) -> float:
        if not results:
            return 0.0
        return sum(result.deterministic_score for result in results) / len(results)

    @classmethod
    def _score(cls, components: dict[str, bool]) -> float:
        return sum(
            cls._SCORE_WEIGHTS[name] for name, passed in components.items() if passed
        )

    @staticmethod
    def _value(item: Any, name: str) -> Any:
        if isinstance(item, dict):
            return item.get(name)
        return getattr(item, name, None)

    @classmethod
    def _text(cls, item: Any, name: str) -> str:
        return cls._text_value(cls._value(item, name))

    @staticmethod
    def _text_value(value: Any) -> str:
        return value.strip() if isinstance(value, str) else ""

    @staticmethod
    def _sequence(item: Any, name: str) -> list[Any]:
        value = AssessmentQualityValidator._value(item, name)
        return list(value) if isinstance(value, (list, tuple)) else []

    @staticmethod
    def _valid_enum(value: Any, enum_type: type[Enum]) -> bool:
        if isinstance(value, enum_type):
            return True
        return isinstance(value, str) and value in {member.value for member in enum_type}

    @staticmethod
    def _fail(
        findings: list[QualityFinding], code: str, message: str, question_index: int
    ) -> None:
        findings.append(QualityFinding(code, message, question_index))

    @staticmethod
    def _warn(
        findings: list[QualityFinding], code: str, message: str, question_index: int
    ) -> None:
        findings.append(QualityFinding(code, message, question_index))


def normalize_quality_text(value: str) -> str:
    """Normalize comparison text without changing the stored source text."""
    return " ".join(re.sub(r"[^\w\s]", " ", value.casefold()).split())
