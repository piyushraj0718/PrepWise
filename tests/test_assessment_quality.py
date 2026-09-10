import pytest

from app.domain.assessment_quality import (
    AssessmentQualityConfig,
    AssessmentQualityValidator,
    QualityDecision,
    RegenerationDecisionPolicy,
    normalize_quality_text,
)


def valid_question(stem: str = "What improves lookup performance?") -> dict:
    return {
        "stem": stem,
        "options": [
            {"option_key": "A", "option_text": "An index"},
            {"option_key": "B", "option_text": "A slower disk"},
            {"option_key": "C", "option_text": "A blank table"},
            {"option_key": "D", "option_text": "A deleted query"},
        ],
        "correct_option_key": "A",
        "explanation": "An index improves lookup performance.",
        "difficulty": "easy",
        "bloom_level": "understand",
        "cited_chunk_ids": ["chunk-1"],
    }


def validate(question: dict | list[dict], questions: list[dict] | None = None):
    validator = AssessmentQualityValidator()
    batch = questions if questions is not None else (
        question if isinstance(question, list) else [question]
    )
    return validator.validate(batch, {"chunk-1"})


def failure_codes(result) -> set[str]:
    return {finding.code for finding in result.hard_failures}


def test_valid_question_is_accepted_with_full_deterministic_score() -> None:
    result = validate(valid_question())

    assert result.accepted
    assert result.decision is QualityDecision.ACCEPT
    assert result.deterministic_score == pytest.approx(1.0)
    assert result.hard_failures == []


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("stem", "   ", "empty_stem"),
        ("explanation", "", "empty_explanation"),
        ("difficulty", "expert", "invalid_difficulty"),
        ("bloom_level", "recall", "invalid_bloom_level"),
        ("cited_chunk_ids", [], "empty_citations"),
        ("cited_chunk_ids", ["unknown"], "invalid_citation"),
    ],
)
def test_empty_and_invalid_scalar_fields_are_hard_failures(field, value, code) -> None:
    question = valid_question()
    question[field] = value

    result = validate(question)

    assert code in failure_codes(result)
    assert not result.accepted


def test_invalid_option_count_is_rejected() -> None:
    question = valid_question()
    question["options"].pop()

    assert "invalid_option_count" in failure_codes(validate(question))


def test_duplicate_option_keys_and_text_are_rejected() -> None:
    question = valid_question()
    question["options"][1]["option_key"] = "A"
    question["options"][2]["option_text"] = "  AN INDEX! "

    codes = failure_codes(validate(question))

    assert {"duplicate_option_keys", "duplicate_option_text"} <= codes


def test_invalid_correct_option_is_rejected() -> None:
    question = valid_question()
    question["correct_option_key"] = "Z"

    assert "invalid_correct_option" in failure_codes(validate(question))


@pytest.mark.parametrize("text", ["all of the above", "none of the above", "NONE OF THE ABOVE"])
def test_forbidden_option_patterns_are_rejected(text: str) -> None:
    question = valid_question()
    question["options"][3]["option_text"] = text

    assert "forbidden_option_pattern" in failure_codes(validate(question))


@pytest.mark.parametrize("text", [
    "Supported by [1]",
    "Supported by [chunk_id=chunk-1]",
    "Supported by citation: chunk-1",
])
def test_citation_markers_in_generated_text_are_rejected(text: str) -> None:
    question = valid_question()
    question["explanation"] = text

    assert "citation_markup" in failure_codes(validate(question))


def test_duplicate_citations_are_rejected() -> None:
    question = valid_question()
    question["cited_chunk_ids"] = ["chunk-1", "chunk-1"]

    assert "duplicate_citations" in failure_codes(validate(question))


def test_normalized_exact_duplicate_stems_are_rejected() -> None:
    first = valid_question("What improves lookup performance?")
    second = valid_question("  what improves lookup performance!!! ")

    result = validate(first, [first, second])

    assert "duplicate_stem" in failure_codes(result.question_results[1])
    assert result.question_results[1].duplicate_of_question_index == 0


def test_near_duplicate_stems_are_rejected_with_explicit_threshold() -> None:
    validator = AssessmentQualityValidator(
        AssessmentQualityConfig(near_duplicate_threshold=0.8)
    )
    questions = [
        valid_question("What improves database lookup performance?"),
        valid_question("What improves database lookup speed?"),
    ]

    result = validator.validate(questions, {"chunk-1"})

    assert "near_duplicate_stem" in failure_codes(result.question_results[1])
    assert result.question_results[1].near_duplicate_of_question_index == 0


def test_legitimate_similar_but_distinct_stems_are_not_flagged() -> None:
    questions = [
        valid_question("What improves database lookup performance?"),
        valid_question("Which storage method reduces duplicate records?"),
    ]

    result = validate(questions)

    assert result.accepted
    assert result.question_results[1].hard_failures == []


def test_normalization_preserves_only_comparison_representation() -> None:
    original = "  Which INDEX improves lookup-performance?  "

    assert normalize_quality_text(
        original) == "which index improves lookup performance"
    assert original != normalize_quality_text(original)


def test_short_content_is_a_warning_not_a_semantic_failure() -> None:
    question = valid_question("Why?")
    question["explanation"] = "Because."

    result = validate(question)

    assert result.accepted
    assert {warning.code for warning in result.warnings} == {
        "short_stem", "short_explanation"
    }


def test_batch_score_is_average_of_question_scores() -> None:
    questions = [valid_question(), valid_question(
        "Which method reduces duplicate records?")]

    result = validate(questions)

    expected = sum(
        question.deterministic_score for question in result.question_results) / 2
    assert result.deterministic_score == pytest.approx(expected)


def test_regeneration_policy_accepts_valid_result() -> None:
    result = validate(valid_question())
    policy = RegenerationDecisionPolicy(max_attempts=3)

    assert policy.decide(result, attempt_number=1) is QualityDecision.ACCEPT


def test_regeneration_policy_regenerates_failed_result_before_limit() -> None:
    result = validate({**valid_question(), "stem": ""})
    policy = RegenerationDecisionPolicy(max_attempts=3)

    assert policy.decide(
        result, attempt_number=1) is QualityDecision.REGENERATE


def test_regeneration_policy_rejects_failed_result_at_limit() -> None:
    result = validate({**valid_question(), "stem": ""})
    policy = RegenerationDecisionPolicy(max_attempts=3)

    assert policy.decide(result, attempt_number=3) is QualityDecision.REJECT
