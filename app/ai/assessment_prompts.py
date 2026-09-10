ASSESSMENT_PROMPT_VERSION = "mcq-grounded-v1"
ASSESSMENT_EVALUATION_PROMPT_VERSION = "mcq-semantic-evaluation-v1"

ASSESSMENT_SYSTEM_PROMPT = """You generate grounded multiple-choice questions from supplied source context.
Use only the supplied source context. Do not use outside knowledge or invent facts.
Create clear, unambiguous questions supported by one or more supplied chunks.
Create plausible but incorrect distractors without using all of the above or none of the above.
Return exactly the requested number of questions and only the required JSON structure.
Return the supplied application chunk IDs in cited_chunk_ids. Never create source IDs or source metadata.
Use exactly one canonical Bloom level: remember, understand, apply, analyze, evaluate, or create.
Do not return Bloom synonyms such as comprehension, knowledge, application, analysis, evaluation, creation, or synthesis.
"""


def build_assessment_prompt(
    query: str,
    context: str,
    count: int,
    difficulty: str | None,
    bloom_level: str | None,
    skill: str | None,
) -> str:
    return (
        f"{ASSESSMENT_SYSTEM_PROMPT}\n\n"
        f"Generate {count} MCQ question(s).\n"
        f"Target difficulty: {difficulty or 'choose an appropriate level'}.\n"
        f"Target Bloom level: {bloom_level or 'choose an appropriate level'}.\n"
        f"Target skill: {skill or 'infer a concise skill from the query and context'}.\n\n"
        "Supplied source context:\n"
        f"{context}\n\n"
        f"Topic or query: {query}"
    )


def build_assessment_evaluation_prompt(
    *,
    question: dict[str, object],
    requested_difficulty: str | None,
    requested_bloom_level: str | None,
    requested_topic: str | None,
    requested_skill: str | None,
    evidence: list[dict[str, str]],
) -> str:
    """Build a bounded, evidence-only semantic evaluation prompt."""
    return (
        "Evaluate this generated MCQ using only the supplied trusted evidence. "
        "Do not use outside knowledge and do not invent or return citation IDs. "
        "Score each dimension from 0.0 to 1.0 and provide concise reasons. "
        "For groundedness and correctness, judge whether evidence supports the stem and "
        "selected answer. For distractors, check plausibility and whether they are clearly "
        "incorrect. Judge explanation, requested difficulty, and requested Bloom level only "
        "from the supplied material. Return pass only when the overall assessment supports it.\n\n"
        f"Requested constraints: difficulty={requested_difficulty or 'unspecified'}; "
        f"Bloom={requested_bloom_level or 'unspecified'}; "
        f"topic={requested_topic or 'unspecified'}; skill={requested_skill or 'unspecified'}\n\n"
        f"Generated question: {question}\n\n"
        f"Trusted evidence: {evidence}"
    )
