"""Optional manual smoke test for Gemini-backed semantic MCQ evaluation.

Set GEMINI_API_KEY and ASSESSMENT_SEMANTIC_EVALUATOR_ENABLED=true in .env.
This script never prints credentials and is not collected by pytest.
"""

import argparse

from app.api.dependencies import get_assessment_generation_service
from app.core.config import get_settings
from app.db.session import SessionLocal
from app.schemas.assessments import QuestionGenerationRequest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--document-id", required=True)
    args = parser.parse_args()

    if not get_settings().assessment_semantic_evaluator_enabled:
        raise SystemExit(
            "Set ASSESSMENT_SEMANTIC_EVALUATOR_ENABLED=true before this smoke test"
        )

    request = QuestionGenerationRequest(
        query=args.query, document_id=args.document_id, count=1
    )
    with SessionLocal() as session:
        run, questions = get_assessment_generation_service(session).generate(request)
    print("Generation run:", run.id)
    print("Semantic evaluation completed before accepting:", questions[0].id)


if __name__ == "__main__":
    main()
