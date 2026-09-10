"""Optional manual smoke test for configured grounded Gemini question generation."""

import argparse

from app.api.dependencies import get_assessment_generation_service
from app.db.session import SessionLocal
from app.schemas.assessments import QuestionGenerationRequest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--document-id", default=None)
    parser.add_argument("--skill", default=None)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=None)
    parser.add_argument("--difficulty", default=None)
    parser.add_argument("--bloom-level", default=None)
    args = parser.parse_args()

    request = QuestionGenerationRequest(
        query=args.query,
        document_id=args.document_id,
        skill=args.skill,
        count=args.count,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        difficulty=args.difficulty,
        bloom_level=args.bloom_level,
    )
    with SessionLocal() as session:
        run, questions = get_assessment_generation_service(
            session).generate(request)
    print("Generation run:", run.id)
    for question in questions:
        print(question.stem)
        for option in question.options:
            print(f"  {option.option_key}. {option.option_text}")
        print("Answer:", question.correct_option_key)
        print("Citations:", ", ".join(
            citation.chunk_id for citation in question.citations))


if __name__ == "__main__":
    main()
