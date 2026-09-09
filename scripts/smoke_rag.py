"""Optional manual smoke test for the configured local retrieval and Gemini provider."""

import argparse

from app.api.dependencies import get_rag_service
from app.db.session import SessionLocal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, default=None)
    parser.add_argument("--document-id", default=None)
    args = parser.parse_args()

    with SessionLocal() as session:
        result = get_rag_service(session).ask(
            args.query,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            document_id=args.document_id,
        )
    print(result.answer)
    print("Citations:", ", ".join(
        citation.chunk_id for citation in result.citations))


if __name__ == "__main__":
    main()
