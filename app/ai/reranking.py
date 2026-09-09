from collections.abc import Sequence
from numbers import Real
from typing import Protocol
import math


class RerankerProvider(Protocol):
    model_name: str

    def score(
        self, query: str, candidates: Sequence[tuple[str, str]]
    ) -> list[float]:
        ...


class SentenceTransformerCrossEncoderReranker:
    """Local cross-encoder provider with lazy model loading."""

    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ) -> None:
        self.model_name = model_name
        self._model = None

    def score(
        self, query: str, candidates: Sequence[tuple[str, str]]
    ) -> list[float]:
        if not candidates:
            return []
        model = self._get_model()
        try:
            scores = model.predict([(query, text) for _, text in candidates])
            values = list(scores)
        except Exception as error:
            raise RuntimeError("The reranker model failed") from error
        if len(values) != len(candidates) or any(
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in values
        ):
            raise RuntimeError("The reranker returned invalid scores")
        return [float(value) for value in values]

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as error:
                raise RuntimeError(
                    "The local reranker requires sentence-transformers"
                ) from error
            self._model = CrossEncoder(self.model_name)
        return self._model
