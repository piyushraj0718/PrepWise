from collections.abc import Sequence
from typing import Protocol


class EmbeddingProvider(Protocol):
    model_name: str

    def embed_text(self, text: str) -> list[float]:
        ...

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class SentenceTransformerEmbeddingProvider:
    """Local provider that loads its model only when the application uses it."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "The local embedding provider requires sentence-transformers"
            ) from error

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(list(texts), convert_to_numpy=True)
        return [vector.tolist() for vector in vectors]
