import json
import math
from dataclasses import dataclass
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LLMProviderError(RuntimeError):
    pass


class LLMConfigurationError(LLMProviderError):
    pass


class LLMResponseError(LLMProviderError):
    pass


@dataclass(frozen=True)
class LLMAnswer:
    answer: str
    cited_chunk_ids: list[str]


class LLMProvider(Protocol):
    model_name: str

    def generate(self, prompt: str) -> LLMAnswer:
        ...


class GeminiLLMProvider:
    """Gemini REST provider; credentials are supplied at call time from settings."""

    def __init__(self, api_key: str | None, model_name: str, timeout_seconds: float) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str) -> LLMAnswer:
        if not self.api_key or not self.api_key.strip():
            raise LLMConfigurationError("The LLM API key is not configured")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise LLMConfigurationError("The LLM timeout is invalid")

        request_body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "answer": {"type": "STRING"},
                        "cited_chunk_ids": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"},
                        },
                    },
                    "required": ["answer", "cited_chunk_ids"],
                },
            },
        }
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:generateContent?key={self.api_key}"
        )
        request = Request(
            url,
            data=json.dumps(request_body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raise LLMProviderError("The LLM API request failed") from error
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
            raise LLMProviderError("The LLM API request failed") from error
        return self._parse_response(payload)

    @staticmethod
    def _parse_response(payload: object) -> LLMAnswer:
        try:
            text = payload["candidates"][0]["content"]["parts"][0]["text"]
            structured = json.loads(text)
            answer = structured["answer"]
            cited_chunk_ids = structured["cited_chunk_ids"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise LLMResponseError(
                "The LLM returned an invalid response") from error
        if (
            not isinstance(answer, str)
            or not answer.strip()
            or not isinstance(cited_chunk_ids, list)
            or not all(isinstance(chunk_id, str) and chunk_id for chunk_id in cited_chunk_ids)
        ):
            raise LLMResponseError("The LLM returned an invalid response")
        return LLMAnswer(answer=answer.strip(), cited_chunk_ids=cited_chunk_ids)
