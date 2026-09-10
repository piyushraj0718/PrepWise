import json
import math
import socket
import time
from dataclasses import dataclass
from errno import ECONNABORTED, ECONNREFUSED, ECONNRESET, EHOSTUNREACH, ENETUNREACH, ETIMEDOUT
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

    def generate_structured(
        self, prompt: str, response_schema: dict[str, object]
    ) -> object:
        ...


class GeminiLLMProvider:
    """Gemini REST provider; credentials are supplied at call time from settings."""

    _MAX_ALLOWED_RETRIES = 5
    _MAX_ALLOWED_RETRY_DELAY_SECONDS = 30.0
    _TRANSIENT_HTTP_STATUSES = {429, 500, 502, 503, 504}
    _TRANSIENT_ERRNOS = {
        ECONNABORTED,
        ECONNREFUSED,
        ECONNRESET,
        EHOSTUNREACH,
        ENETUNREACH,
        ETIMEDOUT,
    }

    def __init__(
        self,
        api_key: str | None,
        model_name: str,
        timeout_seconds: float,
        max_retries: int = 2,
        retry_base_delay_seconds: float = 0.5,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_base_delay_seconds = retry_base_delay_seconds

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
        payload = self._request_json(request_body)
        return self._parse_response(payload)

    def generate_structured(
        self, prompt: str, response_schema: dict[str, object]
    ) -> object:
        if not self.api_key or not self.api_key.strip():
            raise LLMConfigurationError("The LLM API key is not configured")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise LLMConfigurationError("The LLM timeout is invalid")

        request_body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": response_schema,
            },
        }
        payload = self._request_json(request_body)
        try:
            text = payload["candidates"][0]["content"]["parts"][0]["text"]
            structured = json.loads(text)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise LLMResponseError(
                "The LLM returned an invalid structured response") from error
        if not isinstance(structured, (dict, list)):
            raise LLMResponseError(
                "The LLM returned an invalid structured response")
        return structured

    def _request_json(self, request_body: dict[str, object]) -> object:
        self._validate_retry_configuration()
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
        for attempt in range(self.max_retries + 1):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as error:
                if not self._is_transient_http_error(error) or attempt == self.max_retries:
                    raise LLMProviderError(
                        f"The LLM API request failed with HTTP status {error.code}"
                    ) from error
                self._backoff(attempt)
            except (URLError, TimeoutError, OSError) as error:
                if not self._is_transient_network_error(error) or attempt == self.max_retries:
                    raise LLMProviderError(
                        "The LLM API request failed due to a network error"
                    ) from error
                self._backoff(attempt)
            except json.JSONDecodeError as error:
                raise LLMProviderError(
                    "The LLM API returned invalid JSON") from error
        raise LLMProviderError("The LLM API request failed")

    def _backoff(self, attempt: int) -> None:
        time.sleep(self.retry_base_delay_seconds * (2 ** attempt))

    def _validate_retry_configuration(self) -> None:
        if (
            isinstance(self.max_retries, bool)
            or not isinstance(self.max_retries, int)
            or not 0 <= self.max_retries <= self._MAX_ALLOWED_RETRIES
        ):
            raise LLMConfigurationError("The LLM retry count is invalid")
        if (
            not math.isfinite(self.retry_base_delay_seconds)
            or not 0 <= self.retry_base_delay_seconds <= self._MAX_ALLOWED_RETRY_DELAY_SECONDS
        ):
            raise LLMConfigurationError("The LLM retry delay is invalid")

    @classmethod
    def _is_transient_http_error(cls, error: HTTPError) -> bool:
        return error.code in cls._TRANSIENT_HTTP_STATUSES

    @classmethod
    def _is_transient_network_error(cls, error: BaseException) -> bool:
        reason = error.reason if isinstance(error, URLError) else error
        if isinstance(reason, (ConnectionResetError, ConnectionAbortedError,
                               ConnectionRefusedError, TimeoutError, socket.timeout)):
            return True
        return isinstance(reason, OSError) and reason.errno in cls._TRANSIENT_ERRNOS

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
