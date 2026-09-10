from urllib.error import HTTPError, URLError

import pytest

import app.ai.llm as llm_module
from app.ai.llm import GeminiLLMProvider, LLMProviderError


def response_payload() -> dict[str, object]:
    return {
        "candidates": [{
            "content": {
                "parts": [{
                    "text": '{"answer":"ok","cited_chunk_ids":[]}',
                }],
            },
        }],
    }


class FakeHTTPResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        import json

        return json.dumps(self.payload).encode("utf-8")


def http_error(status: int) -> HTTPError:
    return HTTPError("https://example.test/generate", status, "failure", {}, None)


def provider(max_retries: int = 2) -> GeminiLLMProvider:
    return GeminiLLMProvider(
        "test-key",
        "gemini-test",
        3.0,
        max_retries=max_retries,
        retry_base_delay_seconds=0.0,
    )


def test_503_succeeds_after_retry(monkeypatch) -> None:
    calls = 0

    def request(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise http_error(503)
        return FakeHTTPResponse(response_payload())

    monkeypatch.setattr(llm_module, "urlopen", request)

    result = provider().generate("prompt")

    assert result.answer == "ok"
    assert calls == 2


def test_503_fails_after_max_retries(monkeypatch) -> None:
    calls = 0

    def request(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise http_error(503)

    monkeypatch.setattr(llm_module, "urlopen", request)

    with pytest.raises(LLMProviderError, match="HTTP status 503"):
        provider(max_retries=2).generate("prompt")

    assert calls == 3


def test_429_is_retried(monkeypatch) -> None:
    calls = 0

    def request(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise http_error(429)
        return FakeHTTPResponse(response_payload())

    monkeypatch.setattr(llm_module, "urlopen", request)

    assert provider().generate("prompt").answer == "ok"
    assert calls == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_permanent_http_errors_are_not_retried(monkeypatch, status: int) -> None:
    calls = 0

    def request(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise http_error(status)

    monkeypatch.setattr(llm_module, "urlopen", request)

    with pytest.raises(LLMProviderError, match=f"HTTP status {status}"):
        provider().generate("prompt")

    assert calls == 1


def test_connection_reset_is_retried(monkeypatch) -> None:
    calls = 0

    def request(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise URLError(ConnectionResetError("connection reset"))
        return FakeHTTPResponse(response_payload())

    monkeypatch.setattr(llm_module, "urlopen", request)

    assert provider().generate("prompt").answer == "ok"
    assert calls == 2


def test_retry_delay_is_bounded_and_exponential(monkeypatch) -> None:
    delays: list[float] = []

    def request(*args: object, **kwargs: object) -> object:
        raise http_error(503)

    monkeypatch.setattr(llm_module, "urlopen", request)
    monkeypatch.setattr(llm_module.time, "sleep", delays.append)

    with pytest.raises(LLMProviderError):
        GeminiLLMProvider(
            "test-key",
            "gemini-test",
            3.0,
            max_retries=2,
            retry_base_delay_seconds=0.25,
        ).generate("prompt")

    assert delays == [0.25, 0.5]
