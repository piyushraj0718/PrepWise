from typing import Protocol


class AssessmentLLMProvider(Protocol):
    model_name: str

    def generate_structured(
        self, prompt: str, response_schema: dict[str, object]
    ) -> object:
        ...
