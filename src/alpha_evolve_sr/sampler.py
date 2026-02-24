"""LLM sampling with pluggable provider backends."""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod

from dotenv import load_dotenv

from .config import SamplerConfig
from .logging_config import get_logger
from .messages import LLMResponse

logger = get_logger("sampler")


# ---------------------------------------------------------------------------
# Provider interface and implementations
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    """Abstract base for LLM provider implementations."""

    @abstractmethod
    def generate(self, prompt: str, config: SamplerConfig) -> LLMResponse:
        """Send *prompt* to the LLM and return a structured response."""


class OpenAIProvider(LLMProvider):
    """Provider for the OpenAI API."""

    def __init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(self, prompt: str, config: SamplerConfig) -> LLMResponse:
        model = config.model_name or "gpt-5-mini"
        response = self._client.responses.create(
            model=model,
            input=prompt,
            reasoning={"effort": "medium"},
        )
        return LLMResponse(
            response_text=response.output_text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


class QwenProvider(LLMProvider):
    """Provider for Qwen models via the OpenAI-compatible API."""

    def __init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            api_key=os.getenv("QWEN_API_KEY"),
            base_url=os.getenv("QWEN_BASE_URL"),
        )

    def generate(self, prompt: str, config: SamplerConfig) -> LLMResponse:
        model = config.model_name or "qwen3-max"

        kwargs: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.temperature,
            "stream": False,
            "timeout": config.request_timeout_seconds,
        }
        if model == "qwen3-max-preview":
            kwargs["extra_body"] = {"enable_thinking": True}

        completion = self._client.chat.completions.create(**kwargs)
        return LLMResponse(
            response_text=completion.choices[0].message.content,
            input_tokens=completion.usage.prompt_tokens,
            output_tokens=completion.usage.completion_tokens,
        )


class GeminiProvider(LLMProvider):
    """Provider for Google Gemini models."""

    def __init__(self) -> None:
        from google import genai

        self._client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    def generate(self, prompt: str, config: SamplerConfig) -> LLMResponse:
        from google.genai import types

        model_id = config.model_name or "gemini-3-pro-preview"

        response = self._client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=config.temperature),
        )
        return LLMResponse(
            response_text=response.text,
            input_tokens=response.usage_metadata.prompt_token_count,
            output_tokens=response.usage_metadata.candidates_token_count,
        )


_PROVIDERS: dict[str, type[LLMProvider]] = {
    "openai": OpenAIProvider,
    "qwen": QwenProvider,
    "gemini": GeminiProvider,
}


def _make_provider(name: str) -> LLMProvider:
    """Instantiate an ``LLMProvider`` by name."""
    cls = _PROVIDERS.get(name)
    if cls is None:
        raise ValueError(f"Unknown LLM provider {name!r}. Choose from {list(_PROVIDERS)}")
    return cls()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
class LLM:
    """Language model that predicts continuation of provided source code."""

    def __init__(
        self,
        config: SamplerConfig | None = None,
    ) -> None:
        load_dotenv()
        self._config = config or SamplerConfig()
        self._cost_per_ktoken = self._config.cost_per_ktoken
        self._provider = _make_provider(self._config.provider)

    def clean(self) -> None:
        """Release any held resources (currently a no-op)."""

    def query(self, prompt: str) -> LLMResponse | None:
        """Make a single LLM call with retries; return *None* on failure."""
        cfg = self._config
        last_error: Exception | None = None
        for attempt in range(cfg.max_retries):
            try:
                resp = self._provider.generate(prompt, cfg)
                cost = (
                    resp.input_tokens * self._cost_per_ktoken[0]
                    + resp.output_tokens * self._cost_per_ktoken[1]
                ) / 1000
                return LLMResponse(
                    response_text=resp.response_text,
                    input_tokens=resp.input_tokens,
                    output_tokens=resp.output_tokens,
                    token_cost=cost,
                )
            except Exception as e:
                last_error = e
                logger.warning("Request attempt %d failed: %s. Retrying...", attempt + 1, e)
                time.sleep(cfg.retry_delay_seconds)
        logger.error("All %d retries failed for LLM request. Last error: %s", cfg.max_retries, last_error)
        return None
