"""LLM sampling with pluggable provider backends."""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from collections.abc import Collection
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from .config import SamplerConfig
from .exceptions import LLMProviderError
from .logging_config import get_logger
from .messages import LLMResponse

logger = get_logger("sampler")


# ---------------------------------------------------------------------------
# Provider interface and implementations (Phase 5)
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    """Abstract base for LLM provider implementations."""

    @abstractmethod
    def generate(self, prompt: str, config: SamplerConfig) -> LLMResponse:
        """Send *prompt* to the LLM and return a structured response."""


class OpenAIProvider(LLMProvider):
    """Provider for the OpenAI API."""

    def generate(self, prompt: str, config: SamplerConfig) -> LLMResponse:
        from openai import OpenAI

        model = config.model_name or "gpt-5-mini"
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.responses.create(
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

    def generate(self, prompt: str, config: SamplerConfig) -> LLMResponse:
        from openai import OpenAI

        model = config.model_name or "qwen3-max"
        client = OpenAI(
            api_key=os.getenv("QWEN_API_KEY"),
            base_url=os.getenv("QWEN_BASE_URL"),
        )

        kwargs: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.temperature,
            "stream": False,
            "timeout": config.request_timeout_seconds,
        }
        if model == "qwen3-max-preview":
            kwargs["extra_body"] = {"enable_thinking": True}

        completion = client.chat.completions.create(**kwargs)
        return LLMResponse(
            response_text=completion.choices[0].message.content,
            input_tokens=completion.usage.prompt_tokens,
            output_tokens=completion.usage.completion_tokens,
        )


class GeminiProvider(LLMProvider):
    """Provider for Google Gemini models."""

    def generate(self, prompt: str, config: SamplerConfig) -> LLMResponse:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        model_id = config.model_name or "gemini-3-pro-preview"

        response = client.models.generate_content(
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
        self._samples_per_prompt = self._config.samples_per_prompt
        self._cost_per_ktoken = self._config.cost_per_ktoken
        self._provider = _make_provider(self._config.provider)
        # Create a persistent executor for concurrent sampling (not Gemini, not single-sample)
        if self._samples_per_prompt > 1:
            self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
                max_workers=self._samples_per_prompt
            )
        else:
            self._executor = None

    def clean(self) -> None:
        """Shut down the thread pool executor."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
            logger.debug("LLM executor shutdown complete")

    def draw_samples(self, prompt: str) -> Collection[LLMResponse] | None:
        """Return predicted continuations of *prompt*."""
        if self._samples_per_prompt == 1:
            try:
                sample_info = self._query_with_retry(prompt)
                if sample_info:
                    return [sample_info]
                return []
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error("Exception in draw_samples: %s", e)
                return None

        try:
            futures = {
                self._executor.submit(self._query_with_retry, prompt): i
                for i in range(self._samples_per_prompt)
            }
            results: list[LLMResponse] = []
            for future in as_completed(futures):
                try:
                    info = future.result()
                    results.append(info)
                except Exception as e:
                    logger.error("Request failed: %s", e)
            logger.debug("Collected %d samples", len(results))
            return results
        except KeyboardInterrupt:
            self.clean()
            raise
        except Exception as e:
            logger.error("Exception in draw_samples: %s", e)
            return None

    def _query_with_retry(self, prompt: str) -> LLMResponse | None:
        """Query the LLM provider with retry logic."""
        cfg = self._config
        for attempt in range(cfg.max_retries):
            try:
                resp = self._provider.generate(prompt, cfg)
                resp.token_cost = (
                    resp.input_tokens * self._cost_per_ktoken[0]
                    + resp.output_tokens * self._cost_per_ktoken[1]
                ) / 1000
                return resp
            except Exception as e:
                logger.warning("Request attempt %d failed: %s. Retrying...", attempt + 1, e)
                time.sleep(cfg.retry_delay_seconds)
        raise LLMProviderError(
            f"All {cfg.max_retries} retries failed for LLM request"
        )
