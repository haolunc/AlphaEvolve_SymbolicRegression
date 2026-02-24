# Sampler

```{contents}
:local:
:depth: 2
```

---

## Overview

The sampler module wraps LLM API calls behind a pluggable provider interface. The `LLM` class handles retry logic, cost tracking, and optional concurrent sampling, while `LLMProvider` subclasses implement the actual API calls.

---

## LLM Class

`LLM` is the public API. It takes a `SamplerConfig` and provides one method:

```python
llm = LLM(config=sampler_config)
responses: Collection[LLMResponse] | None = llm.draw_samples(prompt)
```

**Key behaviors:**
- **Retry logic** — each request is retried up to `max_retries` times with `retry_delay_seconds` between attempts. If all retries fail, raises `LLMProviderError`.
- **Cost tracking** — token costs are computed from `cost_per_ktoken` and attached to each `LLMResponse`.
- **Concurrent sampling** — when `samples_per_prompt > 1`, a `ThreadPoolExecutor` dispatches multiple requests in parallel.

---

## Provider Abstraction

All providers implement the `LLMProvider` abstract base class:

```python
class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, config: SamplerConfig) -> LLMResponse:
        """Send prompt to the LLM and return a structured response."""
```

Providers are registered in the `_PROVIDERS` dict and instantiated by name via `_make_provider()`.

---

## Built-in Providers

| Provider | Class | Default Model | API Key Env Var |
|----------|-------|---------------|-----------------|
| `"openai"` | `OpenAIProvider` | `gpt-5-mini` | `OPENAI_API_KEY` |
| `"qwen"` | `QwenProvider` | `qwen3-max` | `QWEN_API_KEY` + `QWEN_BASE_URL` |
| `"gemini"` | `GeminiProvider` | `gemini-3-pro-preview` | `GOOGLE_API_KEY` |

Provider selection is controlled by `SamplerConfig.provider`. The default model can be overridden with `SamplerConfig.model_name`.

---

## How to Add a New Provider

1. **Create a subclass** of `LLMProvider`:

```python
class MyProvider(LLMProvider):
    def generate(self, prompt: str, config: SamplerConfig) -> LLMResponse:
        # Call your API here
        return LLMResponse(
            response_text=...,
            input_tokens=...,
            output_tokens=...,
        )
```

2. **Register it** in the `_PROVIDERS` dict in `sampler.py`:

```python
_PROVIDERS: dict[str, type[LLMProvider]] = {
    "openai": OpenAIProvider,
    "qwen": QwenProvider,
    "gemini": GeminiProvider,
    "myprovider": MyProvider,  # ← add here
}
```

3. **Use it** in your config YAML:

```yaml
sampler:
  provider: myprovider
  model_name: my-model-id
```

---

## Concurrent Sampling

When `samples_per_prompt > 1`, the `LLM` class creates a persistent `ThreadPoolExecutor` with `samples_per_prompt` workers. Each call to `draw_samples()` submits that many parallel requests and collects results via `as_completed()`. Failed individual requests are logged but don't abort the batch.

When `samples_per_prompt == 1` (the default), no executor is created — the request runs directly in the calling thread.
