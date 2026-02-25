# Sampler

```{contents}
:local:
:depth: 2
```

---

## Overview

The sampler module wraps LLM API calls behind a pluggable provider interface. The `LLM` class handles retry logic and cost tracking, while `LLMProvider` subclasses implement the actual API calls for each backend.

Concurrency is **not** managed by the `LLM` class -- the pipeline's `ThreadPoolExecutor` submits separate `_sampler_task` futures for each sample. This keeps the `LLM` class simple: one call in, one response out.

---

## LLM Class

`LLM` is the public API. It takes a `SamplerConfig` and exposes a single method:

```python
llm = LLM(config=sampler_config)
response: LLMResponse | None = llm.query(prompt)
```

### query()

Makes a single LLM call with retries. Returns `LLMResponse` on success, `None` on failure (no exception raised to the caller).

**Behavior:**

1. Call `self._provider.generate(prompt, config)` to get a raw `LLMResponse`
2. Compute token cost: $\text{cost} = \frac{\text{input\_tokens} \times c_{\text{in}} + \text{output\_tokens} \times c_{\text{out}}}{1000}$
3. Return a new `LLMResponse` with `token_cost` filled in
4. On exception: log, sleep `retry_delay_seconds`, retry up to `max_retries` times
5. If all retries fail: log the last error and return `None`

### Cost tracking

Cost per request is computed inline from `SamplerConfig.cost_per_ktoken`, a two-element list `[input_cost, output_cost]` representing dollars per 1K tokens. The computed cost is attached to the returned `LLMResponse.token_cost`.

---

## Provider Abstraction

All providers implement the `LLMProvider` abstract base class:

```python
class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, config: SamplerConfig) -> LLMResponse:
        """Send prompt to the LLM and return a structured response."""
```

Providers are registered in the `_PROVIDERS` dict and instantiated by name via `_make_provider(name)`.

---

## Built-in Providers

| Provider | Class | Default Model | API Key Env Var | Notes |
|----------|-------|---------------|-----------------|-------|
| `"openai"` | `OpenAIProvider` | `gpt-5-mini` | `OPENAI_API_KEY` | Uses `responses.create()` with reasoning effort `"medium"` |
| `"qwen"` | `QwenProvider` | `qwen3.5-plus` | `QWEN_API_KEY` + `QWEN_BASE_URL` | Enables `enable_thinking` for `qwen3-max-preview` model |
| `"gemini"` | `GeminiProvider` | `gemini-3-pro-preview` | `GOOGLE_API_KEY` | Uses `google.genai` client |

Provider selection is controlled by `SamplerConfig.provider` (default: `"qwen"`). The default model can be overridden with `SamplerConfig.model_name`.

---

## How to Add a Provider

1. **Create a subclass** of `LLMProvider`:

```python
class MyProvider(LLMProvider):
    def __init__(self) -> None:
        # Initialize your API client
        self._client = ...

    def generate(self, prompt: str, config: SamplerConfig) -> LLMResponse:
        # Call your API, return an LLMResponse (token_cost can be 0.0)
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
    "myprovider": MyProvider,  # add here
}
```

3. **Use it** in your config YAML:

```yaml
sampler:
  provider: myprovider
  model_name: my-model-id
```

---

## Concurrency Note

The `LLM` class makes **one request per call** -- it has no internal threading or batching. Concurrency is managed entirely by the pipeline in `cli.py`:

- `SamplerConfig.samples_per_prompt` controls how many samples to draw per prompt
- The pipeline submits `samples_per_prompt` separate `_sampler_task` futures to a `ThreadPoolExecutor`
- Each future calls `llm.query(prompt_code)` independently

This design keeps the sampler stateless and easy to test, while the pipeline controls parallelism.
