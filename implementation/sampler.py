"""Class for sampling new programs with LLM."""
from collections.abc import Collection
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from dotenv import load_dotenv
from openai import OpenAI
from logging_utils import setup_logger

load_dotenv()  # load .env file
logger = setup_logger(__name__)

class LLM:
    """Language model that predicts continuation of provided source code."""

    def __init__(self, samples_per_prompt: int, llm_provider: str = "qwen") -> None:
        self._samples_per_prompt = samples_per_prompt
        self._executor = None
        self.llm_provider = llm_provider

    def clean(self):
        """Clean up resources."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
            logger.debug("LLM executor shutdown complete")

    def draw_samples(self, prompt: str) -> Collection[str]:
        """Returns multiple predicted continuations of `prompt`."""
        # For Gemini, don't use ThreadPoolExecutor, always return single sample
        if self.llm_provider == "gemini":
            try:
                sample_info = self.query_llm_with_retry(prompt)
                if sample_info:
                    return [sample_info]
                return []
            except KeyboardInterrupt:
                logger.warning("KeyboardInterrupt received during LLM sampling")
                raise
            except Exception as e:
                logger.error(f"Exception in draw_samples: {e}")
                return None

        if self._samples_per_prompt==1:
            try:
                sample_info = self.query_llm_with_retry(prompt)
                logger.debug(f"Successfully collected 1 sample")
                if sample_info:
                    return [sample_info]
                return []
            except KeyboardInterrupt:
                logger.warning("KeyboardInterrupt received during LLM sampling")
                raise
            except Exception as e:
                logger.error(f"Exception in draw_samples: {e}")
                return None
        else:

            # Execute requests in parallel using ThreadPoolExecutor
            self._executor = ThreadPoolExecutor(max_workers=self._samples_per_prompt)
            try:
                # Submit all requests
                future_to_request = {
                    self._executor.submit(self.query_llm_with_retry, prompt): i
                    for i in range(self._samples_per_prompt)
                }

                # Collect results as they complete
                all_samples_info = []
                for future in as_completed(future_to_request):
                    try:
                        sample_info = future.result()
                        all_samples_info.append(sample_info)
                    except Exception as e:
                        logger.error(f"Request failed with error: {e}")
                logger.debug(f"Successfully collected {len(all_samples_info)} samples")
                return all_samples_info
            except KeyboardInterrupt:
                logger.warning("KeyboardInterrupt received during LLM sampling, cleaning up...")
                self.clean()
                raise
            except Exception as e:
                logger.error(f"Exception in draw_samples: {e}")
                return None
            finally:
                self.clean()


    def query_llm_with_retry(self, prompt: str, max_retries=5):
        for attempt in range(max_retries):
            try:
                if self.llm_provider == "openai":
                    model = "gpt-5-mini"
                    api_key = os.getenv("OPENAI_API_KEY")
                    client = OpenAI(
                        api_key=api_key,
                    )
                    response = client.responses.create(
                        model=model,
                        input=prompt,
                        reasoning={
                            "effort": "medium",
                            # "summary": "auto" # reasoning summary is not supported for personal use yet
                            }, # "low", "medium", "high" default is "medium"
                        # temperature=0.8, default is 1
                    )
                    response_info = {
                        "response_text": response.output_text,
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                        # "reasoning_summary": response.usage.output[0]["summary"][0]["text"],
                    }                    
                elif self.llm_provider == "qwen":
                    # model = "qwen-turbo-latest"
                    model = "qwen3-max"
                    api_key = os.getenv("QWEN_API_KEY")
                    base_url = os.getenv("QWEN_BASE_URL")
                    client = OpenAI(
                        api_key=api_key,
                        base_url=base_url,
                    )

                    # Enable thinking for qwen3-max-preview
                    if model == "qwen3-max-preview":
                        completion = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=1,
                            stream=False,
                            extra_body={"enable_thinking": True},
                            timeout=180  # 180 second timeout
                        )

                    else:
                        completion = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=1,
                            stream=False,
                            timeout=180  # 180 second timeout
                        )
                    response_text = completion.choices[0].message.content
                    response_info = {
                        "response_text": response_text,
                        "input_tokens": completion.usage.prompt_tokens,
                        "output_tokens": completion.usage.completion_tokens,
                    }

                elif self.llm_provider == "gemini":
                    response_text = completion.choices[0].message.content

                    response_info = {
                        "response_text": response_text,
                        "input_tokens": completion.usage.prompt_tokens,
                        "output_tokens": completion.usage.completion_tokens,
                    }

                    from google import genai
                    from google.genai import types

                    api_key = os.getenv("GOOGLE_API_KEY")
                    client = genai.Client(api_key=api_key)
                    model_id = "gemini-3-pro-preview"

                    response = client.models.generate_content(
                        model=model_id,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.7,
                        )
                    )

                    response_info = {
                        "response_text": response.text,
                        "input_tokens": response.usage_metadata.prompt_token_count,
                        "output_tokens": response.usage_metadata.candidates_token_count,
                    }

                # logger.info(f"Response: {response_info}")
                return response_info
            except Exception as e:
                logger.warning(f"Request attempt {attempt+1} failed: {e}. Retrying...")
                time.sleep(5)
                continue
        logger.error("All retries failed for LLM request")
        return None

