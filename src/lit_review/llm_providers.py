"""Multi-provider LLM service using LiteLLM for unified API access."""

import logging
import os
from typing import Any, Optional

import litellm
from litellm import completion

logger = logging.getLogger(__name__)


class LLMProvider:
    """Unified LLM provider interface using LiteLLM."""

    def __init__(self, config):
        """Initialize LLM provider with configuration.

        Args:
            config: Configuration object with LLM settings
        """
        self.config = config
        self.provider = config.llm_provider
        self.model = config.llm_model
        self.temperature = config.llm_temperature
        self.max_tokens = config.llm_max_tokens

        # Configure LiteLLM
        self._configure_litellm()

        # Set up provider-specific configuration
        self._setup_provider()

        logger.info(
            f"Initialized LLM provider: {self.provider} with model: {self.model}"
        )

    def _configure_litellm(self):
        """Configure LiteLLM settings."""
        # Enable caching for cost optimization
        litellm.cache = litellm.Cache()

        # Set timeout
        litellm.request_timeout = 120.0  # 2 minutes

        # Enable retries
        litellm.num_retries = 3

        # Set logging level
        if self.config.debug:
            litellm.set_verbose = True

    def _setup_provider(self):
        """Set up provider-specific configuration."""
        provider_config = self.config.llm_provider_configs.get(self.provider, {})

        # Set API keys based on provider
        if self.provider == "openai":
            api_key = provider_config.get("api_key") or os.getenv("OPENAI_API_KEY")
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key

        elif self.provider == "anthropic":
            api_key = provider_config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                os.environ["ANTHROPIC_API_KEY"] = api_key

        elif self.provider == "google":
            api_key = provider_config.get("api_key") or os.getenv("GOOGLE_API_KEY")
            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key
            # For Vertex AI
            project_id = provider_config.get("project_id") or os.getenv(
                "GOOGLE_PROJECT_ID"
            )
            if project_id:
                os.environ["GOOGLE_PROJECT_ID"] = project_id

        elif self.provider == "together":
            api_key = provider_config.get("api_key") or os.getenv("TOGETHER_API_KEY")
            if api_key:
                os.environ["TOGETHER_API_KEY"] = api_key

        # Override model if specified in provider config
        if "model" in provider_config:
            self.model = provider_config["model"]

    def _get_model_string(self) -> str:
        """Get the correct model string for LiteLLM based on provider.

        Returns:
            Model string in LiteLLM format
        """
        # LiteLLM uses provider prefixes for non-OpenAI models
        if self.provider == "openai":
            return self.model
        elif self.provider == "anthropic":
            return (
                "claude-3-sonnet-20240229" if "claude" not in self.model else self.model
            )
        elif self.provider == "google":
            return (
                f"vertex_ai/{self.model}"
                if "vertex_ai/" not in self.model
                else self.model
            )
        elif self.provider == "together":
            return (
                f"together_ai/{self.model}"
                if "together_ai/" not in self.model
                else self.model
            )
        else:
            # Default to using model as-is
            return self.model

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Create a chat completion using the configured provider.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max tokens
            response_format: Response format specification (e.g., {"type": "json_object"})

        Returns:
            Completion response in OpenAI format
        """
        try:
            model_string = self._get_model_string()
            logger.debug(f"Making completion request to {model_string}")

            # Build completion arguments
            completion_args = {
                "model": model_string,
                "messages": messages,
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
            }

            # Add response format if specified and supported
            if response_format and self._supports_json_mode():
                completion_args["response_format"] = response_format

            # Make the completion request
            response = completion(**completion_args)

            # Log token usage for monitoring
            if hasattr(response, "usage"):
                logger.debug(
                    f"Token usage - Prompt: {response.usage.prompt_tokens}, "
                    f"Completion: {response.usage.completion_tokens}, "
                    f"Total: {response.usage.total_tokens}"
                )

            return response

        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise

    def _supports_json_mode(self) -> bool:
        """Check if the current provider/model supports JSON response format.

        Returns:
            True if JSON mode is supported
        """
        # OpenAI GPT-4 and GPT-3.5 support JSON mode
        if self.provider == "openai" and (
            "gpt-4" in self.model or "gpt-3.5" in self.model
        ):
            return True

        # Anthropic Claude 3 supports JSON mode through prompting
        if self.provider == "anthropic" and "claude-3" in self.model:
            return False  # Handle through prompting instead

        # Other providers generally don't support structured output directly
        return False

    def extract_with_fallback(
        self,
        messages: list[dict[str, str]],
        fallback_provider: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Try extraction with fallback to another provider on failure.

        Args:
            messages: List of message dictionaries
            fallback_provider: Alternative provider to try on failure

        Returns:
            Completion response or None if all attempts fail
        """
        try:
            # Try primary provider
            return self.chat_completion(messages)

        except Exception as e:
            logger.warning(f"Primary provider {self.provider} failed: {e}")

            if fallback_provider and fallback_provider != self.provider:
                logger.info(f"Attempting fallback to {fallback_provider}")

                # Temporarily switch provider
                original_provider = self.provider
                original_model = self.model

                try:
                    self.provider = fallback_provider
                    self._setup_provider()
                    self.model = self._get_fallback_model(fallback_provider)

                    response = self.chat_completion(messages)

                    # Restore original settings
                    self.provider = original_provider
                    self.model = original_model
                    self._setup_provider()

                    return response

                except Exception as fallback_error:
                    logger.error(
                        f"Fallback provider {fallback_provider} also failed: {fallback_error}"
                    )

                    # Restore original settings
                    self.provider = original_provider
                    self.model = original_model
                    self._setup_provider()

            return None

    def _get_fallback_model(self, provider: str) -> str:
        """Get appropriate fallback model for a provider.

        Args:
            provider: Provider name

        Returns:
            Model name for the provider
        """
        fallback_models = {
            "openai": "gpt-3.5-turbo",  # Cheaper, faster fallback
            "anthropic": "claude-3-haiku-20240307",  # Cheaper Claude model
            "google": "gemini-pro",
            "together": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        }
        return fallback_models.get(provider, "gpt-3.5-turbo")

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost for the completion based on provider pricing.

        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Estimated cost in USD
        """
        # Simplified pricing (as of 2024) - should be updated regularly
        pricing = {
            "gpt-4o": {"prompt": 0.005, "completion": 0.015},  # per 1K tokens
            "gpt-4": {"prompt": 0.03, "completion": 0.06},
            "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
            "claude-3-sonnet-20240229": {"prompt": 0.003, "completion": 0.015},
            "claude-3-haiku-20240307": {"prompt": 0.00025, "completion": 0.00125},
            "gemini-pro": {"prompt": 0.00025, "completion": 0.0005},
        }

        model_pricing = pricing.get(self.model, pricing["gpt-3.5-turbo"])

        prompt_cost = (prompt_tokens / 1000) * model_pricing["prompt"]
        completion_cost = (completion_tokens / 1000) * model_pricing["completion"]

        return prompt_cost + completion_cost
