"""Tests for the LLMProvider module using LiteLLM."""

from unittest.mock import MagicMock, Mock, patch
import pytest
import os
from src.lit_review.llm_providers import LLMProvider


class MockConfig:
    """Mock configuration for testing."""
    def __init__(self, provider="together", model=None):
        self.llm_provider = provider
        self.llm_model = model or "Qwen/Qwen2.5-72B-Instruct-Turbo"
        self.llm_temperature = 0.1
        self.llm_max_tokens = 4000
        self.debug = False
        self.llm_provider_configs = {}


class TestLLMProvider:
    """Test cases for LLMProvider class."""

    def test_init_with_together(self):
        """Test LLMProvider initialization with TogetherAI."""
        config = MockConfig(provider="together")
        
        with patch.dict(os.environ, {"TOGETHER_API_KEY": "test-key"}):
            provider = LLMProvider(config)
            
            assert provider.provider == "together"
            assert provider.model == "Qwen/Qwen2.5-72B-Instruct-Turbo"
            assert provider.temperature == 0.1
            assert provider.max_tokens == 4000

    def test_init_with_openai(self):
        """Test LLMProvider initialization with OpenAI."""
        config = MockConfig(provider="openai", model="gpt-4o")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = LLMProvider(config)
            
            assert provider.provider == "openai"
            assert provider.model == "gpt-4o"

    def test_get_model_string_together(self):
        """Test model string formatting for TogetherAI."""
        config = MockConfig(provider="together")
        provider = LLMProvider(config)
        
        model_string = provider._get_model_string()
        assert model_string == "together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo"

    def test_get_model_string_openai(self):
        """Test model string formatting for OpenAI."""
        config = MockConfig(provider="openai", model="gpt-4o")
        provider = LLMProvider(config)
        
        model_string = provider._get_model_string()
        assert model_string == "gpt-4o"

    @patch("litellm.completion")
    def test_chat_completion_success(self, mock_completion):
        """Test successful chat completion."""
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "4"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_completion.return_value = mock_response
        
        config = MockConfig()
        provider = LLMProvider(config)
        
        messages = [{"role": "user", "content": "What is 2+2?"}]
        response = provider.chat_completion(messages)
        
        assert response.choices[0].message.content == "4"
        mock_completion.assert_called_once()
        
        # Check the model string was correctly formatted
        call_args = mock_completion.call_args[1]
        assert call_args["model"] == "together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo"

    @patch("litellm.completion")
    def test_chat_completion_with_json_mode(self, mock_completion):
        """Test chat completion with JSON response format."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"answer": 4}'
        mock_completion.return_value = mock_response
        
        config = MockConfig(provider="openai", model="gpt-4o")
        provider = LLMProvider(config)
        
        messages = [{"role": "user", "content": "What is 2+2? Reply in JSON."}]
        response = provider.chat_completion(
            messages,
            response_format={"type": "json_object"}
        )
        
        # For OpenAI GPT-4, response_format should be passed
        call_args = mock_completion.call_args[1]
        assert "response_format" in call_args

    @patch("litellm.completion")
    def test_extract_with_fallback(self, mock_completion):
        """Test extraction with fallback to another provider."""
        # First call fails
        mock_completion.side_effect = [
            Exception("API Error"),
            Mock(choices=[Mock(message=Mock(content="Fallback response"))])
        ]
        
        config = MockConfig(provider="together")
        with patch.dict(os.environ, {
            "TOGETHER_API_KEY": "test-key",
            "OPENAI_API_KEY": "test-key"
        }):
            provider = LLMProvider(config)
            
            messages = [{"role": "user", "content": "Test"}]
            response = provider.extract_with_fallback(messages, "openai")
            
            assert response.choices[0].message.content == "Fallback response"
            assert mock_completion.call_count == 2

    def test_supports_json_mode(self):
        """Test JSON mode support detection."""
        # OpenAI GPT-4 should support JSON mode
        config = MockConfig(provider="openai", model="gpt-4o")
        provider = LLMProvider(config)
        assert provider._supports_json_mode() is True
        
        # TogetherAI models don't support structured output directly
        config = MockConfig(provider="together")
        provider = LLMProvider(config)
        assert provider._supports_json_mode() is False

    def test_estimate_cost(self):
        """Test cost estimation."""
        config = MockConfig(provider="openai", model="gpt-4o")
        provider = LLMProvider(config)
        
        # 1000 prompt tokens + 500 completion tokens
        cost = provider.estimate_cost(1000, 500)
        
        # GPT-4o: $0.005/1K prompt + $0.015/1K completion
        expected = (1000/1000 * 0.005) + (500/1000 * 0.015)
        assert cost == expected

    def test_setup_provider_with_env_vars(self):
        """Test provider setup with environment variables."""
        config = MockConfig(provider="together")
        
        with patch.dict(os.environ, {"TOGETHER_API_KEY": "env-test-key"}):
            provider = LLMProvider(config)
            assert os.environ.get("TOGETHER_API_KEY") == "env-test-key"

    def test_litellm_configuration(self):
        """Test LiteLLM configuration settings."""
        config = MockConfig()
        config.debug = True
        
        with patch("litellm.set_verbose") as mock_verbose:
            provider = LLMProvider(config)
            
            # Check that caching is enabled
            from litellm import cache
            assert cache is not None

    @patch("litellm.completion")
    def test_error_handling(self, mock_completion):
        """Test error handling in chat completion."""
        mock_completion.side_effect = Exception("API Error")
        
        config = MockConfig()
        provider = LLMProvider(config)
        
        messages = [{"role": "user", "content": "Test"}]
        
        with pytest.raises(Exception) as exc_info:
            provider.chat_completion(messages)
        
        assert "API Error" in str(exc_info.value)

    def test_get_fallback_model(self):
        """Test fallback model selection."""
        config = MockConfig()
        provider = LLMProvider(config)
        
        # Test each provider's fallback
        assert provider._get_fallback_model("openai") == "gpt-3.5-turbo"
        assert provider._get_fallback_model("anthropic") == "claude-3-haiku-20240307"
        assert provider._get_fallback_model("google") == "gemini-pro"
        assert provider._get_fallback_model("together") == "Qwen/Qwen2.5-72B-Instruct-Turbo"
        assert provider._get_fallback_model("unknown") == "gpt-3.5-turbo"


class TestProviderIntegration:
    """Integration tests for different providers."""

    @pytest.mark.skipif(not os.getenv("TOGETHER_API_KEY"), reason="No TogetherAI key")
    def test_together_real_api(self):
        """Test real TogetherAI API call (requires API key)."""
        config = MockConfig(provider="together")
        provider = LLMProvider(config)
        
        messages = [
            {"role": "user", "content": "What is 2+2? Reply with just the number."}
        ]
        
        response = provider.chat_completion(
            messages,
            temperature=0,
            max_tokens=10
        )
        
        assert hasattr(response, 'choices')
        assert len(response.choices) > 0
        content = response.choices[0].message.content.strip()
        assert "4" in content

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OpenAI key")
    def test_openai_real_api(self):
        """Test real OpenAI API call (requires API key)."""
        config = MockConfig(provider="openai", model="gpt-3.5-turbo")
        provider = LLMProvider(config)
        
        messages = [
            {"role": "user", "content": "What is 2+2? Reply with just the number."}
        ]
        
        response = provider.chat_completion(
            messages,
            temperature=0,
            max_tokens=10
        )
        
        assert hasattr(response, 'choices')
        content = response.choices[0].message.content.strip()
        assert "4" in content