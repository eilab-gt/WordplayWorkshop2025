#!/usr/bin/env python3
"""Test script to verify LLM provider configuration."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lit_review.llm_providers import LLMProvider
from src.lit_review.utils import load_config


def test_provider():
    """Test the configured LLM provider."""
    print("🧪 Testing LLM Provider Configuration\n")

    try:
        # Load configuration
        config = load_config("config/config.yaml")
        print("✓ Configuration loaded")
        print(f"  Provider: {config.llm_provider}")
        print(f"  Model: {config.llm_model}\n")

        # Initialize provider
        provider = LLMProvider(config)
        print("✓ Provider initialized\n")

        # Test with a simple completion
        print("📝 Testing completion...")
        response = provider.chat_completion(
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer concisely.",
                },
                {
                    "role": "user",
                    "content": "What LLM are you and what's your model name? Answer in one sentence.",
                },
            ],
            temperature=0.1,
            max_tokens=100,
        )

        answer = response.choices[0].message.content
        print(f"✓ Response received: {answer}\n")

        # Test JSON extraction capability
        print("🔍 Testing JSON extraction...")
        response = provider.chat_completion(
            [
                {
                    "role": "system",
                    "content": "Extract information and return as JSON.",
                },
                {
                    "role": "user",
                    "content": 'Extract: The paper "LLM Wargaming" by Smith et al. (2024) is about AI. Return JSON with title and year.',
                },
            ],
            temperature=0.1,
            max_tokens=100,
        )

        json_response = response.choices[0].message.content
        print(f"✓ JSON response: {json_response}\n")

        # Show token usage if available
        if hasattr(response, "usage"):
            print("📊 Token usage:")
            print(f"  Prompt tokens: {response.usage.prompt_tokens}")
            print(f"  Completion tokens: {response.usage.completion_tokens}")
            print(f"  Total tokens: {response.usage.total_tokens}")

            # Estimate cost
            if hasattr(provider, "estimate_cost"):
                cost = provider.estimate_cost(
                    response.usage.prompt_tokens, response.usage.completion_tokens
                )
                print(f"  Estimated cost: ${cost:.4f}\n")

        print("✅ All tests passed! Your LLM provider is configured correctly.\n")

        # Test fallback if configured
        if config.llm_provider != "openai":
            print("🔄 Testing fallback to OpenAI...")
            try:
                fallback_response = provider.extract_with_fallback(
                    [{"role": "user", "content": "Hello"}], fallback_provider="openai"
                )
                if fallback_response:
                    print("✓ Fallback successful\n")
                else:
                    print(
                        "⚠️  Fallback not available (OpenAI key might not be configured)\n"
                    )
            except Exception as e:
                print(f"⚠️  Fallback test skipped: {e!s}\n")

    except Exception as e:
        print(f"\n❌ Error: {e!s}")
        print("\n🔍 Troubleshooting tips:")
        print("1. Check that config/config.yaml exists")
        print("2. Verify your .env file contains the required API key")
        print("3. Ensure the provider and model names are correct")
        print("4. Check your API key has sufficient credits/permissions")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(test_provider())
