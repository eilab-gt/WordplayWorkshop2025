#!/usr/bin/env python3
"""Test script to verify LLM provider configuration."""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lit_review.llm_providers import LLMProvider
from src.lit_review.utils import load_config


def test_provider(provider_name=None, model_name=None):
    """Test a specific LLM provider."""
    print(f"🧪 Testing LLM Provider: {provider_name or 'configured'}\n")

    try:
        # Load configuration
        config = load_config("config/config.yaml")
        
        # Override provider if specified
        if provider_name:
            config.llm_provider = provider_name
            # Set default model for provider
            if not model_name:
                default_models = {
                    "openai": "gpt-3.5-turbo",
                    "anthropic": "claude-3-sonnet-20240229",
                    "google": "gemini-pro",
                    "together": "Qwen/Qwen2.5-72B-Instruct-Turbo"
                }
                config.llm_model = default_models.get(provider_name, config.llm_model)
            else:
                config.llm_model = model_name
        
        print("✓ Configuration loaded")
        print(f"  Provider: {config.llm_provider}")
        print(f"  Model: {config.llm_model}\n")

        # Check API key
        api_key_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "together": "TOGETHER_API_KEY"
        }
        
        key_var = api_key_vars.get(config.llm_provider)
        if key_var:
            if os.getenv(key_var):
                print(f"✓ API key found: {key_var}")
            else:
                print(f"❌ API key missing: {key_var}")
                print(f"   Please set {key_var} in your .env file")
                return False

        # Initialize provider
        provider = LLMProvider(config)
        print("✓ Provider initialized\n")

        # Test with a simple completion
        print("📝 Testing completion...")
        messages = [
            {"role": "user", "content": "What is 2+2? Reply with just the number."}
        ]

        response = provider.chat_completion(
            messages=messages,
            temperature=0,
            max_tokens=10
        )

        if hasattr(response, 'choices'):
            content = response.choices[0].message.content
            print(f"✓ Response: {content}")
        else:
            print(f"✓ Response received (dict format)")

        # Test extraction prompt
        print("\n📊 Testing extraction...")
        extraction_messages = [
            {
                "role": "system",
                "content": "Extract key information from academic papers."
            },
            {
                "role": "user",
                "content": "Paper: 'LLMs in Wargaming'. Extract: game_type (one word)"
            }
        ]

        response = provider.chat_completion(
            messages=extraction_messages,
            temperature=0.1,
            max_tokens=100
        )

        if hasattr(response, 'choices'):
            content = response.choices[0].message.content
            print(f"✓ Extraction: {content}")
        else:
            print(f"✓ Extraction response received")

        print(f"\n✅ Provider '{config.llm_provider}' is working correctly!")
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your .env file has the required API key")
        print("2. Ensure config/config.yaml exists")
        print("3. Verify your API key is valid")
        return False


def test_all_providers():
    """Test all configured providers."""
    providers = ["openai", "anthropic", "google", "together"]
    results = {}
    
    print("🧪 Testing All Providers\n")
    print("=" * 50)
    
    for provider in providers:
        print(f"\nTesting {provider}...")
        print("-" * 30)
        success = test_provider(provider)
        results[provider] = success
        print("=" * 50)
    
    # Summary
    print("\n📊 Summary:")
    for provider, success in results.items():
        status = "✅ Working" if success else "❌ Failed"
        print(f"  {provider}: {status}")


def main():
    parser = argparse.ArgumentParser(description="Test LLM provider configuration")
    parser.add_argument(
        "--provider", 
        choices=["openai", "anthropic", "google", "together"],
        help="Specific provider to test"
    )
    parser.add_argument(
        "--model",
        help="Specific model to test"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all providers"
    )
    
    args = parser.parse_args()
    
    if args.all:
        test_all_providers()
    else:
        test_provider(args.provider, args.model)


if __name__ == "__main__":
    main()