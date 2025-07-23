#!/usr/bin/env python3
"""Setup script for configuring Google Gemini API access."""

import os
from pathlib import Path

import litellm


def test_gemini_access():
    """Test if Gemini API is properly configured."""
    print("Testing Google Gemini API access...")

    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("\n❌ Gemini API key not found!")
        print("\nTo get a Gemini API key:")
        print("1. Go to https://makersuite.google.com/app/apikey")
        print("2. Create a new API key")
        print("3. Set it as an environment variable:")
        print("   export GEMINI_API_KEY='your-key-here'")
        print("\nOr add it to a .env file:")
        print("   GEMINI_API_KEY=your-key-here")
        return False

    print(f"✓ API key found: {api_key[:8]}...")

    # Test the API
    try:
        response = litellm.completion(
            model="gemini/gemini-pro",
            messages=[
                {"role": "user", "content": "Say 'Hello, World!' in one sentence."}
            ],
            temperature=0.1,
            max_tokens=50,
        )

        print("✓ API test successful!")
        print(f"Response: {response.choices[0].message.content}")
        return True

    except Exception as e:
        print(f"\n❌ API test failed: {e}")
        print("\nPossible issues:")
        print("- Invalid API key")
        print("- API not enabled for your Google Cloud project")
        print("- Rate limits or quota exceeded")
        return False


def create_env_file():
    """Create or update .env file with Gemini API key."""
    env_path = Path(".env")

    api_key = input("\nEnter your Gemini API key: ").strip()

    if not api_key:
        print("No API key provided.")
        return

    # Read existing .env
    existing_lines = []
    if env_path.exists():
        with open(env_path) as f:
            existing_lines = f.readlines()

    # Update or add GEMINI_API_KEY
    key_found = False
    new_lines = []

    for line in existing_lines:
        if line.strip().startswith("GEMINI_API_KEY="):
            new_lines.append(f"GEMINI_API_KEY={api_key}\n")
            key_found = True
        else:
            new_lines.append(line)

    if not key_found:
        new_lines.append(f"\nGEMINI_API_KEY={api_key}\n")

    # Write back
    with open(env_path, "w") as f:
        f.writelines(new_lines)

    print("\n✓ Updated .env file with Gemini API key")

    # Test the key
    os.environ["GEMINI_API_KEY"] = api_key
    test_gemini_access()


def show_supported_models():
    """Show all Gemini models supported by LiteLLM."""
    print("\nSupported Gemini models:")
    print("- gemini/gemini-pro (recommended for text extraction)")
    print("- gemini/gemini-1.5-flash (faster, lower cost)")
    print("- gemini/gemini-1.5-pro (most capable)")
    print(
        "\nNote: Some models may require additional setup or have different rate limits."
    )


def main():
    """Main setup function."""
    print("=== Google Gemini API Setup ===\n")

    # Check current status
    if test_gemini_access():
        print("\n✅ Gemini API is already configured and working!")
        show_supported_models()
        return

    # Offer to set up
    print("\nWould you like to set up Gemini API access now?")
    choice = input("Enter 'y' to continue: ").strip().lower()

    if choice == "y":
        create_env_file()
        show_supported_models()
    else:
        print("\nTo manually set up:")
        print("1. Get API key from https://makersuite.google.com/app/apikey")
        print("2. Add to .env file: GEMINI_API_KEY=your-key-here")
        print("3. Run this script again to test")


if __name__ == "__main__":
    main()
