# 🔑 LLM Provider Setup Guide

This guide explains how to configure multiple LLM providers for the literature review pipeline, with secure API key management.

## 📋 Overview

The pipeline now supports multiple LLM providers through LiteLLM:
- **OpenAI** (GPT-4, GPT-3.5)
- **Anthropic** (Claude 3 family)
- **Google** (Gemini, Vertex AI)
- **TogetherAI** (Open source models)

## 🔒 Secure API Key Setup

### Step 1: Create Environment File

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your API keys:
```bash
# Primary LLM Provider Keys
OPENAI_API_KEY=sk-your-actual-openai-key
ANTHROPIC_API_KEY=sk-ant-your-actual-anthropic-key
GOOGLE_API_KEY=your-actual-google-key
TOGETHER_API_KEY=your-actual-together-key

# Other Service Keys
SEMANTIC_SCHOLAR_API_KEY=your-actual-key
UNPAYWALL_EMAIL=your@email.com
```

**⚠️ IMPORTANT**:
- Never commit `.env` to version control
- `.env` is already in `.gitignore` for your protection
- Keep your API keys secret and rotate them regularly

### Step 2: Configure Your Provider

1. Copy the example configuration:
```bash
cp config/config.yaml.example config/config.yaml
```

2. Edit `config/config.yaml` to select your provider:
```yaml
llm:
  # Choose your provider: openai, anthropic, google, together
  provider: "anthropic"  # Changed from default "openai"

  # The model will be auto-selected based on provider
  # Or you can specify it explicitly:
  model: "claude-3-sonnet-20240229"

  temperature: 0.1
  max_tokens: 4000
```

**Note**: `config/config.yaml` is also in `.gitignore` to protect your configuration.

## 🚀 Provider-Specific Setup

### OpenAI
```yaml
llm:
  provider: "openai"
  model: "gpt-4o"  # or "gpt-4", "gpt-3.5-turbo"
```

Required environment variable:
- `OPENAI_API_KEY`

### Anthropic (Claude)
```yaml
llm:
  provider: "anthropic"
  model: "claude-3-sonnet-20240229"  # or "claude-3-opus-20240229"
```

Required environment variable:
- `ANTHROPIC_API_KEY`

### Google (Gemini/Vertex AI)
```yaml
llm:
  provider: "google"
  model: "gemini-pro"  # or "gemini-pro-vision"
```

Required environment variables:
- `GOOGLE_API_KEY`
- `GOOGLE_PROJECT_ID` (optional, for Vertex AI)

### TogetherAI
```yaml
llm:
  provider: "together"
  model: "Qwen/Qwen2.5-72B-Instruct-Turbo"
```

Required environment variable:
- `TOGETHER_API_KEY`

## 🧪 Testing Your Setup

1. Install the updated dependencies:
```bash
uv pip install -e .
```

2. Test your configuration:
```python
from src.lit_review.utils import load_config
from src.lit_review.llm_providers import LLMProvider

# Load configuration
config = load_config('config/config.yaml')

# Initialize provider
provider = LLMProvider(config)

# Test with a simple completion
response = provider.chat_completion([
    {"role": "user", "content": "Hello, which LLM are you?"}
])
print(response.choices[0].message.content)
```

## 💰 Cost Considerations

Different providers have different pricing:

| Provider | Model | Input Cost | Output Cost |
|----------|-------|------------|-------------|
| OpenAI | GPT-4o | $5/1M tokens | $15/1M tokens |
| OpenAI | GPT-3.5 | $0.50/1M tokens | $1.50/1M tokens |
| Anthropic | Claude 3 Sonnet | $3/1M tokens | $15/1M tokens |
| Google | Gemini Pro | $0.25/1M tokens | $0.50/1M tokens |
| TogetherAI | Mixtral 8x7B | $0.25/1M tokens | $0.25/1M tokens |

*Prices as of 2024, check provider websites for current pricing*

## 🔄 Switching Providers

To switch providers, simply:

1. Ensure you have the API key in `.env`
2. Update `provider` in `config/config.yaml`
3. Optionally update the `model` setting
4. Run your pipeline as normal

The code automatically handles provider differences!

## 🛡️ Security Best Practices

1. **Use environment variables**: Never hardcode API keys
2. **Rotate keys regularly**: Set reminders to update keys
3. **Limit key permissions**: Use read-only keys where possible
4. **Monitor usage**: Check your provider dashboards regularly
5. **Use separate keys**: Different keys for dev/prod environments

## 🆘 Troubleshooting

### "API key not found" error
- Check that your `.env` file exists and contains the key
- Ensure the environment variable name matches exactly
- Try: `echo $OPENAI_API_KEY` to verify it's loaded

### "Model not found" error
- Verify the model name is correct for your provider
- Check provider documentation for available models
- Some models require special access (e.g., GPT-4)

### Rate limiting errors
- Add delays between requests in config
- Consider using a cheaper/faster model
- Implement exponential backoff (already in LiteLLM)

## 📚 Additional Resources

- [LiteLLM Documentation](https://docs.litellm.ai/)
- [OpenAI API Docs](https://platform.openai.com/docs)
- [Anthropic API Docs](https://docs.anthropic.com/)
- [Google AI Docs](https://ai.google.dev/)
- [TogetherAI Docs](https://docs.together.ai/)
