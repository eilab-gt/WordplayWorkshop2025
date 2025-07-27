# 🔑 API Key Setup Guide

## Current Status

Based on your `.env` file, you have configured:
- ✅ OpenAI API Key 
- ✅ TogetherAI API Key
- ✅ Semantic Scholar API Key
- ✅ Unpaywall Email

You still need:
- ❌ Anthropic API Key (for Claude)
- ❌ Google API Key (for Gemini)

## Getting Missing API Keys

### Anthropic (Claude)
1. Visit [console.anthropic.com](https://console.anthropic.com)
2. Sign up or log in
3. Go to API Keys section
4. Create a new API key
5. Replace `sk-ant-your-anthropic-api-key-here` in your `.env`

### Google (Gemini)
1. Visit [makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
2. Sign in with Google account
3. Click "Get API Key"
4. Create new project or select existing
5. Replace `your-google-api-key-here` in your `.env`

## Configuration Best Practices

### Step 1: Update Your .env
Remove the LLM-specific settings from your `.env`:
```bash
# Remove these lines:
LLM_MODEL=gpt-4o
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=4000
```

### Step 2: Create config.yaml
```bash
cp config/config.yaml.example config/config.yaml
```

### Step 3: Select Your Provider
Edit `config/config.yaml`:
```yaml
llm:
  # Choose: openai, anthropic, google, together
  provider: "openai"  # You can use this now
  
  # Or try TogetherAI for open-source models:
  # provider: "together"
```

## Available Providers

### With Your Current Keys

#### OpenAI (Ready to Use)
```yaml
llm:
  provider: "openai"
  model: "gpt-4o"  # or "gpt-3.5-turbo" for cheaper option
```

#### TogetherAI (Ready to Use)
```yaml
llm:
  provider: "together"
  model: "Qwen/Qwen2.5-72B-Instruct-Turbo"
  # Other options:
  # - "meta-llama/Llama-3.3-70B-Instruct"
  # - "deepseek-ai/DeepSeek-V3"
```

### After Getting Missing Keys

#### Anthropic
```yaml
llm:
  provider: "anthropic"
  model: "claude-3-sonnet-20240229"
  # Or: "claude-3-opus-20240229" (more powerful)
  # Or: "claude-3-haiku-20240307" (cheaper)
```

#### Google
```yaml
llm:
  provider: "google"
  model: "gemini-pro"
  # Or: "gemini-pro-vision" (multimodal)
```

## Testing Your Setup

1. **Test current providers**:
```bash
# Test OpenAI
uv run python scripts/test_llm_provider.py --provider openai

# Test TogetherAI
uv run python scripts/test_llm_provider.py --provider together
```

2. **After adding new keys**:
```bash
# Test all providers
uv run python scripts/test_llm_provider.py --all
```

## Cost Comparison

| Provider | Model | Input Cost | Output Cost | Notes |
|----------|-------|------------|-------------|-------|
| OpenAI | GPT-4o | $5/1M | $15/1M | Best quality |
| OpenAI | GPT-3.5 | $0.50/1M | $1.50/1M | Good balance |
| Together | Qwen2.5 72B Turbo | ~$0.30/1M | ~$0.30/1M | Best value for extraction |
| Anthropic | Claude 3 Sonnet | $3/1M | $15/1M | Strong alternative |
| Google | Gemini Pro | $0.25/1M | $0.50/1M | Cheapest major provider |

## Recommendations

1. **For Quality**: Use OpenAI GPT-4o (you're ready!)
2. **For Best Value**: Use TogetherAI Qwen2.5 (you're ready!)
3. **For Balance**: Get Anthropic key for Claude 3 Sonnet
4. **For Experimentation**: Get Google key for Gemini Pro

## Security Reminders

- ✅ Your `.env` is already in `.gitignore`
- ✅ Never share API keys in code or commits
- ✅ Rotate keys regularly
- ✅ Use different keys for dev/prod