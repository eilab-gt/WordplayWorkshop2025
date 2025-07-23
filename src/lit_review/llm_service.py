"""Model-agnostic LLM service using LiteLLM and FastAPI."""

import os
import logging
from typing import Optional, Dict, Any, List
from enum import Enum
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import litellm
from litellm import completion
import uvicorn

logger = logging.getLogger(__name__)

# Configure LiteLLM
litellm.drop_params = True  # Drop unsupported params for different models
litellm.set_verbose = False  # Set to True for debugging

class ModelProvider(str, Enum):
    """Supported model providers."""
    OPENAI = "openai"
    GEMINI = "gemini"
    CLAUDE = "claude"
    LOCAL = "local"
    
class ExtractionRequest(BaseModel):
    """Request model for paper extraction."""
    text: str = Field(..., description="Paper text to extract information from")
    model: str = Field(default="gemini/gemini-pro", description="Model to use for extraction")
    extraction_fields: Optional[List[str]] = Field(
        default=None,
        description="Specific fields to extract"
    )
    temperature: float = Field(default=0.1, ge=0, le=2, description="Model temperature")
    max_tokens: Optional[int] = Field(default=4000, description="Maximum tokens in response")

class ExtractionResponse(BaseModel):
    """Response model for extraction results."""
    success: bool
    extracted_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    model_used: str
    tokens_used: Optional[int] = None

# Initialize FastAPI app
app = FastAPI(
    title="Literature Review LLM Service",
    description="Model-agnostic LLM service for paper extraction",
    version="1.0.0"
)

# Available models configuration
AVAILABLE_MODELS = {
    "gemini/gemini-pro": {
        "provider": ModelProvider.GEMINI,
        "env_key": "GEMINI_API_KEY",
        "max_tokens": 32768,
        "description": "Google Gemini Pro - Good for general extraction"
    },
    "gemini/gemini-1.5-flash": {
        "provider": ModelProvider.GEMINI,
        "env_key": "GEMINI_API_KEY",
        "max_tokens": 1048576,
        "description": "Google Gemini 1.5 Flash - Fast and efficient"
    },
    "gpt-3.5-turbo": {
        "provider": ModelProvider.OPENAI,
        "env_key": "OPENAI_API_KEY",
        "max_tokens": 4096,
        "description": "OpenAI GPT-3.5 Turbo - Fast and cost-effective"
    },
    "gpt-4": {
        "provider": ModelProvider.OPENAI,
        "env_key": "OPENAI_API_KEY",
        "max_tokens": 8192,
        "description": "OpenAI GPT-4 - High quality extraction"
    },
    "claude-3-haiku-20240307": {
        "provider": ModelProvider.CLAUDE,
        "env_key": "ANTHROPIC_API_KEY",
        "max_tokens": 200000,
        "description": "Claude 3 Haiku - Fast and efficient"
    },
    "claude-3-sonnet-20240229": {
        "provider": ModelProvider.CLAUDE,
        "env_key": "ANTHROPIC_API_KEY",
        "max_tokens": 200000,
        "description": "Claude 3 Sonnet - Balanced performance"
    }
}

def get_extraction_prompt(text: str, fields: Optional[List[str]] = None) -> str:
    """Generate extraction prompt based on requested fields."""
    
    base_fields = [
        "research_questions",
        "key_contributions",
        "simulation_approach",
        "llm_usage",
        "human_llm_comparison",
        "evaluation_metrics",
        "prompting_strategies",
        "emerging_behaviors",
        "datasets_used",
        "limitations"
    ]
    
    extraction_fields = fields if fields else base_fields
    
    prompt = f"""You are a research assistant analyzing academic papers about LLM-powered wargames and simulations.

Extract the following information from the paper text provided. Return your response as a valid JSON object with these fields:

{json.dumps(extraction_fields, indent=2)}

For each field:
- Extract relevant information if present
- Use null if the information is not found
- Keep responses concise but informative
- Focus on factual information from the paper

Paper text:
{text[:50000]}  # Limit text length to avoid token limits

Return only the JSON object, no additional text."""
    
    return prompt

@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Literature Review LLM Service",
        "version": "1.0.0",
        "available_models": list(AVAILABLE_MODELS.keys()),
        "endpoints": ["/extract", "/models", "/health"]
    }

@app.get("/models")
async def list_models():
    """List available models and their status."""
    models_status = {}
    
    for model_name, config in AVAILABLE_MODELS.items():
        env_key = config["env_key"]
        api_key_present = bool(os.getenv(env_key))
        
        models_status[model_name] = {
            "available": api_key_present,
            "provider": config["provider"],
            "max_tokens": config["max_tokens"],
            "description": config["description"],
            "api_key_configured": api_key_present
        }
    
    return models_status

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "llm-extraction"}

@app.post("/extract", response_model=ExtractionResponse)
async def extract_paper_info(request: ExtractionRequest):
    """Extract information from paper text using specified model."""
    
    try:
        # Check if model is available
        if request.model not in AVAILABLE_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Model {request.model} not supported. Available models: {list(AVAILABLE_MODELS.keys())}"
            )
        
        # Check if API key is configured
        model_config = AVAILABLE_MODELS[request.model]
        if not os.getenv(model_config["env_key"]):
            raise HTTPException(
                status_code=401,
                detail=f"API key not configured for {request.model}. Set {model_config['env_key']} environment variable."
            )
        
        # Generate extraction prompt
        prompt = get_extraction_prompt(request.text, request.extraction_fields)
        
        # Call LiteLLM
        logger.info(f"Calling {request.model} for extraction")
        
        response = completion(
            model=request.model,
            messages=[
                {"role": "system", "content": "You are a helpful research assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            response_format={"type": "json_object"} if "gpt" in request.model else None
        )
        
        # Parse response
        content = response.choices[0].message.content
        
        # Try to parse as JSON
        try:
            extracted_data = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                extracted_data = json.loads(json_match.group())
            else:
                extracted_data = {"raw_response": content}
        
        # Get token usage if available
        tokens_used = None
        if hasattr(response, 'usage'):
            tokens_used = response.usage.total_tokens
        
        return ExtractionResponse(
            success=True,
            extracted_data=extracted_data,
            model_used=request.model,
            tokens_used=tokens_used
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Extraction error: {str(e)}")
        return ExtractionResponse(
            success=False,
            error=str(e),
            model_used=request.model
        )

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server."""
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run server
    run_server()