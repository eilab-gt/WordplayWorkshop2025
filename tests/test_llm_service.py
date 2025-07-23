"""Tests for the LLM service module."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os

from src.lit_review.llm_service import app, AVAILABLE_MODELS


class TestLLMService:
    """Test suite for LLM service endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns service info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Literature Review LLM Service"
        assert "available_models" in data
        assert "endpoints" in data
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy", "service": "llm-extraction"}
    
    def test_models_endpoint_no_keys(self, client):
        """Test models endpoint when no API keys are set."""
        with patch.dict(os.environ, {}, clear=True):
            response = client.get("/models")
            assert response.status_code == 200
            models = response.json()
            
            # Check all models show as unavailable
            for model_info in models.values():
                assert model_info["available"] is False
                assert model_info["api_key_configured"] is False
    
    def test_models_endpoint_with_keys(self, client):
        """Test models endpoint with API keys configured."""
        with patch.dict(os.environ, {
            "GEMINI_API_KEY": "test-key",
            "OPENAI_API_KEY": "test-key"
        }):
            response = client.get("/models")
            assert response.status_code == 200
            models = response.json()
            
            # Gemini models should be available
            assert models["gemini/gemini-pro"]["available"] is True
            assert models["gemini/gemini-1.5-flash"]["available"] is True
            
            # OpenAI models should be available
            assert models["gpt-3.5-turbo"]["available"] is True
            assert models["gpt-4"]["available"] is True
    
    def test_extract_invalid_model(self, client):
        """Test extraction with invalid model."""
        response = client.post("/extract", json={
            "text": "Test text",
            "model": "invalid-model"
        })
        assert response.status_code == 400
        assert "not supported" in response.json()["detail"]
    
    def test_extract_no_api_key(self, client):
        """Test extraction when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            response = client.post("/extract", json={
                "text": "Test text",
                "model": "gemini/gemini-pro"
            })
            assert response.status_code == 401
            assert "API key not configured" in response.json()["detail"]
    
    @patch('src.lit_review.llm_service.completion')
    def test_extract_success(self, mock_completion, client):
        """Test successful extraction."""
        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''
        {
            "research_questions": "What are the research questions?",
            "key_contributions": "Key contributions of the paper",
            "simulation_approach": "Simulation methodology",
            "llm_usage": "How LLMs are used",
            "human_llm_comparison": null,
            "evaluation_metrics": "Metrics used",
            "prompting_strategies": null,
            "emerging_behaviors": null,
            "datasets_used": "Dataset information",
            "limitations": "Study limitations"
        }
        '''
        mock_response.usage.total_tokens = 500
        mock_completion.return_value = mock_response
        
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            response = client.post("/extract", json={
                "text": "This is a test paper about LLM-powered wargaming.",
                "model": "gemini/gemini-pro",
                "temperature": 0.1
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["extracted_data"] is not None
            assert data["model_used"] == "gemini/gemini-pro"
            assert data["tokens_used"] == 500
    
    @patch('src.lit_review.llm_service.completion')
    def test_extract_json_parse_error(self, mock_completion, client):
        """Test extraction when JSON parsing fails."""
        # Mock invalid JSON response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is not valid JSON"
        mock_completion.return_value = mock_response
        
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            response = client.post("/extract", json={
                "text": "Test text",
                "model": "gemini/gemini-pro"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "raw_response" in data["extracted_data"]
    
    @patch('src.lit_review.llm_service.completion')
    def test_extract_with_custom_fields(self, mock_completion, client):
        """Test extraction with custom fields."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"custom_field": "value"}'
        mock_completion.return_value = mock_response
        
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            response = client.post("/extract", json={
                "text": "Test text",
                "model": "gemini/gemini-pro",
                "extraction_fields": ["custom_field"]
            })
            
            assert response.status_code == 200
            # Verify custom fields were passed to prompt
            mock_completion.assert_called_once()
            call_args = mock_completion.call_args[1]
            assert "custom_field" in call_args["messages"][1]["content"]
    
    @patch('src.lit_review.llm_service.completion')
    def test_extract_api_error(self, mock_completion, client):
        """Test extraction when API call fails."""
        mock_completion.side_effect = Exception("API Error")
        
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            response = client.post("/extract", json={
                "text": "Test text",
                "model": "gemini/gemini-pro"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False
            assert data["error"] == "API Error"
    
    def test_extract_request_validation(self, client):
        """Test request validation."""
        # Missing required field
        response = client.post("/extract", json={
            "model": "gemini/gemini-pro"
        })
        assert response.status_code == 422
        
        # Invalid temperature
        response = client.post("/extract", json={
            "text": "Test",
            "temperature": 3.0  # Out of range
        })
        assert response.status_code == 422
    
    def test_get_extraction_prompt(self):
        """Test prompt generation."""
        from src.lit_review.llm_service import get_extraction_prompt
        
        # Test with default fields
        prompt = get_extraction_prompt("Test paper text")
        assert "research_questions" in prompt
        assert "key_contributions" in prompt
        assert "Test paper text" in prompt
        
        # Test with custom fields
        prompt = get_extraction_prompt("Test text", ["field1", "field2"])
        assert "field1" in prompt
        assert "field2" in prompt
        assert "research_questions" not in prompt


class TestModelConfiguration:
    """Test model configuration and availability."""
    
    def test_available_models_structure(self):
        """Test AVAILABLE_MODELS has correct structure."""
        assert len(AVAILABLE_MODELS) >= 6  # At least 6 models
        
        for model_id, config in AVAILABLE_MODELS.items():
            assert "provider" in config
            assert "env_key" in config
            assert "max_tokens" in config
            assert "description" in config
            
            # Check model ID format
            if config["provider"] == "gemini":
                assert model_id.startswith("gemini/")
            elif config["provider"] == "openai":
                assert model_id in ["gpt-3.5-turbo", "gpt-4"]
    
    def test_model_providers(self):
        """Test all providers are represented."""
        providers = {config["provider"] for config in AVAILABLE_MODELS.values()}
        assert "gemini" in providers
        assert "openai" in providers
        assert "claude" in providers


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_extract_empty_text(self, client):
        """Test extraction with empty text."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            response = client.post("/extract", json={
                "text": "",
                "model": "gemini/gemini-pro"
            })
            assert response.status_code == 200
    
    def test_extract_very_long_text(self, client):
        """Test extraction with very long text."""
        long_text = "x" * 100000  # 100k characters
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            response = client.post("/extract", json={
                "text": long_text,
                "model": "gemini/gemini-pro"
            })
            assert response.status_code == 200
            # Should truncate in prompt
    
    @patch('src.lit_review.llm_service.completion')
    def test_extract_with_json_embedded_in_text(self, mock_completion, client):
        """Test extraction when JSON is embedded in other text."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''
        Here is the extraction:
        {"field": "value"}
        That's all!
        '''
        mock_completion.return_value = mock_response
        
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            response = client.post("/extract", json={
                "text": "Test",
                "model": "gemini/gemini-pro"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["extracted_data"]["field"] == "value"