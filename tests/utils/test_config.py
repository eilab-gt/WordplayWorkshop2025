"""Tests for the Config module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.lit_review.utils import Config, ConfigLoader, load_config


class TestConfig:
    """Test cases for Config class."""

    def test_config_init(self, sample_config):
        """Test Config object initialization."""
        # sample_config is already a Config object from our fixture
        config = sample_config

        assert config is not None
        assert hasattr(config, "search_years")
        assert hasattr(config, "openai_key")
        assert hasattr(config, "data_dir")

    def test_config_get(self, sample_config):
        """Test getting configuration values."""
        config = sample_config

        # Test attributes exist
        assert config.openai_key == "test-key"
        assert config.semantic_scholar_key == "test-key"
        
        # Test paths
        assert config.data_dir.exists()
        assert config.cache_dir.exists()
        
        # Test search parameters
        assert isinstance(config.search_years, tuple)
        assert len(config.search_years) == 2

    def test_config_get_all(self, sample_config):
        """Test getting entire configuration."""
        config = sample_config

        # Config is a dataclass, check its attributes
        assert hasattr(config, "__dict__")
        config_dict = vars(config)
        assert isinstance(config_dict, dict)
        assert "openai_key" in config_dict
        assert "data_dir" in config_dict

    def test_config_reload(self, sample_config_path, temp_dir):
        """Test configuration reload functionality."""
        # First load the config
        loader1 = ConfigLoader(str(sample_config_path))
        config1 = loader1.load()
        initial_value = config1.llm_min_params
        
        # Modify the config file
        with open(sample_config_path) as f:
            data = yaml.safe_load(f)
        
        data["search"]["llm_min_params"] = 200_000_000
        
        with open(sample_config_path, "w") as f:
            yaml.dump(data, f)
        
        # Load again (simulating reload)
        loader2 = ConfigLoader(str(sample_config_path))
        config2 = loader2.load()
        
        # Check the value changed
        assert config2.llm_min_params == 200_000_000
        assert config2.llm_min_params != initial_value


class TestConfigLoader:
    """Test cases for ConfigLoader class."""

    def test_load_yaml(self, sample_config_path):
        """Test loading YAML configuration."""
        loader = ConfigLoader(str(sample_config_path))
        config = loader.load()

        assert isinstance(config, Config)
        assert hasattr(config, "openai_key")
        assert config.openai_key == "test-key"

    def test_load_nonexistent_file(self):
        """Test loading non-existent file."""
        loader = ConfigLoader("nonexistent.yaml")

        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_load_invalid_yaml(self, temp_dir):
        """Test loading invalid YAML."""
        # Create invalid YAML file
        invalid_yaml = Path(temp_dir) / "invalid.yaml"
        invalid_yaml.write_text("invalid: yaml: content: [")
        
        loader = ConfigLoader(str(invalid_yaml))

        with pytest.raises(yaml.YAMLError):
            loader.load()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env_test_key"})
    def test_env_var_substitution(self, temp_dir):
        """Test environment variable substitution."""
        # Create config with env var
        config_text = f"""
        api_keys:
          openai: ${{OPENAI_API_KEY}}
          semantic_scholar: regular_value
        search:
          queries:
            preset1: test
        paths:
          data_dir: {temp_dir}/data
          pdf_cache: {temp_dir}/cache
          output_dir: {temp_dir}/output
        """

        config_file = Path(temp_dir) / "env_test.yaml"
        config_file.write_text(config_text)
        
        loader = ConfigLoader(str(config_file))
        config = loader.load()

        assert config.openai_key == "env_test_key"
        assert config.semantic_scholar_key == "regular_value"

    @patch.dict(os.environ, {})
    def test_missing_env_var(self, temp_dir):
        """Test handling of missing environment variables."""
        # Create config with missing env var
        config_text = f"""
        api_keys:
          openai: ${{MISSING_VAR}}
          semantic_scholar: test-key
        search:
          queries:
            preset1: test
        paths:
          data_dir: {temp_dir}/data
          pdf_cache: {temp_dir}/cache
          output_dir: {temp_dir}/output
        """

        config_file = Path(temp_dir) / "missing_env.yaml"
        config_file.write_text(config_text)
        
        loader = ConfigLoader(str(config_file))
        config = loader.load()

        # Should resolve to None for missing env var
        assert config.openai_key is None

    def test_validate_config(self, sample_config_path):
        """Test configuration validation."""
        loader = ConfigLoader(str(sample_config_path))
        
        # Should not raise exception for valid config
        config = loader.load()
        assert config is not None

    def test_validate_missing_required(self, temp_dir):
        """Test validation with missing required fields."""
        # Config missing required fields
        config_text = """
        search:
          queries: {}
        # Missing api_keys and paths sections
        """
        
        config_file = Path(temp_dir) / "invalid.yaml"
        config_file.write_text(config_text)
        
        loader = ConfigLoader(str(config_file))
        
        # The loader should handle missing fields with defaults
        config = loader.load()
        assert config is not None

    def test_validate_invalid_api_keys(self, temp_dir):
        """Test validation with invalid API key structure."""
        config_text = f"""
        search:
          queries: {{}}
        api_keys: "not_a_dict"  # Should be dict
        paths:
          data_dir: {temp_dir}/data
          pdf_cache: {temp_dir}/cache
          output_dir: {temp_dir}/output
        """
        
        config_file = Path(temp_dir) / "invalid_api_keys.yaml"
        config_file.write_text(config_text)
        
        loader = ConfigLoader(str(config_file))
        
        # Should raise an error when loading invalid structure
        with pytest.raises((ValueError, TypeError, AttributeError)):
            loader.load()


class TestLoadConfigFunction:
    """Test cases for load_config convenience function."""

    def test_load_config_function(self, sample_config_path):
        """Test the load_config convenience function."""
        config = load_config(str(sample_config_path))

        assert isinstance(config, Config)
        assert config.openai_key == "test-key"

    def test_load_config_with_env_vars(self, temp_dir):
        """Test load_config with environment variables."""
        with patch.dict(os.environ, {"TEST_API_KEY": "test_value"}):
            # Create config with env var
            config_text = f"""
            api_keys:
              openai: ${{TEST_API_KEY}}
              semantic_scholar: regular_key
            search:
              queries:
                preset1: test
            paths:
              data_dir: {temp_dir}/data
              pdf_cache: {temp_dir}/cache  
              output_dir: {temp_dir}/output
            """

            config_file = Path(temp_dir) / "test_config.yaml"
            config_file.write_text(config_text)

            config = load_config(str(config_file))

            assert config.openai_key == "test_value"

    def test_config_path_expansion(self, temp_dir):
        """Test path expansion in configuration."""
        # Create config with paths that need expansion
        config_text = f"""
        paths:
          data_dir: ~/data
          cache_dir: {temp_dir}/pdfs
          output_dir: ./outputs
        search:
          queries:
            preset1: test
        api_keys:
          openai: test-key
        """

        config_file = Path(temp_dir) / "path_test.yaml"
        config_file.write_text(config_text)

        config = load_config(str(config_file))

        # Paths should be Path objects (expansion happens at usage time)
        assert isinstance(config.data_dir, Path)
        assert isinstance(config.cache_dir, Path)
        assert str(config.data_dir) == "~/data"  # Path stored as-is, expanded when used
        assert str(config.cache_dir) == f"{temp_dir}/pdfs"
