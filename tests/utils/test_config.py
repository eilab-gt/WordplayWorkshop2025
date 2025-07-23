"""Tests for the Config module."""
import pytest
import yaml
import os
from pathlib import Path
from unittest.mock import patch, mock_open

from src.lit_review.utils import Config, ConfigLoader, load_config


class TestConfig:
    """Test cases for Config class."""
    
    def test_config_init(self, sample_config):
        """Test Config object initialization."""
        config = Config(str(sample_config))
        
        assert config._config is not None
        assert 'search' in config._config
        assert 'api_keys' in config._config
        assert 'paths' in config._config
    
    def test_config_get(self, sample_config):
        """Test getting configuration values."""
        config = Config(str(sample_config))
        
        # Test simple get
        assert config.get('search.queries.preset1') is not None
        
        # Test nested get
        model = config.get('extraction.model')
        assert model == 'gpt-4'
        
        # Test with default
        missing = config.get('nonexistent.key', default='default_value')
        assert missing == 'default_value'
    
    def test_config_get_all(self, sample_config):
        """Test getting entire configuration."""
        config = Config(str(sample_config))
        
        all_config = config.get_all()
        assert isinstance(all_config, dict)
        assert 'search' in all_config
        assert 'api_keys' in all_config
    
    def test_config_reload(self, sample_config):
        """Test configuration reload."""
        config = Config(str(sample_config))
        
        # Modify the file
        with open(sample_config, 'r') as f:
            data = yaml.safe_load(f)
        
        data['test_key'] = 'test_value'
        
        with open(sample_config, 'w') as f:
            yaml.dump(data, f)
        
        # Reload
        config.reload()
        
        # Check new value is loaded
        assert config.get('test_key') == 'test_value'


class TestConfigLoader:
    """Test cases for ConfigLoader class."""
    
    def test_load_yaml(self, sample_config):
        """Test loading YAML configuration."""
        loader = ConfigLoader()
        config_dict = loader.load_yaml(str(sample_config))
        
        assert isinstance(config_dict, dict)
        assert 'search' in config_dict
        assert 'api_keys' in config_dict
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file."""
        loader = ConfigLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_yaml('nonexistent.yaml')
    
    def test_load_invalid_yaml(self, temp_dir):
        """Test loading invalid YAML."""
        loader = ConfigLoader()
        
        # Create invalid YAML file
        invalid_yaml = Path(temp_dir) / 'invalid.yaml'
        invalid_yaml.write_text('invalid: yaml: content: [')
        
        with pytest.raises(yaml.YAMLError):
            loader.load_yaml(str(invalid_yaml))
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'env_test_key'})
    def test_env_var_substitution(self, temp_dir):
        """Test environment variable substitution."""
        loader = ConfigLoader()
        
        # Create config with env var
        config_text = """
        api_keys:
          openai: ${OPENAI_API_KEY}
          other: regular_value
        """
        
        config_file = Path(temp_dir) / 'env_test.yaml'
        config_file.write_text(config_text)
        
        config_dict = loader.load_yaml(str(config_file))
        
        assert config_dict['api_keys']['openai'] == 'env_test_key'
        assert config_dict['api_keys']['other'] == 'regular_value'
    
    @patch.dict(os.environ, {})
    def test_missing_env_var(self, temp_dir):
        """Test handling of missing environment variables."""
        loader = ConfigLoader()
        
        # Create config with missing env var
        config_text = """
        api_keys:
          openai: ${MISSING_VAR}
        """
        
        config_file = Path(temp_dir) / 'missing_env.yaml'
        config_file.write_text(config_text)
        
        config_dict = loader.load_yaml(str(config_file))
        
        # Should keep the placeholder or use empty string
        assert config_dict['api_keys']['openai'] in ['${MISSING_VAR}', '']
    
    def test_validate_config(self, sample_config):
        """Test configuration validation."""
        loader = ConfigLoader()
        config_dict = loader.load_yaml(str(sample_config))
        
        # Should not raise exception for valid config
        loader.validate_config(config_dict)
    
    def test_validate_missing_required(self):
        """Test validation with missing required fields."""
        loader = ConfigLoader()
        
        # Config missing required fields
        invalid_config = {
            'search': {}  # Missing other required sections
        }
        
        with pytest.raises(ValueError):
            loader.validate_config(invalid_config)
    
    def test_validate_invalid_api_keys(self):
        """Test validation with invalid API key structure."""
        loader = ConfigLoader()
        
        invalid_config = {
            'search': {'queries': {}, 'sources': {}},
            'api_keys': 'not_a_dict',  # Should be dict
            'paths': {},
            'extraction': {},
            'failure_vocabularies': {},
            'viz': {},
            'export': {}
        }
        
        with pytest.raises(ValueError):
            loader.validate_config(invalid_config)


class TestLoadConfigFunction:
    """Test cases for load_config convenience function."""
    
    def test_load_config_function(self, sample_config):
        """Test the load_config convenience function."""
        config = load_config(str(sample_config))
        
        assert isinstance(config, Config)
        assert config.get('search.queries.preset1') is not None
    
    def test_load_config_with_env_vars(self, temp_dir):
        """Test load_config with environment variables."""
        with patch.dict(os.environ, {'TEST_VAR': 'test_value'}):
            # Create config with env var
            config_text = """
            test:
              value: ${TEST_VAR}
            search:
              queries: {}
              sources: {}
            api_keys: {}
            paths: {}
            extraction: {}
            failure_vocabularies: {}
            viz: {}
            export: {}
            """
            
            config_file = Path(temp_dir) / 'test_config.yaml'
            config_file.write_text(config_text)
            
            config = load_config(str(config_file))
            
            assert config.get('test.value') == 'test_value'
    
    def test_config_path_expansion(self, temp_dir):
        """Test path expansion in configuration."""
        # Create config with paths that need expansion
        config_text = f"""
        paths:
          data_dir: ~/data
          pdf_cache: {temp_dir}/pdfs
          output_dir: ./outputs
        search:
          queries: {{}}
          sources: {{}}
        api_keys: {{}}
        extraction: {{}}
        failure_vocabularies: {{}}
        viz: {{}}
        export: {{}}
        """
        
        config_file = Path(temp_dir) / 'path_test.yaml'
        config_file.write_text(config_text)
        
        config = load_config(str(config_file))
        
        # Paths should be expanded
        data_dir = config.get('paths.data_dir')
        assert '~' not in data_dir  # Should be expanded
        assert Path(data_dir).is_absolute() or data_dir.startswith(os.path.expanduser('~'))