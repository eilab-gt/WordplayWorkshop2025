"""Helper utilities for tests."""

from pathlib import Path
from src.lit_review.utils.config import Config, ConfigLoader
import yaml


def create_test_config(temp_dir: str, **overrides) -> Config:
    """Create a test configuration with optional overrides.
    
    Args:
        temp_dir: Temporary directory for test files
        **overrides: Configuration values to override
        
    Returns:
        Config object for testing
    """
    # Default test configuration
    default_config = {
        "search": {
            "queries": {
                "preset1": '"LLM" AND ("wargaming" OR "wargame")',
            },
            "sources": {
                "arxiv": {"enabled": True, "max_results": 10},
            },
        },
        "api_keys": {
            "openai": "test-key",
            "semantic_scholar": "test-key",
        },
        "extraction": {
            "model": "gpt-4",
            "temperature": 0.3,
        },
        "failure_vocabularies": {
            "escalation": ["escalation", "nuclear"],
            "bias": ["bias", "unfair"],
        },
        "paths": {
            "data_dir": f"{temp_dir}/data",
            "pdf_cache": f"{temp_dir}/pdf_cache",
            "output_dir": f"{temp_dir}/output",
        },
        "processing": {
            "batch_sizes": {
                "harvesting": 10,
                "pdf_download": 5,
            }
        }
    }
    
    # Apply overrides
    _deep_update(default_config, overrides)
    
    # Save config file
    config_path = Path(temp_dir) / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(default_config, f)
    
    # Create directories
    Path(default_config["paths"]["data_dir"]).mkdir(parents=True, exist_ok=True)
    Path(default_config["paths"]["pdf_cache"]).mkdir(parents=True, exist_ok=True)
    Path(default_config["paths"]["output_dir"]).mkdir(parents=True, exist_ok=True)
    
    # Load and return Config object
    loader = ConfigLoader(str(config_path))
    return loader.load()


def _deep_update(base_dict: dict, update_dict: dict) -> dict:
    """Recursively update a dictionary."""
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict:
            base_dict[key] = _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict