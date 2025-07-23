"""Configuration management for the literature review pipeline."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Central configuration object for the pipeline."""

    # Search parameters
    search_years: tuple = (2018, 2025)
    llm_min_params: int = 100_000_000
    inclusion_flags: list = field(default_factory=list)
    wargame_terms: list = field(default_factory=list)
    llm_terms: list = field(default_factory=list)
    action_terms: list = field(default_factory=list)
    exclusion_terms: list = field(default_factory=list)

    # Failure vocabulary
    failure_vocab: dict[str, list] = field(default_factory=dict)

    # API keys
    semantic_scholar_key: str | None = None
    openai_key: str | None = None
    unpaywall_email: str | None = None

    # Rate limits
    rate_limits: dict[str, dict[str, int]] = field(default_factory=dict)

    # Paths
    cache_dir: Path = Path("./pdf_cache")
    output_dir: Path = Path("./output")
    data_dir: Path = Path("./data")
    log_dir: Path = Path("./logs")
    plugin_dir: Path = Path("./plugins")

    # Data file paths
    raw_papers_path: Path = Path("./data/raw/papers_raw.csv")
    screening_progress_path: Path = Path("./data/processed/screening_progress.csv")
    extraction_results_path: Path = Path("./data/extracted/extraction.csv")
    logging_db_path: Path = Path("./logs/logging.db")

    # LLM settings
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4000
    extraction_prompt: str = ""
    awscale_prompt: str = ""

    # Processing settings
    dedup_methods: list = field(default_factory=list)
    title_similarity_threshold: float = 0.9
    pdf_max_size_mb: int = 50
    pdf_timeout_seconds: int = 30

    # Batch sizes
    batch_size_harvesting: int = 50
    batch_size_pdf: int = 10
    batch_size_llm: int = 5

    # Visualization
    viz_format: str = "png"
    viz_dpi: int = 300
    viz_style: str = "seaborn-v0_8-darkgrid"
    viz_figsize: tuple = (10, 6)
    viz_colors: dict[str, dict[str, str]] = field(default_factory=dict)

    # Export settings
    export_compression: str = "zip"
    export_include_pdfs: bool = False
    export_include_logs: bool = True
    zenodo_enabled: bool = False
    zenodo_token: str | None = None

    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Development
    debug: bool = False
    dry_run: bool = False
    sample_size: int | None = None
    use_cache: bool = True
    parallel_workers: int = 4


class ConfigLoader:
    """Loads and manages configuration from YAML and environment variables."""

    def __init__(self, config_path: str = "config.yaml", env_path: str = ".env"):
        """Initialize the config loader.

        Args:
            config_path: Path to the YAML configuration file
            env_path: Path to the .env file
        """
        self.config_path = Path(config_path)
        self.env_path = Path(env_path)
        self._raw_config: dict[str, Any] = {}

    def load(self) -> Config:
        """Load configuration from files and environment.

        Returns:
            Populated Config object
        """
        # Load environment variables
        if self.env_path.exists():
            load_dotenv(self.env_path)
            logger.info(f"Loaded environment from {self.env_path}")

        # Load YAML configuration
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path) as f:
            self._raw_config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {self.config_path}")

        # Build Config object
        return self._build_config()

    def _build_config(self) -> Config:
        """Build Config object from raw configuration."""
        config = Config()

        # Search parameters
        search = self._raw_config.get("search", {})
        years = search.get("years", {})
        config.search_years = (years.get("start", 2018), years.get("end", 2025))
        config.llm_min_params = search.get("llm_min_params", 100_000_000)
        config.inclusion_flags = search.get("inclusion_flags", [])
        config.wargame_terms = search.get("wargame_terms", [])
        config.llm_terms = search.get("llm_terms", [])
        config.action_terms = search.get("action_terms", [])
        config.exclusion_terms = search.get("exclusion_terms", [])

        # Failure vocabulary
        config.failure_vocab = self._raw_config.get("failure_vocab", {})

        # API keys (with environment variable substitution)
        api_keys = self._raw_config.get("api_keys", {})
        config.semantic_scholar_key = self._resolve_env(
            api_keys.get("semantic_scholar")
        )
        config.openai_key = self._resolve_env(api_keys.get("openai"))
        config.unpaywall_email = self._resolve_env(api_keys.get("unpaywall_email"))

        # Rate limits
        config.rate_limits = self._raw_config.get("rate_limits", {})

        # Paths
        paths = self._raw_config.get("paths", {})
        config.cache_dir = Path(paths.get("cache_dir", "./pdf_cache"))
        config.output_dir = Path(paths.get("output_dir", "./output"))
        config.data_dir = Path(paths.get("data_dir", "./data"))
        config.log_dir = Path(paths.get("log_dir", "./logs"))
        config.plugin_dir = Path(paths.get("plugin_dir", "./plugins"))

        # Data file paths
        config.raw_papers_path = Path(
            paths.get("raw_papers", "./data/raw/papers_raw.csv")
        )
        config.screening_progress_path = Path(
            paths.get("screening_progress", "./data/processed/screening_progress.csv")
        )
        config.extraction_results_path = Path(
            paths.get("extraction_results", "./data/extracted/extraction.csv")
        )
        config.logging_db_path = Path(paths.get("logging_db", "./logs/logging.db"))

        # LLM settings
        llm = self._raw_config.get("llm", {})
        config.llm_model = llm.get("model", "gpt-4o")
        config.llm_temperature = llm.get("temperature", 0.1)
        config.llm_max_tokens = llm.get("max_tokens", 4000)
        config.extraction_prompt = llm.get("extraction_prompt", "")
        config.awscale_prompt = llm.get("awscale_prompt", "")

        # Processing settings
        processing = self._raw_config.get("processing", {})
        dedup = processing.get("dedup", {})
        config.dedup_methods = dedup.get("methods", ["doi_exact", "title_fuzzy"])
        config.title_similarity_threshold = dedup.get("title_similarity_threshold", 0.9)

        pdf = processing.get("pdf", {})
        config.pdf_max_size_mb = pdf.get("max_file_size_mb", 50)
        config.pdf_timeout_seconds = pdf.get("timeout_seconds", 30)

        batch = processing.get("batch_sizes", {})
        config.batch_size_harvesting = batch.get("harvesting", 50)
        config.batch_size_pdf = batch.get("pdf_download", 10)
        config.batch_size_llm = batch.get("llm_extraction", 5)

        # Visualization
        viz = self._raw_config.get("visualization", {})
        config.viz_format = viz.get("format", "png")
        config.viz_dpi = viz.get("dpi", 300)
        config.viz_style = viz.get("style", "seaborn-v0_8-darkgrid")
        config.viz_figsize = tuple(viz.get("figsize", [10, 6]))
        config.viz_colors = viz.get("colors", {})

        # Export settings
        export = self._raw_config.get("export", {})
        config.export_compression = export.get("compression", "zip")
        config.export_include_pdfs = export.get("include_pdfs", False)
        config.export_include_logs = export.get("include_logs", True)

        zenodo = export.get("zenodo", {})
        config.zenodo_enabled = zenodo.get("enabled", False)
        config.zenodo_token = self._resolve_env(zenodo.get("access_token"))

        # Logging
        logging_config = self._raw_config.get("logging", {})
        config.log_level = logging_config.get("level", "INFO")
        config.log_format = logging_config.get(
            "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Development
        dev = self._raw_config.get("development", {})
        config.debug = dev.get("debug", False)
        config.dry_run = dev.get("dry_run", False)
        config.sample_size = dev.get("sample_size")
        config.use_cache = dev.get("use_cache", True)
        config.parallel_workers = dev.get("parallel_workers", 4)

        # Create directories if they don't exist
        self._create_directories(config)

        return config

    def _resolve_env(self, value: str | None) -> str | None:
        """Resolve environment variable references in config values.

        Args:
            value: String that may contain ${VAR_NAME} references

        Returns:
            Resolved string or None
        """
        if not value:
            return None

        if value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.getenv(env_var)

        return value

    def _create_directories(self, config: Config) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            config.cache_dir,
            config.output_dir,
            config.data_dir,
            config.log_dir,
            config.plugin_dir,
            config.data_dir / "raw",
            config.data_dir / "processed",
            config.data_dir / "extracted",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Ensured directories exist: {directories}")


# Convenience function
def load_config(config_path: str = "config.yaml", env_path: str = ".env") -> Config:
    """Load configuration from files.

    Args:
        config_path: Path to YAML configuration
        env_path: Path to .env file

    Returns:
        Loaded Config object
    """
    loader = ConfigLoader(config_path, env_path)
    return loader.load()
