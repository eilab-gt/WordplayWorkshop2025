"""Enhanced LLM-based extraction that uses the LiteLLM service."""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text

from ..utils.content_cache import ContentCache

logger = logging.getLogger(__name__)


class EnhancedLLMExtractor:
    """Enhanced extractor that supports multiple content types and LLM providers."""

    def __init__(self, config, llm_service_url: str = "http://localhost:8000"):
        """Initialize enhanced LLM extractor.

        Args:
            config: Configuration object
            llm_service_url: URL of the LiteLLM service
        """
        self.config = config
        self.llm_service_url = llm_service_url

        # Initialize content cache
        self.content_cache = ContentCache(config)

        # Default model preferences
        self.model_preferences = [
            "gemini/gemini-pro",  # Primary choice
            "gpt-3.5-turbo",  # Fallback
            "claude-3-haiku-20240307",  # Alternative
        ]

        # Track statistics
        self.stats = {
            "total_attempted": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "pdf_processed": 0,
            "tex_processed": 0,
            "html_processed": 0,
            "llm_errors": 0,
            "awscale_assignments": 0,
        }

    def check_service_health(self) -> bool:
        """Check if LLM service is healthy."""
        try:
            response = requests.get(f"{self.llm_service_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"LLM service health check failed: {e}")
            return False

    def get_available_models(self) -> dict[str, Any]:
        """Get available models from the service."""
        try:
            response = requests.get(f"{self.llm_service_url}/models", timeout=5)
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return {}

    def extract_all(self, df: pd.DataFrame, parallel: bool = True) -> pd.DataFrame:
        """Extract information from all papers.

        Args:
            df: DataFrame with paper information
            parallel: Whether to process in parallel

        Returns:
            DataFrame with extracted information
        """
        # Check service health
        if not self.check_service_health():
            logger.error("LLM service is not healthy. Please start the service first.")
            return df

        # Get available models
        available_models = self.get_available_models()
        logger.info(f"Available models: {list(available_models.keys())}")

        # Filter to papers that need extraction
        papers_to_process = self._filter_papers_for_extraction(df)

        logger.info(f"Starting enhanced extraction for {len(papers_to_process)} papers")
        self.stats["total_attempted"] = len(papers_to_process)

        if len(papers_to_process) == 0:
            logger.warning("No papers to extract")
            return df

        # Initialize extraction columns
        extraction_cols = [
            "content_type",  # New: pdf, tex, html
            "research_questions",
            "key_contributions",
            "simulation_approach",
            "llm_usage",
            "human_llm_comparison",
            "evaluation_metrics",
            "prompting_strategies",
            "emerging_behaviors",
            "datasets_used",
            "limitations",
            "awscale",
            "extraction_status",
            "extraction_model",
            "extraction_confidence",
        ]

        for col in extraction_cols:
            if col not in df.columns:
                df[col] = ""

        # Process papers
        if parallel:
            results = self._extract_parallel(papers_to_process)
        else:
            results = self._extract_sequential(papers_to_process)

        # Update DataFrame with results
        for idx, result in results.items():
            for key, value in result.items():
                if key in df.columns:
                    df.at[idx, key] = value

        self._log_statistics()
        return df

    def _filter_papers_for_extraction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter papers that are ready for extraction."""
        # Papers with PDFs
        pdf_papers = df[
            df["pdf_path"].notna()
            & (df["pdf_path"] != "")
            & (
                df["pdf_status"].str.startswith("downloaded")
                | (df["pdf_status"] == "cached")
            )
        ]

        # Papers with arXiv IDs (can fetch TeX/HTML)
        arxiv_papers = df[
            df["arxiv_id"].notna()
            & (df["arxiv_id"] != "")
            & ~df.index.isin(pdf_papers.index)
        ]

        # Combine
        return pd.concat([pdf_papers, arxiv_papers]).drop_duplicates()

    def _extract_sequential(self, df: pd.DataFrame) -> dict[Any, dict[str, Any]]:
        """Extract information sequentially."""
        results = {}

        for idx, paper in df.iterrows():
            logger.info(f"Extracting paper {idx}: {paper['title'][:50]}...")
            result = self._extract_single_paper(paper)
            results[idx] = result

        return results

    def _extract_parallel(self, df: pd.DataFrame) -> dict[Any, dict[str, Any]]:
        """Extract information in parallel."""
        results = {}
        max_workers = min(
            5,
            (
                self.config.parallel_workers
                if hasattr(self.config, "parallel_workers")
                else 3
            ),
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_idx = {
                executor.submit(self._extract_single_paper, row): idx
                for idx, row in df.iterrows()
            }

            # Collect results
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results[idx] = result
                except Exception as e:
                    logger.error(f"Error extracting paper {idx}: {e}")
                    results[idx] = {"extraction_status": "error"}

        return results

    def _extract_single_paper(self, paper: pd.Series) -> dict[str, Any]:
        """Extract information from a single paper."""
        try:
            # Try to get content in priority order
            content, content_type = self._get_paper_content(paper)

            if not content or len(content) < 100:
                logger.warning(
                    f"Insufficient content for paper: {paper.get('title', 'Unknown')[:50]}"
                )
                return {"extraction_status": "insufficient_content"}

            # Extract using LLM service
            extracted = self._llm_service_extract(content, paper)

            if extracted:
                # Add content type
                extracted["content_type"] = content_type

                # Update stats
                self.stats[f"{content_type}_processed"] += 1
                self.stats["successful_extractions"] += 1

                return extracted
            else:
                self.stats["llm_errors"] += 1
                return {"extraction_status": "llm_extraction_failed"}

        except Exception as e:
            logger.error(f"Error extracting paper: {e}")
            self.stats["failed_extractions"] += 1
            return {"extraction_status": "error"}

    def _get_paper_content(self, paper: pd.Series) -> tuple[str, str]:
        """Get paper content in priority order: TeX > HTML > PDF."""
        content = ""
        content_type = "none"

        # Try TeX first if arXiv paper
        if paper.get("arxiv_id"):
            content, success = self._extract_tex_content(paper["arxiv_id"])
            if success:
                content_type = "tex"
                return content, content_type

            # Try HTML as fallback
            content, success = self._extract_html_content(paper["arxiv_id"])
            if success:
                content_type = "html"
                return content, content_type

        # Fall back to PDF
        if paper.get("pdf_path"):
            pdf_path = Path(paper["pdf_path"])
            if pdf_path.exists():
                content = self._extract_pdf_text(pdf_path)
                if content:
                    content_type = "pdf"

        return content, content_type

    def _extract_tex_content(self, arxiv_id: str) -> tuple[str, bool]:
        """Extract content from arXiv TeX source using cache."""
        try:
            from ..harvesters.arxiv_harvester import ArxivHarvester

            # Use a minimal config for the harvester
            class MinimalConfig:
                def __init__(self):
                    self.rate_limits = {"arxiv": {"delay_milliseconds": 0}}

            harvester = ArxivHarvester(MinimalConfig())

            # Generate paper ID for caching
            paper_id = f"arxiv:{arxiv_id}"

            # Define fetch function for cache
            def fetch_func():
                tex_content = harvester.fetch_tex_source(arxiv_id)
                if tex_content:
                    # Clean TeX content before caching
                    cleaned = self._clean_tex_content(tex_content)
                    return cleaned
                return None

            # Get from cache or fetch
            cache_path, was_cached = self.content_cache.get_or_fetch(
                paper_id,
                "tex",
                fetch_func,
                source_url=f"https://arxiv.org/e-print/{arxiv_id}",
            )

            if cache_path and cache_path.exists():
                # Read content from cache
                tex_content = cache_path.read_text(encoding="utf-8")
                return tex_content, True

        except Exception as e:
            logger.error(f"Error extracting TeX for {arxiv_id}: {e}")

        return "", False

    def _extract_html_content(self, arxiv_id: str) -> tuple[str, bool]:
        """Extract content from arXiv HTML version using cache."""
        try:
            from ..harvesters.arxiv_harvester import ArxivHarvester

            class MinimalConfig:
                def __init__(self):
                    self.rate_limits = {"arxiv": {"delay_milliseconds": 0}}

            harvester = ArxivHarvester(MinimalConfig())

            # Generate paper ID for caching
            paper_id = f"arxiv:{arxiv_id}"

            # Define fetch function for cache
            def fetch_func():
                html_content = harvester.fetch_html_source(arxiv_id)
                if html_content:
                    # Parse HTML and extract text
                    soup = BeautifulSoup(html_content, "html.parser")
                    main_content = soup.find("main") or soup.find("article") or soup
                    text = main_content.get_text(separator="\n", strip=True)
                    return text
                return None

            # Get from cache or fetch
            cache_path, was_cached = self.content_cache.get_or_fetch(
                paper_id,
                "html",
                fetch_func,
                source_url=f"https://ar5iv.org/abs/{arxiv_id}",
            )

            if cache_path and cache_path.exists():
                # Read content from cache
                text_content = cache_path.read_text(encoding="utf-8")
                return text_content, True

        except Exception as e:
            logger.error(f"Error extracting HTML for {arxiv_id}: {e}")

        return "", False

    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from PDF."""
        try:
            text = extract_text(pdf_path, maxpages=50)
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF text from {pdf_path}: {e}")
            return ""

    def _clean_tex_content(self, tex_content: str) -> str:
        """Clean TeX content for better extraction."""
        # Remove comments
        tex_content = re.sub(r"%.*?\n", "\n", tex_content)

        # Remove common TeX commands but keep content
        patterns_to_simplify = [
            (r"\\textbf\{([^}]+)\}", r"\1"),  # Bold text
            (r"\\textit\{([^}]+)\}", r"\1"),  # Italic text
            (r"\\emph\{([^}]+)\}", r"\1"),  # Emphasized text
            (r"\\cite\{[^}]+\}", "[citation]"),  # Citations
            (r"\\ref\{[^}]+\}", "[ref]"),  # References
            (r"\\label\{[^}]+\}", ""),  # Labels
            (r"\\begin\{equation\}.*?\\end\{equation\}", "[equation]", re.DOTALL),
            (r"\\begin\{figure\}.*?\\end\{figure\}", "[figure]", re.DOTALL),
            (r"\\begin\{table\}.*?\\end\{table\}", "[table]", re.DOTALL),
        ]

        for pattern, replacement, *flags in patterns_to_simplify:
            tex_content = re.sub(
                pattern, replacement, tex_content, flags=flags[0] if flags else 0
            )

        # Extract section content
        sections = re.findall(
            r"\\section\{([^}]+)\}(.*?)(?=\\section|\\end\{document\}|$)",
            tex_content,
            re.DOTALL,
        )

        # Build cleaned content
        cleaned = []
        for section_title, section_content in sections:
            cleaned.append(f"\n## {section_title}\n")
            cleaned.append(section_content.strip())

        return "\n".join(cleaned)

    def _llm_service_extract(
        self, content: str, paper: pd.Series
    ) -> dict[str, Any] | None:
        """Extract information using the LLM service."""
        try:
            # Prepare metadata
            metadata = f"""
Paper Title: {paper.get("title", "Unknown")}
Authors: {paper.get("authors", "Unknown")}
Year: {paper.get("year", "Unknown")}
Venue: {paper.get("venue", "Unknown")}
Abstract: {paper.get("abstract", "Not available")}
"""

            # Combine metadata and content
            full_text = metadata + "\n\nFull Text:\n" + content[:40000]  # Limit length

            # Try models in preference order
            for model in self.model_preferences:
                try:
                    # Make extraction request
                    response = requests.post(
                        f"{self.llm_service_url}/extract",
                        json={
                            "text": full_text,
                            "model": model,
                            "temperature": 0.1,
                            "max_tokens": 4000,
                        },
                        timeout=60,
                    )

                    if response.status_code == 200:
                        result = response.json()
                        if result["success"] and result["extracted_data"]:
                            extracted = result["extracted_data"]

                            # Add metadata
                            extracted["extraction_status"] = "success"
                            extracted["extraction_model"] = model
                            extracted["extraction_confidence"] = (
                                self._calculate_confidence(extracted)
                            )

                            # Assign AWScale
                            extracted["awscale"] = self._assign_awscale(extracted)

                            return extracted

                    elif response.status_code == 401:
                        logger.warning(f"API key not configured for {model}")
                        continue

                except requests.exceptions.RequestException as e:
                    logger.error(f"Request error with {model}: {e}")
                    continue

            logger.error("All models failed for extraction")
            return None

        except Exception as e:
            logger.error(f"LLM service extraction error: {e}")
            return None

    def _assign_awscale(self, extracted: dict[str, Any]) -> int:
        """Assign AWScale rating based on extracted information."""
        score = 3  # Default

        # Adjust based on simulation approach
        sim_approach = str(extracted.get("simulation_approach", "")).lower()
        if any(term in sim_approach for term in ["matrix", "seminar", "tabletop"]):
            score += 1
        elif any(term in sim_approach for term in ["digital", "automated", "ai-only"]):
            score -= 1

        # Adjust based on human-LLM comparison
        human_llm = str(extracted.get("human_llm_comparison", "")).lower()
        if "human" in human_llm and "llm" in human_llm:
            score += 1

        return max(1, min(5, score))

    def _calculate_confidence(self, extracted: dict[str, Any]) -> float:
        """Calculate confidence score for extraction."""
        confidence = 0.5

        # Check for key fields
        key_fields = [
            "research_questions",
            "key_contributions",
            "simulation_approach",
            "llm_usage",
            "evaluation_metrics",
        ]

        for field in key_fields:
            if field in extracted and extracted[field] and extracted[field] != "null":
                confidence += 0.1

        return min(1.0, confidence)

    def _log_statistics(self):
        """Log extraction statistics."""
        logger.info("Enhanced LLM extraction statistics:")
        logger.info(f"  Total attempted: {self.stats['total_attempted']}")
        logger.info(f"  Successful extractions: {self.stats['successful_extractions']}")
        logger.info(f"  Failed extractions: {self.stats['failed_extractions']}")
        logger.info(f"  PDF processed: {self.stats['pdf_processed']}")
        logger.info(f"  TeX processed: {self.stats['tex_processed']}")
        logger.info(f"  HTML processed: {self.stats['html_processed']}")
        logger.info(f"  LLM errors: {self.stats['llm_errors']}")
        logger.info(f"  AWScale assignments: {self.stats['awscale_assignments']}")
