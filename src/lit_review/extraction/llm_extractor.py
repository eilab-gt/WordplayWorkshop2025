"""LLM-based extraction of structured information from papers."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from pdfminer.high_level import extract_text
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

from ..llm_providers import LLMProvider

logger = logging.getLogger(__name__)


class LLMExtractor:
    """Extracts structured information from PDFs using LLMs."""

    def __init__(self, config):
        """Initialize LLM extractor.

        Args:
            config: Configuration object
        """
        self.config = config
        self.llm_provider = LLMProvider(config)
        self.model = config.llm_model
        self.temperature = config.llm_temperature
        self.max_tokens = config.llm_max_tokens

        # Load prompts
        self.extraction_prompt = config.extraction_prompt
        self.awscale_prompt = config.awscale_prompt

        # Track statistics
        self.stats = {
            "total_attempted": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "pdf_errors": 0,
            "llm_errors": 0,
            "awscale_assignments": 0,
        }

    def extract_all(self, df: pd.DataFrame, parallel: bool = True) -> pd.DataFrame:
        """Extract information from all papers with PDFs.

        Args:
            df: DataFrame with paper information and PDF paths
            parallel: Whether to process in parallel

        Returns:
            DataFrame with extracted information
        """
        # Filter to papers with PDFs
        pdf_papers = df[
            df["pdf_path"].notna()
            & (df["pdf_path"] != "")
            & (
                df["pdf_status"].str.startswith("downloaded")
                | (df["pdf_status"] == "cached")
            )
        ].copy()

        logger.info(f"Starting LLM extraction for {len(pdf_papers)} papers")
        self.stats["total_attempted"] = len(pdf_papers)

        if len(pdf_papers) == 0:
            logger.warning("No papers with PDFs to extract")
            return df

        # Initialize extraction columns
        extraction_cols = [
            "venue_type",
            "game_type",
            "open_ended",
            "quantitative",
            "llm_family",
            "llm_role",
            "eval_metrics",
            "failure_modes",
            "awscale",
            "code_release",
            "grey_lit_flag",
            "extraction_status",
            "extraction_confidence",
        ]

        for col in extraction_cols:
            if col not in df.columns:
                df[col] = ""

        # Process papers
        if parallel:
            results = self._extract_parallel(pdf_papers)
        else:
            results = self._extract_sequential(pdf_papers)

        # Update DataFrame with results
        for idx, result in results.items():
            for key, value in result.items():
                if key in df.columns:
                    df.at[idx, key] = value

        self._log_statistics()
        return df

    def _extract_sequential(self, df: pd.DataFrame) -> dict[Any, dict[str, Any]]:
        """Extract information sequentially.

        Args:
            df: DataFrame with papers to process

        Returns:
            Dictionary mapping index to extraction results
        """
        results = {}

        for idx, paper in df.iterrows():
            logger.info(f"Extracting paper {idx}: {paper['title'][:50]}...")
            result = self._extract_single_paper(paper)
            results[idx] = result

        return results

    def _extract_parallel(self, df: pd.DataFrame) -> dict[Any, dict[str, Any]]:
        """Extract information in parallel.

        Args:
            df: DataFrame with papers to process

        Returns:
            Dictionary mapping index to extraction results
        """
        results = {}
        batch_size = self.config.batch_size_llm
        max_workers = min(batch_size, self.config.parallel_workers)

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
        """Extract information from a single paper.

        Args:
            paper: Paper data as pandas Series

        Returns:
            Dictionary with extracted information
        """
        try:
            # Extract text from PDF
            pdf_path = Path(paper["pdf_path"])
            if not pdf_path.exists():
                logger.warning(f"PDF not found: {pdf_path}")
                self.stats["pdf_errors"] += 1
                return {"extraction_status": "pdf_not_found"}

            # Get PDF text and metadata
            text, metadata = self._extract_pdf_content(pdf_path)

            if not text or len(text) < 100:
                logger.warning(f"Insufficient text extracted from {pdf_path}")
                self.stats["pdf_errors"] += 1
                return {"extraction_status": "insufficient_text"}

            # Prepare context for LLM
            context = self._prepare_context(paper, text, metadata)

            # Extract structured information
            extracted = self._llm_extract(context)

            if extracted:
                # Assign AWScale
                awscale = self._assign_awscale(context, extracted)
                extracted["awscale"] = awscale

                # Add metadata
                extracted["extraction_status"] = "success"
                extracted["extraction_confidence"] = self._calculate_confidence(
                    extracted
                )

                self.stats["successful_extractions"] += 1
                return extracted
            else:
                self.stats["llm_errors"] += 1
                return {"extraction_status": "llm_extraction_failed"}

        except Exception as e:
            logger.error(f"Error extracting paper: {e}")
            self.stats["failed_extractions"] += 1
            return {"extraction_status": "error"}

    def _extract_pdf_content(self, pdf_path: Path) -> tuple[str, dict[str, Any]]:
        """Extract text and metadata from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Tuple of (text, metadata)
        """
        try:
            # Extract text
            text = extract_text(pdf_path, maxpages=50)  # Limit pages for efficiency

            # Extract metadata
            metadata = {}
            with open(pdf_path, "rb") as f:
                parser = PDFParser(f)
                doc = PDFDocument(parser)

                if doc.info:
                    info = doc.info[0] if doc.info else {}
                    metadata = {
                        "title": info.get("Title", b"").decode(
                            "utf-8", errors="ignore"
                        ),
                        "author": info.get("Author", b"").decode(
                            "utf-8", errors="ignore"
                        ),
                        "subject": info.get("Subject", b"").decode(
                            "utf-8", errors="ignore"
                        ),
                        "pages": sum(1 for _ in PDFPage.create_pages(doc)),
                    }

            return text, metadata

        except Exception as e:
            logger.error(f"Error extracting PDF content from {pdf_path}: {e}")
            return "", {}

    def _prepare_context(
        self, paper: pd.Series, text: str, metadata: dict[str, Any]
    ) -> str:
        """Prepare context for LLM extraction.

        Args:
            paper: Paper metadata
            text: Extracted PDF text
            metadata: PDF metadata

        Returns:
            Formatted context string
        """
        # Limit text length to avoid token limits
        max_chars = 30000  # Approximately 7-8k tokens
        if len(text) > max_chars:
            # Try to get introduction and conclusion
            intro_end = min(max_chars // 2, len(text))
            conclusion_start = max(len(text) - max_chars // 2, intro_end)
            text = (
                text[:intro_end]
                + "\n\n[... middle section omitted ...]\n\n"
                + text[conclusion_start:]
            )

        context = f"""
Paper Title: {paper.get("title", "Unknown")}
Authors: {paper.get("authors", "Unknown")}
Year: {paper.get("year", "Unknown")}
Venue: {paper.get("venue", "Unknown")}
Abstract: {paper.get("abstract", "Not available")}

PDF Pages: {metadata.get("pages", "Unknown")}

Full Text (truncated if needed):
{text}
"""

        return context

    def _llm_extract(self, context: str) -> Optional[dict[str, Any]]:
        """Extract structured information using LLM.

        Args:
            context: Paper context

        Returns:
            Extracted information or None
        """
        try:
            # Build messages
            messages = [
                {"role": "system", "content": self.extraction_prompt},
                {"role": "user", "content": context},
            ]

            # Use JSON response format if supported
            response_format = (
                {"type": "json_object"}
                if self.llm_provider._supports_json_mode()
                else None
            )

            response = self.llm_provider.chat_completion(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format=response_format,
            )

            # Parse response
            content = response.choices[0].message.content
            extracted = json.loads(content)

            # Validate required fields
            required_fields = [
                "venue_type",
                "game_type",
                "open_ended",
                "quantitative",
                "llm_family",
                "llm_role",
            ]

            for field in required_fields:
                if field not in extracted:
                    logger.warning(f"Missing required field: {field}")
                    extracted[field] = "unknown"

            # Process boolean fields
            for field in ["open_ended", "quantitative", "grey_lit_flag"]:
                if field in extracted:
                    value = str(extracted[field]).lower()
                    extracted[field] = "yes" if value in ["true", "yes", "1"] else "no"

            # Process list fields
            if "failure_modes" in extracted and isinstance(
                extracted["failure_modes"], list
            ):
                extracted["failure_modes"] = "|".join(extracted["failure_modes"])

            return extracted

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"LLM extraction error: {e}")
            return None

    def _assign_awscale(self, context: str, extracted: dict[str, Any]) -> int:
        """Assign AWScale rating using LLM.

        Args:
            context: Paper context
            extracted: Already extracted information

        Returns:
            AWScale rating (1-5)
        """
        try:
            # Prepare focused context for AWScale
            awscale_context = f"""
Game Type: {extracted.get("game_type", "unknown")}
Open-ended: {extracted.get("open_ended", "unknown")}
Quantitative: {extracted.get("quantitative", "unknown")}
Evaluation Metrics: {extracted.get("eval_metrics", "unknown")}

Relevant excerpt from paper:
{context[:5000]}  # First part of context
"""

            messages = [
                {"role": "system", "content": self.awscale_prompt},
                {"role": "user", "content": awscale_context},
            ]

            response = self.llm_provider.chat_completion(
                messages=messages,
                temperature=0.1,  # Low temperature for consistency
                max_tokens=10,
            )

            # Parse response
            content = response.choices[0].message.content.strip()

            # Try to extract number
            try:
                score = int(content[0])  # Get first character as number
                if 1 <= score <= 5:
                    self.stats["awscale_assignments"] += 1
                    return score
            except (ValueError, IndexError):
                pass

            # Fallback: Use heuristics
            return self._awscale_heuristic(extracted)

        except Exception as e:
            logger.error(f"AWScale assignment error: {e}")
            return self._awscale_heuristic(extracted)

    def _awscale_heuristic(self, extracted: dict[str, Any]) -> int:
        """Assign AWScale using heuristics.

        Args:
            extracted: Extracted information

        Returns:
            AWScale rating (1-5)
        """
        score = 3  # Default to balanced

        # Adjust based on game type
        game_type = extracted.get("game_type", "").lower()
        if "matrix" in game_type or "seminar" in game_type:
            score += 1
        elif "digital" in game_type:
            score -= 1

        # Adjust based on quantitative nature
        if extracted.get("quantitative") == "yes":
            score -= 1

        # Adjust based on open-ended nature
        if extracted.get("open_ended") == "yes":
            score += 1

        # Ensure score is in valid range
        return max(1, min(5, score))

    def _calculate_confidence(self, extracted: dict[str, Any]) -> float:
        """Calculate confidence score for extraction.

        Args:
            extracted: Extracted information

        Returns:
            Confidence score (0-1)
        """
        # Start with base confidence
        confidence = 0.5

        # Add points for non-empty required fields
        required_fields = [
            "venue_type",
            "game_type",
            "open_ended",
            "quantitative",
            "llm_family",
            "llm_role",
        ]

        for field in required_fields:
            if (
                field in extracted
                and extracted[field]
                and extracted[field] != "unknown"
            ):
                confidence += 0.08

        # Add points for optional fields
        optional_fields = ["eval_metrics", "failure_modes", "code_release"]
        for field in optional_fields:
            if extracted.get(field):
                confidence += 0.03

        # Cap at 1.0
        return min(1.0, confidence)

    def _log_statistics(self):
        """Log extraction statistics."""
        logger.info("LLM extraction statistics:")
        logger.info(f"  Total attempted: {self.stats['total_attempted']}")
        logger.info(f"  Successful extractions: {self.stats['successful_extractions']}")
        logger.info(f"  Failed extractions: {self.stats['failed_extractions']}")
        logger.info(f"  PDF errors: {self.stats['pdf_errors']}")
        logger.info(f"  LLM errors: {self.stats['llm_errors']}")
        logger.info(f"  AWScale assignments: {self.stats['awscale_assignments']}")

    def extract_single_pdf(
        self, pdf_path: str, paper_metadata: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Extract information from a single PDF file.

        Args:
            pdf_path: Path to PDF file
            paper_metadata: Optional paper metadata

        Returns:
            Extracted information
        """
        # Create a minimal paper Series
        paper = pd.Series(
            {
                "pdf_path": pdf_path,
                "title": (
                    paper_metadata.get("title", "Unknown")
                    if paper_metadata
                    else "Unknown"
                ),
                "authors": (
                    paper_metadata.get("authors", "Unknown")
                    if paper_metadata
                    else "Unknown"
                ),
                "year": paper_metadata.get("year", 0) if paper_metadata else 0,
                "abstract": (
                    paper_metadata.get("abstract", "") if paper_metadata else ""
                ),
            }
        )

        return self._extract_single_paper(paper)
