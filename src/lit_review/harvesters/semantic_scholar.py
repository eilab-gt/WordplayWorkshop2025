"""Semantic Scholar harvester implementation."""

import logging
import time
from typing import Any
from urllib.parse import quote

import requests

from .base import BaseHarvester, Paper

logger = logging.getLogger(__name__)


class SemanticScholarHarvester(BaseHarvester):
    """Harvester for Semantic Scholar API."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, config):
        """Initialize Semantic Scholar harvester.

        Args:
            config: Configuration object
        """
        super().__init__(config)
        self.api_key = config.semantic_scholar_key
        self.rate_limits = config.rate_limits.get("semantic_scholar", {})
        self.delay_milliseconds = self.rate_limits.get("delay_milliseconds", 100)

        # Set up headers
        self.headers = {}
        if self.api_key:
            self.headers["x-api-key"] = self.api_key

    def search(self, query: str, max_results: int = 100) -> list[Paper]:
        """Search Semantic Scholar for papers.

        Args:
            query: Search query string
            max_results: Maximum number of results

        Returns:
            List of Paper objects
        """
        papers = []

        try:
            logger.info(f"Semantic Scholar: Starting search with query: {query}")

            # Semantic Scholar limits to 100 results per request
            batch_size = min(100, max_results)
            offset = 0

            while len(papers) < max_results:
                # Execute search
                results = self._search_batch(query, batch_size, offset)

                if not results:
                    break

                # Extract papers
                for result in results:
                    paper = self._extract_paper(result)
                    if paper:
                        papers.append(paper)

                        if len(papers) >= max_results:
                            break

                # Check if more results available
                if len(results) < batch_size:
                    break

                offset += batch_size

                # Rate limiting
                time.sleep(self.delay_milliseconds / 1000.0)

            logger.info(f"Semantic Scholar: Found {len(papers)} papers")

        except Exception as e:
            logger.error(f"Semantic Scholar: Error during search: {e}")

        # Filter by year
        papers = self.filter_by_year(papers)

        return papers

    def _search_batch(
        self, query: str, limit: int, offset: int
    ) -> list[dict[str, Any]]:
        """Search for a batch of papers.

        Args:
            query: Search query
            limit: Number of results to retrieve
            offset: Starting offset

        Returns:
            List of result dictionaries
        """
        try:
            # Build search URL
            quote(query)
            url = f"{self.BASE_URL}/paper/search"

            # Parameters
            params = {
                "query": query,
                "limit": limit,
                "offset": offset,
                "fields": "paperId,title,abstract,authors,year,venue,citationCount,url,externalIds,publicationTypes",
            }

            # Add year filter
            start_year, end_year = self.config.search_years
            params["year"] = f"{start_year}-{end_year}"

            # Make request
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()

            data = response.json()
            return data.get("data", [])

        except requests.exceptions.RequestException as e:
            logger.error(f"Semantic Scholar: Request error: {e}")
            return []

    def _extract_paper(self, result: dict[str, Any]) -> Paper | None:
        """Extract Paper object from Semantic Scholar result.

        Args:
            result: Semantic Scholar result dictionary

        Returns:
            Paper object or None if extraction fails
        """
        try:
            # Extract required fields
            title = result.get("title", "")
            if not title:
                return None

            # Extract authors
            authors = []
            for author in result.get("authors", []):
                name = author.get("name", "")
                if name:
                    authors.append(name)

            # Extract year
            year = result.get("year", 0)

            # Get abstract
            abstract = result.get("abstract", "")

            # Create Paper object
            paper = Paper(
                title=self.clean_text(title),
                authors=authors,
                year=year,
                abstract=self.clean_text(abstract),
                source_db="semantic_scholar",
                url=result.get("url", ""),
                venue=result.get("venue", ""),
                citations=result.get("citationCount", 0),
            )

            # Extract identifiers
            external_ids = result.get("externalIds", {})
            paper.doi = external_ids.get("DOI")
            paper.arxiv_id = external_ids.get("ArXiv")

            # Try to get PDF URL if ArXiv
            if paper.arxiv_id:
                paper.pdf_url = f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf"

            # Extract keywords from publication types
            pub_types = result.get("publicationTypes", [])
            if pub_types:
                paper.keywords = pub_types

            return paper

        except Exception as e:
            logger.error(f"Semantic Scholar: Failed to extract paper: {e}")
            return None

    def get_paper_by_id(self, paper_id: str) -> Paper | None:
        """Get a specific paper by Semantic Scholar ID or DOI.

        Args:
            paper_id: Semantic Scholar paper ID or DOI

        Returns:
            Paper object or None if not found
        """
        try:
            # Try different ID formats
            if paper_id.startswith("10."):
                # It's a DOI
                url = f"{self.BASE_URL}/paper/DOI:{paper_id}"
            else:
                # Assume it's a Semantic Scholar ID
                url = f"{self.BASE_URL}/paper/{paper_id}"

            # Add fields
            params = {
                "fields": "paperId,title,abstract,authors,year,venue,citationCount,url,externalIds,publicationTypes"
            }

            # Make request
            response = requests.get(url, params=params, headers=self.headers)

            if response.status_code == 404:
                return None

            response.raise_for_status()
            result = response.json()

            return self._extract_paper(result)

        except Exception as e:
            logger.error(f"Semantic Scholar: Error fetching paper {paper_id}: {e}")
            return None

    def get_recommendations(self, paper_id: str, max_results: int = 20) -> list[Paper]:
        """Get paper recommendations based on a seed paper.

        Args:
            paper_id: Semantic Scholar paper ID
            max_results: Maximum number of recommendations

        Returns:
            List of recommended Paper objects
        """
        try:
            url = f"{self.BASE_URL}/recommendations/v1/papers/forpaper/{paper_id}"

            params = {
                "fields": "paperId,title,abstract,authors,year,venue,citationCount,url,externalIds",
                "limit": max_results,
            }

            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()

            data = response.json()
            papers = []

            for result in data.get("recommendedPapers", []):
                paper = self._extract_paper(result)
                if paper:
                    papers.append(paper)

            return papers

        except Exception as e:
            logger.error(f"Semantic Scholar: Error getting recommendations: {e}")
            return []
