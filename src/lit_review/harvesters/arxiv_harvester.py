"""arXiv harvester implementation."""

import logging
import time
import re
import requests
from typing import Optional

import arxiv

from .base import BaseHarvester, Paper

logger = logging.getLogger(__name__)


class ArxivHarvester(BaseHarvester):
    """Harvester for arXiv papers."""

    def __init__(self, config):
        """Initialize arXiv harvester.

        Args:
            config: Configuration object
        """
        super().__init__(config)
        self.rate_limits = config.rate_limits.get("arxiv", {})
        self.delay_milliseconds = self.rate_limits.get("delay_milliseconds", 333)

    def search(self, query: str, max_results: int = 100) -> list[Paper]:
        """Search arXiv for papers.

        Args:
            query: Search query string
            max_results: Maximum number of results

        Returns:
            List of Paper objects
        """
        papers = []

        try:
            logger.info(f"arXiv: Starting search with query: {query}")

            # Build arXiv query
            arxiv_query = self._build_arxiv_query(query)

            # Create search
            search = arxiv.Search(
                query=arxiv_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )

            # Execute search and collect results
            for i, result in enumerate(search.results()):
                try:
                    paper = self._extract_paper(result)
                    if paper:
                        papers.append(paper)
                        logger.debug(
                            f"arXiv: Added paper {i + 1}: {paper.title[:50]}..."
                        )

                    # Rate limiting
                    time.sleep(self.delay_milliseconds / 1000.0)

                except Exception as e:
                    logger.error(f"arXiv: Error extracting paper {i + 1}: {e}")
                    continue

            logger.info(f"arXiv: Found {len(papers)} papers")

        except Exception as e:
            logger.error(f"arXiv: Error during search: {e}")

        # Filter by year
        papers = self.filter_by_year(papers)

        return papers

    def _build_arxiv_query(self, base_query: str) -> str:
        """Convert general query to arXiv-specific query.

        Args:
            base_query: General search query

        Returns:
            arXiv-formatted query
        """
        # arXiv uses different query syntax
        # Convert quoted terms and boolean operators

        # For arXiv, we'll search in title and abstract
        # and focus on CS categories

        # Extract key terms from our config
        terms = []

        # Add wargame terms
        for term in self.config.wargame_terms:
            terms.append(f'(ti:"{term}" OR abs:"{term}")')

        # Add LLM terms
        llm_terms = []
        for term in self.config.llm_terms:
            llm_terms.append(f'(ti:"{term}" OR abs:"{term}")')

        # Combine with AND
        query_parts = []

        # At least one wargame term
        if terms:
            query_parts.append(f"({' OR '.join(terms)})")

        # At least one LLM term
        if llm_terms:
            query_parts.append(f"({' OR '.join(llm_terms)})")

        # Combine
        arxiv_query = " AND ".join(query_parts)

        # Add category filter for CS
        arxiv_query = f"({arxiv_query}) AND (cat:cs.*)"

        logger.debug(f"arXiv query: {arxiv_query}")

        return arxiv_query

    def _extract_paper(self, result: arxiv.Result) -> Paper | None:
        """Extract Paper object from arXiv result.

        Args:
            result: arXiv Result object

        Returns:
            Paper object or None if extraction fails
        """
        try:
            # Extract required fields
            title = result.title
            if not title:
                return None

            # Extract authors
            authors = [author.name for author in result.authors]

            # Extract year from published date
            year = result.published.year if result.published else 0

            # Get abstract
            abstract = result.summary

            # Extract arXiv ID
            arxiv_id = result.entry_id.split("/")[-1].replace("v", "")

            # Create Paper object
            paper = Paper(
                title=self.clean_text(title),
                authors=authors,
                year=year,
                abstract=self.clean_text(abstract),
                source_db="arxiv",
                url=result.entry_id,
                doi=result.doi,
                arxiv_id=arxiv_id,
                pdf_url=result.pdf_url,
                keywords=[cat.term if hasattr(cat, 'term') else str(cat) for cat in result.categories],
            )

            # Add journal reference if available
            if result.journal_ref:
                paper.venue = result.journal_ref

            return paper

        except Exception as e:
            logger.error(f"arXiv: Failed to extract paper: {e}")
            return None

    def search_by_category(
        self, category: str = "cs.AI", max_results: int = 100
    ) -> list[Paper]:
        """Search within a specific arXiv category.

        Args:
            category: arXiv category (e.g., cs.AI, cs.CL)
            max_results: Maximum number of results

        Returns:
            List of Paper objects
        """
        # Build query with category
        base_query = self.build_query()
        arxiv_query = self._build_arxiv_query(base_query)

        # Add category filter
        category_query = f"{arxiv_query} AND cat:{category}"

        # Use the base search method
        return self.search(category_query, max_results)

    def get_paper_by_id(self, arxiv_id: str) -> Paper | None:
        """Get a specific paper by arXiv ID.

        Args:
            arxiv_id: arXiv ID (e.g., "2301.00234")

        Returns:
            Paper object or None if not found
        """
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            results = list(search.results())

            if results:
                return self._extract_paper(results[0])

            return None

        except Exception as e:
            logger.error(f"arXiv: Error fetching paper {arxiv_id}: {e}")
            return None

    def fetch_tex_source(self, arxiv_id: str) -> Optional[str]:
        """Fetch the TeX source for an arXiv paper.

        Args:
            arxiv_id: arXiv ID (e.g., "2301.00234")

        Returns:
            TeX source content or None if not available
        """
        try:
            # Clean the arxiv ID (remove version if present)
            clean_id = arxiv_id.split('v')[0]
            
            # arXiv source URL
            source_url = f"https://arxiv.org/e-print/{clean_id}"
            
            logger.info(f"Fetching TeX source for arXiv:{clean_id}")
            
            # Add headers to identify as a research tool
            headers = {
                'User-Agent': 'LiteratureReviewPipeline/1.0 (Research Tool)'
            }
            
            response = requests.get(source_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # arXiv returns tar.gz files for source
            # For now, return the raw content - we'll process it later
            content = response.content
            
            # Check if it's actually TeX (starts with common TeX commands)
            text_preview = content[:1000].decode('utf-8', errors='ignore')
            if any(cmd in text_preview for cmd in ['\\documentclass', '\\begin{document}', '\\section']):
                logger.info(f"Successfully fetched TeX source for {arxiv_id}")
                return content.decode('utf-8', errors='ignore')
            else:
                logger.warning(f"Source for {arxiv_id} doesn't appear to be TeX")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching TeX source for {arxiv_id}: {e}")
            return None

    def fetch_html_source(self, arxiv_id: str) -> Optional[str]:
        """Fetch the HTML version of an arXiv paper if available.

        Args:
            arxiv_id: arXiv ID (e.g., "2301.00234")

        Returns:
            HTML content or None if not available
        """
        try:
            # Clean the arxiv ID
            clean_id = arxiv_id.split('v')[0]
            
            # Some arXiv papers have HTML versions
            # Try the ar5iv service which converts arXiv papers to HTML
            html_url = f"https://ar5iv.org/abs/{clean_id}"
            
            logger.info(f"Fetching HTML version for arXiv:{clean_id}")
            
            headers = {
                'User-Agent': 'LiteratureReviewPipeline/1.0 (Research Tool)'
            }
            
            response = requests.get(html_url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                # Check if we got actual HTML content
                if 'text/html' in response.headers.get('content-type', ''):
                    logger.info(f"Successfully fetched HTML for {arxiv_id}")
                    return response.text
                else:
                    logger.warning(f"Response for {arxiv_id} is not HTML")
                    return None
            else:
                logger.warning(f"HTML not available for {arxiv_id} (status: {response.status_code})")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching HTML for {arxiv_id}: {e}")
            return None
