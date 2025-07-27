"""Google Scholar harvester implementation."""

import logging
import time

from requests.exceptions import RequestException
from scholarly import ProxyGenerator, scholarly

from .base import BaseHarvester, Paper

logger = logging.getLogger(__name__)


class GoogleScholarHarvester(BaseHarvester):
    """Harvester for Google Scholar using scholarly library."""

    def __init__(self, config):
        """Initialize Google Scholar harvester.

        Args:
            config: Configuration object
        """
        super().__init__(config)
        self.rate_limits = config.rate_limits.get("google_scholar", {})
        self.delay_seconds = self.rate_limits.get("delay_seconds", 5)
        self._setup_proxy()

    def _setup_proxy(self):
        """Set up proxy for Google Scholar if needed."""
        try:
            # Try to use free proxies to avoid rate limiting
            pg = ProxyGenerator()
            # Skip FreeProxies() as it's causing issues with the current scholarly version
            # Just initialize without proxy for now
            logger.info("Google Scholar: Running without proxy (proxy setup disabled)")
        except Exception as e:
            logger.warning(f"Google Scholar: Could not set up proxy: {e}")
            # Continue without proxy

    def search(self, query: str, max_results: int = 100) -> list[Paper]:
        """Search Google Scholar for papers.

        Args:
            query: Search query string
            max_results: Maximum number of results

        Returns:
            List of Paper objects
        """
        papers = []

        try:
            logger.info(f"Google Scholar: Starting search with query: {query}")

            # Execute search
            try:
                search_query = scholarly.search_pubs(query)
            except Exception as e:
                # Handle common scholarly errors
                if "Cannot Fetch" in str(e) or "CAPTCHA" in str(e):
                    logger.warning(
                        f"Google Scholar: Access blocked or rate limited: {e}"
                    )
                    logger.info(
                        "Google Scholar: Consider using a proxy or reducing request rate"
                    )
                else:
                    logger.error(f"Google Scholar: Search initialization failed: {e}")
                return papers

            # Collect results
            for i, result in enumerate(search_query):
                if i >= max_results:
                    break

                try:
                    # Extract paper information
                    paper = self._extract_paper(result)
                    if paper:
                        papers.append(paper)
                        logger.debug(
                            f"Google Scholar: Added paper {i + 1}: {paper.title[:50]}..."
                        )

                    # Rate limiting
                    time.sleep(self.delay_seconds)

                except Exception as e:
                    logger.error(f"Google Scholar: Error extracting paper {i + 1}: {e}")
                    continue

            logger.info(f"Google Scholar: Found {len(papers)} papers")

        except RequestException as e:
            logger.error(f"Google Scholar: Network error during search: {e}")
        except Exception as e:
            logger.error(f"Google Scholar: Unexpected error during search: {e}")

        # Filter by year
        papers = self.filter_by_year(papers)

        return papers

    def _extract_paper(self, result: dict) -> Optional[Paper]:
        """Extract Paper object from Google Scholar result.

        Args:
            result: Google Scholar result dictionary

        Returns:
            Paper object or None if extraction fails
        """
        try:
            # Get basic info
            info = result.get("bib", {})

            # Extract required fields
            title = info.get("title", "")
            if not title:
                return None

            # Extract authors
            authors = info.get("author", [])
            if isinstance(authors, str):
                # Sometimes it's a single string
                authors = [a.strip() for a in authors.split(" and ")]
            elif not isinstance(authors, list):
                authors = []

            # Extract year
            year = info.get("pub_year", 0)
            try:
                year = int(year) if year else 0
            except (ValueError, TypeError):
                year = 0

            # Get abstract
            abstract = info.get("abstract", "")
            if not abstract:
                # Try to get from description
                abstract = result.get("description", "")

            # Create Paper object
            paper = Paper(
                title=self.clean_text(title),
                authors=authors,
                year=year,
                abstract=self.clean_text(abstract),
                source_db="google_scholar",
                url=result.get("pub_url", result.get("eprint_url")),
                venue=info.get("venue", ""),
                citations=result.get("num_citations", 0),
            )

            # Try to extract DOI
            if abstract:
                paper.doi = self.extract_doi(abstract)

            # Check for arXiv ID
            eprint = result.get("eprint_url", "")
            if "arxiv.org" in eprint:
                # Extract arXiv ID from URL
                import re

                match = re.search(r"arxiv\.org/(?:Union[abs, pdf])/(\d+\.\d+)", eprint)
                if match:
                    paper.arxiv_id = match.group(1)
                    paper.pdf_url = f"https://arxiv.org/pdf/{match.group(1)}.pdf"

            return paper

        except Exception as e:
            logger.error(f"Google Scholar: Failed to extract paper: {e}")
            return None

    def search_advanced(
        self,
        title: Optional[str] = None,
        author: Optional[str] = None,
        pub_year_start: Optional[int] = None,
        pub_year_end: Optional[int] = None,
        max_results: int = 100,
    ) -> list[Paper]:
        """Advanced search with specific fields.

        Args:
            title: Title to search for
            author: Author name to search for
            pub_year_start: Start year for publication date
            pub_year_end: End year for publication date
            max_results: Maximum number of results

        Returns:
            List of Paper objects
        """
        # Build advanced query
        query_parts = []

        if title:
            query_parts.append(f'intitle:"{title}"')

        if author:
            query_parts.append(f'author:"{author}"')

        # Add base query
        base_query = self.build_query()
        query_parts.append(f"({base_query})")

        query = " ".join(query_parts)

        # Handle year filtering
        if pub_year_start or pub_year_end:
            # scholarly doesn't support year range in query,
            # so we'll filter after retrieval
            papers = self.search(
                query, max_results * 2
            )  # Get more to account for filtering

            # Additional year filtering
            if pub_year_start:
                papers = [p for p in papers if p.year >= pub_year_start]
            if pub_year_end:
                papers = [p for p in papers if p.year <= pub_year_end]

            return papers[:max_results]

        return self.search(query, max_results)
