"""Main search harvester that combines multiple sources."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd

from .arxiv_harvester import ArxivHarvester
from .base import Paper
from .crossref import CrossrefHarvester
from .google_scholar import GoogleScholarHarvester
from .semantic_scholar import SemanticScholarHarvester

logger = logging.getLogger(__name__)


class SearchHarvester:
    """Main harvester that combines results from multiple sources."""

    def __init__(self, config):
        """Initialize the search harvester.

        Args:
            config: Configuration object
        """
        logger.info("SearchHarvester.__init__ called")
        self.config = config

        # Store harvester classes for lazy initialization
        self._harvester_classes = {
            "google_scholar": GoogleScholarHarvester,
            "arxiv": ArxivHarvester,
            "semantic_scholar": SemanticScholarHarvester,
            "crossref": CrossrefHarvester,
        }

        # Lazy initialization dict
        self._harvesters: dict[str, Any] = {}

        # Track all results
        self.all_papers: list[Paper] = []
        self.unique_papers: list[Paper] = []

    @property
    def harvesters(self) -> dict[str, Any]:
        """Get harvesters dict (for backward compatibility)."""
        return self

    def __getitem__(self, key: str) -> Any:
        """Get a harvester, initializing it lazily if needed."""
        if key not in self._harvesters and key in self._harvester_classes:
            logger.info(f"Lazily initializing {key} harvester")
            self._harvesters[key] = self._harvester_classes[key](self.config)
        return self._harvesters[key]

    def __contains__(self, key: str) -> bool:
        """Check if a harvester is available."""
        return key in self._harvester_classes

    def keys(self):
        """Get available harvester names."""
        return self._harvester_classes.keys()

    def search_all(
        self,
        sources: list[str] | None = None,
        max_results_per_source: int = 100,
        parallel: bool = True,
    ) -> pd.DataFrame:
        """Search all configured sources and combine results.

        Args:
            sources: List of sources to search (None = all sources)
            max_results_per_source: Maximum results from each source
            parallel: Whether to search sources in parallel

        Returns:
            DataFrame of combined results
        """
        # Determine which sources to use
        if sources is None:
            sources = list(self.harvesters.keys())
        else:
            # Validate sources
            sources = [s for s in sources if s in self.harvesters]

        logger.info(f"Starting search across sources: {sources}")

        # Clear previous results
        self.all_papers = []

        # Build query
        query = self._build_combined_query()

        if parallel and len(sources) > 1:
            # Parallel search
            self._search_parallel(sources, query, max_results_per_source)
        else:
            # Sequential search
            self._search_sequential(sources, query, max_results_per_source)

        # Convert to DataFrame
        df = self._papers_to_dataframe(self.all_papers)

        logger.info(f"Total papers collected: {len(self.all_papers)}")

        return df

    def _build_combined_query(self) -> str:
        """Build a combined query string from configuration.

        Returns:
            Query string
        """
        # Use ArxivHarvester's query builder as it doesn't require network setup
        # All harvesters inherit from BaseHarvester which has build_query()
        temp_harvester = ArxivHarvester(self.config)
        return temp_harvester.build_query()

    def _search_sequential(self, sources: list[str], query: str, max_results: int):
        """Search sources sequentially.

        Args:
            sources: List of source names
            query: Search query
            max_results: Maximum results per source
        """
        for source in sources:
            try:
                logger.info(f"Searching {source}...")
                harvester = self.harvesters[source]
                papers = harvester.search(query, max_results)
                self.all_papers.extend(papers)
                logger.info(f"{source}: Added {len(papers)} papers")
            except Exception as e:
                logger.error(f"Error searching {source}: {e}")

    def _search_parallel(self, sources: list[str], query: str, max_results: int):
        """Search sources in parallel.

        Args:
            sources: List of source names
            query: Search query
            max_results: Maximum results per source
        """
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            # Submit search tasks
            future_to_source = {
                executor.submit(
                    self.harvesters[source].search, query, max_results
                ): source
                for source in sources
            }

            # Collect results as they complete
            for future in as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    papers = future.result()
                    self.all_papers.extend(papers)
                    logger.info(f"{source}: Added {len(papers)} papers")
                except Exception as e:
                    logger.error(f"Error searching {source}: {e}")

    def _papers_to_dataframe(self, papers: list[Paper]) -> pd.DataFrame:
        """Convert list of Paper objects to DataFrame.

        Args:
            papers: List of Paper objects

        Returns:
            DataFrame with paper data
        """
        if not papers:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(
                columns=[
                    "title",
                    "authors",
                    "year",
                    "abstract",
                    "source_db",
                    "url",
                    "doi",
                    "arxiv_id",
                    "venue",
                    "citations",
                    "pdf_url",
                    "keywords",
                ]
            )

        # Convert to dictionaries
        data = [paper.to_dict() for paper in papers]

        # Create DataFrame
        df = pd.DataFrame(data)

        # Sort by year (descending) and citations (descending)
        df = df.sort_values(["year", "citations"], ascending=[False, False])

        # Reset index
        df = df.reset_index(drop=True)

        return df

    def deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate papers based on DOI and title similarity.

        Args:
            df: DataFrame of papers

        Returns:
            Deduplicated DataFrame
        """
        logger.info(f"Starting deduplication of {len(df)} papers")

        # This will be implemented in the Normalizer module
        # For now, just remove exact duplicates

        # First, deduplicate by DOI (if available)
        df_with_doi = df[df["doi"].notna() & (df["doi"] != "")]
        df_without_doi = df[df["doi"].isna() | (df["doi"] == "")]

        # Remove duplicate DOIs (keep first occurrence)
        df_with_doi = df_with_doi.drop_duplicates(subset=["doi"], keep="first")

        # Combine back
        df_dedup = pd.concat([df_with_doi, df_without_doi], ignore_index=True)

        logger.info(f"After deduplication: {len(df_dedup)} papers")

        return df_dedup

    def save_results(self, df: pd.DataFrame, output_path: str | None = None):
        """Save search results to CSV file.

        Args:
            df: DataFrame of results
            output_path: Path to save CSV (uses config default if None)
        """
        if output_path is None:
            output_path = self.config.raw_papers_path

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} papers to {output_path}")

    def get_source_statistics(self, df: pd.DataFrame) -> dict[str, int]:
        """Get statistics about papers from each source.

        Args:
            df: DataFrame of papers

        Returns:
            Dictionary of source counts
        """
        return df["source_db"].value_counts().to_dict()

    def search_seed_papers(self, seed_papers: list[dict[str, Any]]) -> pd.DataFrame:
        """Search for specific seed papers to verify coverage.

        Args:
            seed_papers: List of seed paper information

        Returns:
            DataFrame of found seed papers
        """
        found_papers = []

        for seed in seed_papers:
            # Try to find by title
            title = seed.get("title", "")
            author = seed.get("author", "")
            year = seed.get("year")

            logger.info(f"Searching for seed paper: {title[:50]}...")

            # Search with specific title
            if title:
                # Use Google Scholar for targeted search
                papers = self.harvesters["google_scholar"].search_advanced(
                    title=title,
                    author=author,
                    pub_year_start=year,
                    pub_year_end=year,
                    max_results=5,
                )

                if papers:
                    found_papers.extend(papers)
                    logger.info(f"Found seed paper: {papers[0].title}")
                else:
                    logger.warning(f"Could not find seed paper: {title}")

        return self._papers_to_dataframe(found_papers)
