"""arXiv harvester implementation."""

import logging
import time
from datetime import datetime, timedelta

import arxiv
import requests

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
        """Search arXiv for papers with pagination support.

        Args:
            query: Search query string
            max_results: Maximum number of results

        Returns:
            List of Paper objects
        """
        papers = []

        try:
            logger.info(
                f"arXiv: Starting search with query: {query}, max_results: {max_results}"
            )

            # Build arXiv query
            arxiv_query = self._build_arxiv_query(query)

            # Use date-range splitting to overcome API limitations
            papers = self._search_with_date_splitting(arxiv_query, max_results)

        except Exception as e:
            logger.error(f"arXiv: Error during search: {e}")

        # Filter by year
        papers = self.filter_by_year(papers)

        return papers

    def _search_with_date_splitting(
        self, arxiv_query: str, max_results: int
    ) -> list[Paper]:
        """Split query by date ranges to overcome arXiv API pagination limits.

        Args:
            arxiv_query: Formatted arXiv query
            max_results: Maximum total results to collect

        Returns:
            List of Paper objects
        """
        papers = []

        # Generate monthly date ranges from 2018-01 to 2025-12
        date_ranges = self._generate_monthly_ranges("2018-01", "2025-12")

        logger.info(
            f"arXiv: Using date-range splitting strategy with {len(date_ranges)} monthly chunks"
        )

        for i, (start_date, end_date) in enumerate(date_ranges):
            if len(papers) >= max_results:
                break

            logger.info(
                f"arXiv: Processing date range {i+1}/{len(date_ranges)}: {start_date} to {end_date}"
            )

            # Add date filter to existing query
            date_query = (
                f"({arxiv_query}) AND submittedDate:[{start_date} TO {end_date}]"
            )

            try:
                search = arxiv.Search(
                    query=date_query,
                    max_results=min(100, max_results - len(papers)),
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending,
                )

                batch_papers = []
                batch_count = 0

                # Collect results for this date range
                for result in search.results():
                    paper = self._extract_paper(result)
                    if paper:
                        batch_papers.append(paper)
                        batch_count += 1

                    # Rate limiting
                    time.sleep(self.delay_milliseconds / 1000.0)

                    if batch_count >= min(100, max_results - len(papers)):
                        break

                papers.extend(batch_papers)

                logger.info(
                    f"arXiv: Date range yielded {len(batch_papers)} papers, total: {len(papers)}"
                )

                # Conservative delay between date ranges
                time.sleep(2.0)

            except Exception as e:
                logger.error(f"arXiv: Error in date range {start_date}-{end_date}: {e}")
                continue

        logger.info(
            f"arXiv: Date-range splitting complete. Found {len(papers)} total papers"
        )

        # Deduplicate papers based on arXiv ID
        seen_ids = set()
        unique_papers = []
        for paper in papers:
            if paper.arxiv_id and paper.arxiv_id not in seen_ids:
                seen_ids.add(paper.arxiv_id)
                unique_papers.append(paper)
            elif not paper.arxiv_id:
                # If no arxiv_id, use title + year as deduplication key
                key = f"{paper.title}_{paper.year}"
                if key not in seen_ids:
                    seen_ids.add(key)
                    unique_papers.append(paper)

        logger.info(f"arXiv: After deduplication: {len(unique_papers)} unique papers")
        return unique_papers

    def _generate_monthly_ranges(
        self, start_date: str, end_date: str
    ) -> list[tuple[str, str]]:
        """Generate monthly date ranges for query splitting.

        Args:
            start_date: Start date in YYYY-MM format
            end_date: End date in YYYY-MM format

        Returns:
            List of (start_date, end_date) tuples in YYYYMMDD format
        """
        ranges = []
        current = datetime.strptime(start_date, "%Y-%m")
        end = datetime.strptime(end_date, "%Y-%m")

        while current <= end:
            # Calculate next month
            if current.month == 12:
                next_month = current.replace(year=current.year + 1, month=1)
            else:
                next_month = current.replace(month=current.month + 1)

            # Calculate last day of current month
            last_day = next_month - timedelta(days=1)

            ranges.append((current.strftime("%Y%m%d"), last_day.strftime("%Y%m%d")))

            current = next_month

        return ranges

    def _search_with_category_splitting(
        self, arxiv_query: str, max_results: int
    ) -> list[Paper]:
        """Split query by arXiv categories as fallback mechanism.

        Args:
            arxiv_query: Formatted arXiv query
            max_results: Maximum total results to collect

        Returns:
            List of Paper objects
        """
        papers = []

        # Target categories for LLM and wargaming research
        categories = ["cs.AI", "cs.CL", "cs.LG", "cs.GT", "cs.MA", "cs.CR"]

        logger.info(
            f"arXiv: Using category splitting strategy with {len(categories)} categories"
        )

        for i, category in enumerate(categories):
            if len(papers) >= max_results:
                break

            logger.info(
                f"arXiv: Processing category {i+1}/{len(categories)}: {category}"
            )

            # Add category filter to existing query
            category_query = f"({arxiv_query}) AND cat:{category}"

            try:
                search = arxiv.Search(
                    query=category_query,
                    max_results=min(100, max_results - len(papers)),
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending,
                )

                batch_papers = []
                batch_count = 0

                # Collect results for this category
                for result in search.results():
                    paper = self._extract_paper(result)
                    if paper:
                        batch_papers.append(paper)
                        batch_count += 1

                    # Rate limiting
                    time.sleep(self.delay_milliseconds / 1000.0)

                    if batch_count >= min(100, max_results - len(papers)):
                        break

                papers.extend(batch_papers)

                logger.info(
                    f"arXiv: Category {category} yielded {len(batch_papers)} papers, total: {len(papers)}"
                )

                # Conservative delay between categories
                time.sleep(2.0)

            except Exception as e:
                logger.error(f"arXiv: Error in category {category}: {e}")
                continue

        logger.info(
            f"arXiv: Category splitting complete. Found {len(papers)} total papers"
        )

        # Deduplicate papers based on arXiv ID
        seen_ids = set()
        unique_papers = []
        for paper in papers:
            if paper.arxiv_id and paper.arxiv_id not in seen_ids:
                seen_ids.add(paper.arxiv_id)
                unique_papers.append(paper)
            elif not paper.arxiv_id:
                # If no arxiv_id, use title + year as deduplication key
                key = f"{paper.title}_{paper.year}"
                if key not in seen_ids:
                    seen_ids.add(key)
                    unique_papers.append(paper)

        logger.info(f"arXiv: After deduplication: {len(unique_papers)} unique papers")
        return unique_papers

    def _search_with_pagination(
        self, arxiv_query: str, max_results: int
    ) -> list[Paper]:
        """Search arXiv with pagination to collect all available results.

        Args:
            arxiv_query: Formatted arXiv query
            max_results: Maximum total results to collect

        Returns:
            List of Paper objects
        """
        papers = []
        page_size = 100  # arXiv API batch size
        start = 0

        while len(papers) < max_results:
            current_page_size = min(page_size, max_results - len(papers))

            logger.info(
                f"arXiv: Fetching batch {start//page_size + 1}, "
                f"start={start}, size={current_page_size}"
            )

            # Create search for this batch with offset
            search = arxiv.Search(
                query=arxiv_query,
                max_results=current_page_size,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )

            batch_papers = []
            batch_count = 0

            try:
                # Execute search and collect results for this batch
                results_iter = search.results()

                # Skip to correct offset
                for _ in range(start):
                    try:
                        next(results_iter)
                    except StopIteration:
                        logger.info(f"arXiv: Reached end of results at offset {start}")
                        return papers

                # Collect this batch
                for i in range(current_page_size):
                    try:
                        result = next(results_iter)
                        paper = self._extract_paper(result)
                        if paper:
                            batch_papers.append(paper)
                            logger.debug(
                                f"arXiv: Added paper {start + i + 1}: {paper.title[:50]}..."
                            )

                        batch_count += 1

                        # Rate limiting
                        time.sleep(self.delay_milliseconds / 1000.0)

                    except StopIteration:
                        logger.info(
                            f"arXiv: Reached end of results after {batch_count} papers in batch"
                        )
                        break
                    except Exception as e:
                        logger.error(
                            f"arXiv: Error extracting paper {start + i + 1}: {e}"
                        )
                        continue

                # If we got fewer results than requested, we've hit the end
                if batch_count == 0:
                    logger.info("arXiv: No more results available")
                    break

                papers.extend(batch_papers)
                start += batch_count

                logger.info(
                    f"arXiv: Batch complete. Got {len(batch_papers)} papers, "
                    f"total: {len(papers)}"
                )

                # If we got fewer than expected, we're at the end
                if batch_count < current_page_size:
                    logger.info("arXiv: Reached end of available results")
                    break

                # Additional delay between batches to be respectful
                time.sleep(1.0)

            except Exception as e:
                logger.error(f"arXiv: Error in batch starting at {start}: {e}")
                # Try to continue with next batch
                start += page_size
                continue

        logger.info(f"arXiv: Pagination complete. Found {len(papers)} total papers")
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
                keywords=[
                    cat.term if hasattr(cat, "term") else str(cat)
                    for cat in result.categories
                ],
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

    def fetch_tex_source(self, arxiv_id: str) -> str | None:
        """Fetch the TeX source for an arXiv paper.

        Args:
            arxiv_id: arXiv ID (e.g., "2301.00234")

        Returns:
            TeX source content or None if not available
        """
        try:
            # Clean the arxiv ID (remove version if present)
            clean_id = arxiv_id.split("v")[0]

            # arXiv source URL
            source_url = f"https://arxiv.org/e-print/{clean_id}"

            logger.info(f"Fetching TeX source for arXiv:{clean_id}")

            # Add headers to identify as a research tool
            headers = {"User-Agent": "LiteratureReviewPipeline/1.0 (Research Tool)"}

            response = requests.get(source_url, headers=headers, timeout=30)
            response.raise_for_status()

            # arXiv returns tar.gz files for source
            # For now, return the raw content - we'll process it later
            content = response.content

            # Check if it's actually TeX (starts with common TeX commands)
            text_preview = content[:1000].decode("utf-8", errors="ignore")
            if any(
                cmd in text_preview
                for cmd in ["\\documentclass", "\\begin{document}", "\\section"]
            ):
                logger.info(f"Successfully fetched TeX source for {arxiv_id}")
                return content.decode("utf-8", errors="ignore")
            else:
                logger.warning(f"Source for {arxiv_id} doesn't appear to be TeX")
                return None

        except Exception as e:
            logger.error(f"Error fetching TeX source for {arxiv_id}: {e}")
            return None

    def fetch_html_source(self, arxiv_id: str) -> str | None:
        """Fetch the HTML version of an arXiv paper if available.

        Args:
            arxiv_id: arXiv ID (e.g., "2301.00234")

        Returns:
            HTML content or None if not available
        """
        try:
            # Clean the arxiv ID
            clean_id = arxiv_id.split("v")[0]

            # Some arXiv papers have HTML versions
            # Try the ar5iv service which converts arXiv papers to HTML
            html_url = f"https://ar5iv.org/abs/{clean_id}"

            logger.info(f"Fetching HTML version for arXiv:{clean_id}")

            headers = {"User-Agent": "LiteratureReviewPipeline/1.0 (Research Tool)"}

            response = requests.get(html_url, headers=headers, timeout=30)

            if response.status_code == 200:
                # Check if we got actual HTML content
                if "text/html" in response.headers.get("content-type", ""):
                    logger.info(f"Successfully fetched HTML for {arxiv_id}")
                    return response.text
                else:
                    logger.warning(f"Response for {arxiv_id} is not HTML")
                    return None
            else:
                logger.warning(
                    f"HTML not available for {arxiv_id} (status: {response.status_code})"
                )
                return None

        except Exception as e:
            logger.error(f"Error fetching HTML for {arxiv_id}: {e}")
            return None
