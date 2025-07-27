"""Paper normalizer for deduplication and data cleaning."""

import hashlib
import logging
import re
import unicodedata
from urllib.parse import urlparse

import pandas as pd
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)


class Normalizer:
    """Normalizes and deduplicates papers from multiple sources."""

    def __init__(self, config):
        """Initialize the normalizer.

        Args:
            config: Configuration object
        """
        self.config = config
        self.dedup_methods = config.dedup_methods
        self.title_threshold = config.title_similarity_threshold

        # Track deduplication statistics
        self.stats = {
            "total_input": 0,
            "doi_duplicates": 0,
            "title_duplicates": 0,
            "arxiv_duplicates": 0,
            "total_output": 0,
        }

    def normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize and deduplicate a DataFrame of papers.

        Args:
            df: DataFrame with paper data

        Returns:
            Normalized and deduplicated DataFrame
        """
        logger.info(f"Starting normalization of {len(df)} papers")
        self.stats["total_input"] = len(df)

        # Step 1: Clean and normalize data
        df = self._normalize_fields(df)

        # Step 2: Add normalized identifiers
        df = self._add_normalized_ids(df)

        # Step 3: Deduplicate
        df = self._deduplicate(df)

        # Step 4: Merge duplicate entries
        df = self._merge_duplicates(df)

        # Step 5: Final validation
        df = self._validate_papers(df)

        self.stats["total_output"] = len(df)
        self._log_statistics()

        # Debug logging
        logger.info(f"Final columns in normalize_dataframe: {list(df.columns)}")

        return df

    def _normalize_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize all fields.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with normalized fields
        """
        # Ensure required columns exist
        required_cols = ["title", "doi", "authors", "abstract", "url", "year"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = ""

        # Normalize titles
        df["title_normalized"] = df["title"].apply(self._normalize_title)

        # Normalize DOIs
        df["doi_normalized"] = df["doi"].apply(self._normalize_doi)

        # Normalize author names
        df["authors_normalized"] = df["authors"].apply(self._normalize_authors)

        # Clean abstracts
        df["abstract"] = df["abstract"].apply(self._clean_text)

        # Normalize URLs
        df["url"] = df["url"].apply(self._normalize_url)

        # Ensure years are integers
        df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)

        return df

    def _normalize_title(self, title: str) -> str:
        """Normalize a paper title for comparison.

        Args:
            title: Original title

        Returns:
            Normalized title
        """
        if pd.isna(title) or not title:
            return ""

        # Convert to lowercase
        title = title.lower()

        # Remove unicode accents
        title = unicodedata.normalize("NFKD", title)
        title = "".join([c for c in title if not unicodedata.combining(c)])

        # Remove punctuation and extra whitespace
        title = re.sub(r"[^\w\s]", " ", title)
        title = " ".join(title.split())

        # Remove common words that might differ
        stop_words = {
            "a",
            "an",
            "the",
            "of",
            "in",
            "on",
            "at",
            "to",
            "for",
            "with",
            "and",
            "or",
        }
        words = [w for w in title.split() if w not in stop_words]

        return " ".join(words)

    def _normalize_doi(self, doi: str) -> str:
        """Normalize a DOI for comparison.

        Args:
            doi: Original DOI

        Returns:
            Normalized DOI
        """
        if pd.isna(doi) or not doi:
            return ""

        # Convert to lowercase
        doi = doi.lower().strip()

        # Remove common prefixes
        doi = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", doi)
        doi = re.sub(r"^doi:", "", doi)

        # Ensure it starts with 10.
        if not doi.startswith("10."):
            return ""

        return doi

    def _normalize_authors(self, authors: str) -> str:
        """Normalize author names for comparison.

        Args:
            authors: Semi-colon separated author names

        Returns:
            Normalized author string
        """
        if pd.isna(authors) or not authors:
            return ""

        # Split authors
        author_list = authors.split(";")

        # Normalize each author
        normalized = []
        for author in author_list:
            author = author.strip()
            if author:
                # Remove titles (Dr., Prof., etc.)
                author = re.sub(
                    r"\b(Union[Dr, Prof]|Union[Professor, Mr]|Union[Mrs, Ms]|Union[PhD, Ph]\.D)\b\.?",
                    "",
                    author,
                )
                # Normalize whitespace
                author = " ".join(author.split())
                if author:
                    normalized.append(author)

        return "; ".join(normalized)

    def _normalize_url(self, url: str) -> str:
        """Normalize URLs.

        Args:
            url: Original URL

        Returns:
            Normalized URL
        """
        if pd.isna(url) or not url:
            return ""

        try:
            # Parse URL
            parsed = urlparse(url)

            # Rebuild with just scheme, netloc, and path
            normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

            # Remove trailing slashes
            normalized = normalized.rstrip("/")

            return normalized
        except Exception:
            return url

    def _clean_text(self, text: str) -> str:
        """Clean text fields like abstracts.

        Args:
            text: Original text

        Returns:
            Cleaned text
        """
        if pd.isna(text) or not text:
            return ""

        # Remove excessive whitespace
        text = " ".join(text.split())

        # Remove non-printable characters
        text = "".join(char for char in text if char.isprintable() or char.isspace())

        return text.strip()

    def _add_normalized_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add normalized identifier columns.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with added ID columns
        """
        # Create title slugs for fuzzy matching
        df["title_slug"] = df["title_normalized"].apply(self._create_slug)

        # Create content hashes
        df["content_hash"] = df.apply(self._create_content_hash, axis=1)

        return df

    def _create_slug(self, text: str) -> str:
        """Create a slug from text for exact matching.

        Args:
            text: Input text

        Returns:
            Slug string
        """
        if not text:
            return ""

        # Keep only alphanumeric characters
        slug = re.sub(r"[^a-z0-9]+", "", text)

        # Take first 50 characters
        return slug[:50]

    def _create_content_hash(self, row: pd.Series) -> str:
        """Create a hash of paper content for deduplication.

        Args:
            row: Paper data row

        Returns:
            SHA256 hash string
        """
        # Combine key fields
        content = f"{row['title_normalized']}|{row['authors_normalized']}|{row['year']}"

        # Create hash
        return hashlib.sha256(content.encode()).hexdigest()

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate papers using configured methods.

        Args:
            df: Input DataFrame

        Returns:
            Deduplicated DataFrame
        """
        original_count = len(df)

        for method in self.dedup_methods:
            if method == "doi_exact":
                df = self._deduplicate_by_doi(df)
            elif method == "title_fuzzy":
                df = self._deduplicate_by_title(df)
            elif method == "arxiv_exact":
                df = self._deduplicate_by_arxiv(df)
            elif method == "content_hash":
                df = self._deduplicate_by_hash(df)

        logger.info(f"Deduplication removed {original_count - len(df)} papers")

        return df

    def _deduplicate_by_doi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates by DOI.

        Args:
            df: Input DataFrame

        Returns:
            Deduplicated DataFrame
        """
        # Separate papers with and without DOI
        has_doi = df["doi_normalized"].notna() & (df["doi_normalized"] != "")
        df_with_doi = df[has_doi].copy()
        df_without_doi = df[~has_doi].copy()

        # Count duplicates before removal
        doi_duplicates = len(df_with_doi) - df_with_doi["doi_normalized"].nunique()
        self.stats["doi_duplicates"] += doi_duplicates

        # Remove duplicates, keeping the one with most information
        df_with_doi["info_score"] = (
            df_with_doi["abstract"].str.len().fillna(0)
            + df_with_doi["authors"].str.len().fillna(0)
            + (df_with_doi["citations"].fillna(0) > 0).astype(int) * 100
        )

        df_with_doi = df_with_doi.sort_values("info_score", ascending=False)
        df_with_doi = df_with_doi.drop_duplicates(
            subset=["doi_normalized"], keep="first"
        )
        df_with_doi = df_with_doi.drop("info_score", axis=1)

        # Combine back
        return pd.concat([df_with_doi, df_without_doi], ignore_index=True)

    def _deduplicate_by_title(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates by fuzzy title matching.

        Args:
            df: Input DataFrame

        Returns:
            Deduplicated DataFrame
        """
        # Group by title slug first for efficiency
        groups = df.groupby("title_slug")

        keep_indices = []

        for _slug, group in groups:
            if len(group) == 1:
                # No duplicates in this slug group
                keep_indices.extend(group.index.tolist())
            else:
                # Check for fuzzy matches within group
                titles = group["title_normalized"].tolist()
                indices = group.index.tolist()

                # Track which indices to keep
                processed = set()

                for i, (idx1, title1) in enumerate(zip(indices, titles, strict=False)):
                    if idx1 in processed:
                        continue

                    # This one will be kept
                    keep_indices.append(idx1)
                    processed.add(idx1)

                    # Check against remaining titles
                    for j in range(i + 1, len(titles)):
                        idx2 = indices[j]
                        if idx2 in processed:
                            continue

                        title2 = titles[j]

                        # Calculate similarity
                        similarity = fuzz.ratio(title1, title2) / 100.0

                        if similarity >= self.title_threshold:
                            # Mark as duplicate
                            processed.add(idx2)
                            self.stats["title_duplicates"] += 1

        # Return filtered DataFrame
        return df.loc[keep_indices].copy()

    def _deduplicate_by_arxiv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates by arXiv ID.

        Args:
            df: Input DataFrame

        Returns:
            Deduplicated DataFrame
        """
        # Similar to DOI deduplication
        has_arxiv = df["arxiv_id"].notna() & (df["arxiv_id"] != "")
        df_with_arxiv = df[has_arxiv].copy()
        df_without_arxiv = df[~has_arxiv].copy()

        # Count duplicates
        arxiv_duplicates = len(df_with_arxiv) - df_with_arxiv["arxiv_id"].nunique()
        self.stats["arxiv_duplicates"] += arxiv_duplicates

        # Remove duplicates
        df_with_arxiv = df_with_arxiv.drop_duplicates(subset=["arxiv_id"], keep="first")

        return pd.concat([df_with_arxiv, df_without_arxiv], ignore_index=True)

    def _deduplicate_by_hash(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates by content hash.

        Args:
            df: Input DataFrame

        Returns:
            Deduplicated DataFrame
        """
        return df.drop_duplicates(subset=["content_hash"], keep="first")

    def _merge_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge information from duplicate entries.

        Args:
            df: Deduplicated DataFrame

        Returns:
            DataFrame with merged entries
        """
        # This is a placeholder for more sophisticated merging
        # For now, the deduplication keeps the most informative version
        return df

    def _validate_papers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate papers meet minimum requirements.

        Args:
            df: Input DataFrame

        Returns:
            Validated DataFrame
        """
        original_count = len(df)

        # Remove papers without titles
        df = df[df["title"].notna() & (df["title"] != "")]

        # Remove papers outside year range
        start_year, end_year = self.config.search_years
        df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]

        # Remove papers without abstracts if required
        # Note: For now, always require abstracts with minimum length
        df = df[df["abstract"].notna() & (df["abstract"].str.len() > 50)]

        removed = original_count - len(df)
        if removed > 0:
            logger.info(f"Validation removed {removed} papers")

        # Drop temporary columns
        columns_to_drop = [
            "title_normalized",
            "doi_normalized",
            "authors_normalized",
            "title_slug",
            "content_hash",
        ]
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

        return df

    def _log_statistics(self):
        """Log deduplication statistics."""
        logger.info("Normalization statistics:")
        logger.info(f"  Total input papers: {self.stats['total_input']}")
        logger.info(f"  DOI duplicates removed: {self.stats['doi_duplicates']}")
        logger.info(f"  Title duplicates removed: {self.stats['title_duplicates']}")
        logger.info(f"  ArXiv duplicates removed: {self.stats['arxiv_duplicates']}")
        logger.info(f"  Total output papers: {self.stats['total_output']}")
        logger.info(
            f"  Total removed: {self.stats['total_input'] - self.stats['total_output']}"
        )

    def find_similar_papers(
        self, df: pd.DataFrame, title: str, threshold: float = 0.8
    ) -> pd.DataFrame:
        """Find papers similar to a given title.

        Args:
            df: DataFrame of papers
            title: Title to search for
            threshold: Similarity threshold (0-1)

        Returns:
            DataFrame of similar papers
        """
        # Normalize the search title
        title_norm = self._normalize_title(title)

        # Calculate similarities
        titles = df["title"].apply(self._normalize_title).tolist()

        # Use rapidfuzz for efficient fuzzy matching
        matches = process.extract(title_norm, titles, scorer=fuzz.ratio, limit=10)

        # Filter by threshold
        similar_indices = []
        for _match, score, idx in matches:
            if score / 100.0 >= threshold:
                similar_indices.append(idx)

        return df.iloc[similar_indices].copy()
