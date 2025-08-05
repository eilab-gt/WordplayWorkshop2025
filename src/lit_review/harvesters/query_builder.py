"""Advanced query builder with NEAR operator and wildcard support."""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class QueryBuilder:
    """Build and translate queries with NEAR operators and wildcards for different search engines."""

    def __init__(self) -> None:
        """Initialize the query builder."""
        self.parsed_terms: dict[str, Any] = {}

    def normalize_encoding(self, text: str) -> str:
        """Normalize character encoding, particularly non-standard hyphens.

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        # Replace non-standard hyphens (U+2011) with standard hyphens
        text = text.replace("\u2011", "-")  # Non-breaking hyphen
        text = text.replace("\u2012", "-")  # Figure dash
        text = text.replace("\u2013", "-")  # En dash
        text = text.replace("\u2014", "-")  # Em dash

        return text

    def parse_query_term(self, term: str) -> dict[str, Any]:
        """Parse a query term to identify NEAR operators, wildcards, and other patterns.

        Args:
            term: Query term that may contain NEAR operators or wildcards

        Returns:
            Dictionary with parsed components
        """
        # Normalize encoding first
        term = self.normalize_encoding(term)

        # Check for NEAR operator pattern: "term1" NEAR/N (term2 OR term3)
        near_pattern = r'"([^"]+)"\s*NEAR/(\d+)\s*\(([^)]+)\)'
        near_match = re.match(near_pattern, term.strip())

        if near_match:
            return {
                "type": "near",
                "term1": near_match.group(1),
                "distance": int(near_match.group(2)),
                "term2_group": self._parse_or_group(near_match.group(3)),
            }

        # Check for wildcard
        if "*" in term:
            return {"type": "wildcard", "pattern": term}

        # Check for quoted phrase
        if term.startswith('"') and term.endswith('"'):
            return {"type": "phrase", "text": term[1:-1]}

        # Regular term
        return {"type": "term", "text": term}

    def _parse_or_group(self, or_group: str) -> list[str]:
        """Parse OR group like 'term1 OR term2 OR "term 3"'.

        Args:
            or_group: String containing OR-separated terms

        Returns:
            List of individual terms
        """
        # Split by OR and clean up
        terms = []
        for part in or_group.split(" OR "):
            cleaned = part.strip().strip('"')
            if cleaned:
                terms.append(cleaned)
        return terms

    def build_query_from_config(self, config: Any) -> str:
        """Build a query string from configuration with proper handling of advanced operators.

        Args:
            config: Configuration object with search terms

        Returns:
            Query string for the base search
        """
        # Normalize all terms first
        wargame_terms = [self.normalize_encoding(t) for t in config.wargame_terms]
        llm_terms = [self.normalize_encoding(t) for t in config.llm_terms]
        exclusion_terms = [self.normalize_encoding(t) for t in config.exclusion_terms]

        # Parse all normalized terms
        parsed_wargame = [self.parse_query_term(t) for t in wargame_terms]
        parsed_llm = [self.parse_query_term(t) for t in llm_terms]
        parsed_exclusions = [self.parse_query_term(t) for t in exclusion_terms]

        # Build the query with parsed terms
        wargame_part = self._build_term_group(parsed_wargame)
        llm_part = self._build_term_group(parsed_llm)

        # Build exclusions
        exclusion_parts = []
        for exc in parsed_exclusions:
            if exc["type"] == "near":
                # For NEAR exclusions, we need to handle them specially
                exclusion_parts.append(f"NOT ({self._format_parsed_term(exc)})")
            else:
                exclusion_parts.append(f"NOT {self._format_parsed_term(exc)}")

        # Combine parts
        query = f"({wargame_part}) AND ({llm_part})"
        if exclusion_parts:
            query = f"{query} {' '.join(exclusion_parts)}"

        return query

    def build_secondary_queries(self, config: Any) -> list[dict[str, str]]:
        """Build secondary queries from configuration templates.

        Args:
            config: Configuration object with query strategies

        Returns:
            List of dicts with 'description' and 'query' keys
        """
        secondary_queries = []

        # Get secondary strategies from config
        strategies = config.query_strategies.get("secondary", [])

        for strategy in strategies:
            template = strategy.get("template", "")
            description = strategy.get("description", "")

            # Replace template variables
            query = template

            # For policy/diplomacy, template already has specific terms
            # For grey-lit, template has site: and filetype: operators
            # We just need to replace exclusion_terms

            # Build exclusion string
            exclusion_parts = []
            for term in config.exclusion_terms:
                parsed = self.parse_query_term(term)
                if parsed["type"] == "near":
                    exclusion_parts.append(f"({self._format_parsed_term(parsed)})")
                else:
                    exclusion_parts.append(self._format_parsed_term(parsed))

            exclusion_str = " OR ".join(exclusion_parts)
            query = query.replace("{exclusion_terms}", exclusion_str)

            # Also replace llm_terms if present
            if "{llm_terms}" in query:
                llm_parts = []
                for term in config.llm_terms:
                    parsed = self.parse_query_term(term)
                    llm_parts.append(self._format_parsed_term(parsed))
                llm_str = " OR ".join(llm_parts)
                query = query.replace("{llm_terms}", llm_str)

            # Clean up whitespace
            query = " ".join(query.split())

            secondary_queries.append({"description": description, "query": query})

        return secondary_queries

    def _build_term_group(self, parsed_terms: list[dict[str, Any]]) -> str:
        """Build a term group from parsed terms.

        Args:
            parsed_terms: List of parsed term dictionaries

        Returns:
            Formatted term group string
        """
        formatted_terms = []
        for term in parsed_terms:
            formatted = self._format_parsed_term(term)
            if formatted:
                formatted_terms.append(formatted)

        return " OR ".join(formatted_terms)

    def _format_parsed_term(self, parsed_term: dict[str, Any]) -> str:
        """Format a parsed term for the base query.

        Args:
            parsed_term: Parsed term dictionary

        Returns:
            Formatted term string
        """
        if parsed_term["type"] == "near":
            # For base query, convert NEAR to a simple OR of all terms
            all_terms = [f'"{parsed_term["term1"]}"']
            all_terms.extend([f'"{t}"' for t in parsed_term["term2_group"]])
            return f"({' OR '.join(all_terms)})"

        elif parsed_term["type"] == "wildcard":
            # Keep wildcard as-is for now
            return str(parsed_term["pattern"])

        elif parsed_term["type"] == "phrase":
            return f'"{parsed_term["text"]}"'

        else:  # regular term
            return str(parsed_term["text"])

    def translate_for_google_scholar(self, query: str) -> str:
        """Translate query for Google Scholar syntax.

        Google Scholar doesn't support NEAR, so we convert to alternatives.

        Args:
            query: Base query string

        Returns:
            Google Scholar compatible query
        """
        # Parse the full query to handle NEAR operators
        translated = query

        # Find all NEAR patterns and replace them
        near_pattern = r'"([^"]+)"\s*NEAR/\d+\s*\(([^)]+)\)'

        def replace_near(match: re.Match[str]) -> str:
            term1 = match.group(1)
            term2_group = match.group(2)
            # For Google Scholar, convert NEAR to quoted phrases with all combinations
            terms = self._parse_or_group(term2_group)
            # Use allintitle: for more precise matching
            combinations = [f'"{term1} {t}"' for t in terms]
            return f"({' OR '.join(combinations)})"

        translated = re.sub(near_pattern, replace_near, translated)

        # Handle wildcards - Google Scholar doesn't support *, so remove them
        translated = translated.replace("*", "")

        return translated

    def translate_for_arxiv(self, query: str) -> str:
        """Translate query for arXiv API syntax.

        arXiv uses its own query syntax with ti:, abs:, au: prefixes.

        Args:
            query: Base query string

        Returns:
            arXiv compatible query
        """
        # For arXiv, we search in both title and abstract
        # Convert NEAR to simple AND since arXiv doesn't support proximity

        near_pattern = r'"([^"]+)"\s*NEAR/\d+\s*\(([^)]+)\)'

        def replace_near(match: re.Match[str]) -> str:
            term1 = match.group(1)
            term2_group = match.group(2)
            terms = self._parse_or_group(term2_group)
            # For arXiv, just use AND between terms
            quoted_terms = [f'"{t}"' for t in terms]
            return f'("{term1}" AND ({" OR ".join(quoted_terms)}))'

        translated = re.sub(near_pattern, replace_near, query)

        # Handle wildcards - arXiv doesn't support wildcards
        translated = translated.replace("*", "")

        # Add field prefixes for better results
        # Split the query into parts and add abs: prefix
        parts = []

        # Simple approach: wrap the whole query with abs: and ti:
        if translated:
            parts.append(f"abs:({translated})")
            parts.append(f"ti:({translated})")

        return " OR ".join(parts) if parts else translated

    def translate_for_semantic_scholar(self, query: str) -> str:
        """Translate query for Semantic Scholar API.

        Semantic Scholar has limited query syntax support.

        Args:
            query: Base query string

        Returns:
            Semantic Scholar compatible query
        """
        # Semantic Scholar doesn't support NEAR or complex boolean queries
        # Simplify to basic terms

        # Remove NEAR patterns and convert to simple terms
        near_pattern = r'"([^"]+)"\s*NEAR/\d+\s*\(([^)]+)\)'

        def replace_near(match: re.Match[str]) -> str:
            term1 = match.group(1)
            term2_group = match.group(2)
            terms = self._parse_or_group(term2_group)
            # Just include all terms
            all_terms = [term1, *terms]
            return " ".join(all_terms)

        translated = re.sub(near_pattern, replace_near, query)

        # Remove wildcards
        translated = translated.replace("*", "")

        # Simplify boolean operators
        translated = translated.replace(" AND ", " ")
        translated = translated.replace(" OR ", " ")
        translated = translated.replace(" NOT ", " -")

        # Remove excessive parentheses
        translated = re.sub(r"\(([^()]+)\)", r"\1", translated)

        return translated

    def translate_for_crossref(self, query: str) -> str:
        """Translate query for CrossRef API.

        CrossRef uses simple keyword search.

        Args:
            query: Base query string

        Returns:
            CrossRef compatible query
        """
        # Similar to Semantic Scholar, simplify the query

        # Remove NEAR patterns
        near_pattern = r'"([^"]+)"\s*NEAR/\d+\s*\(([^)]+)\)'

        def replace_near(match: re.Match[str]) -> str:
            term1 = match.group(1)
            # For CrossRef, just use the primary term
            return term1

        translated = re.sub(near_pattern, replace_near, query)

        # Remove wildcards
        translated = translated.replace("*", "")

        # Keep AND/OR but remove NOT (CrossRef doesn't support negation well)
        not_pattern = r'\s*NOT\s+"[^"]+"\s*'
        translated = re.sub(not_pattern, " ", translated)

        # Clean up quotes and parentheses for simpler query
        translated = translated.replace('"', "")
        translated = re.sub(r"\s+", " ", translated)

        return translated.strip()

    def expand_wildcards(self, pattern: str, known_terms: list[str]) -> list[str]:
        """Expand wildcard patterns against known terms.

        Args:
            pattern: Pattern with wildcard (e.g., "wargam*")
            known_terms: List of known terms to match against

        Returns:
            List of matching terms
        """
        if "*" not in pattern:
            return [pattern]

        # Convert wildcard pattern to regex
        regex_pattern = pattern.replace("*", ".*")
        regex_pattern = f"^{regex_pattern}$"

        matches = []
        for term in known_terms:
            if re.match(regex_pattern, term, re.IGNORECASE):
                matches.append(term)

        # If no matches, return the original pattern without wildcard
        return matches if matches else [pattern.replace("*", "")]
