"""Disambiguator module for post-search filtering based on context."""

import logging
import re
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class Disambiguator:
    """Apply disambiguation rules to filter out false positive papers."""

    def __init__(self, config: Any) -> None:
        """Initialize disambiguator with configuration.

        Args:
            config: Configuration object with disambiguation rules
        """
        self.config = config
        self.rules = config.disambiguation if hasattr(config, "disambiguation") else {}
        self.grey_lit_sources = (
            config.grey_lit_sources if hasattr(config, "grey_lit_sources") else []
        )

        # Track disambiguation statistics
        self.stats = {
            "total_papers": 0,
            "matrix_game_filtered": 0,
            "red_teaming_filtered": 0,
            "rl_board_game_filtered": 0,
            "generic_surveys_filtered": 0,
            "grey_lit_tagged": 0,
        }

    def apply_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all disambiguation rules to filter papers.

        Args:
            df: DataFrame of papers to filter

        Returns:
            Filtered DataFrame with disambiguation applied
        """
        logger.info(f"Starting disambiguation on {len(df)} papers")
        self.stats["total_papers"] = len(df)

        # Add columns for tracking
        df["disambiguation_status"] = "passed"
        df["disambiguation_reason"] = ""
        df["grey_lit_flag"] = False

        # Apply grey literature tagging first
        df = self._tag_grey_literature(df)

        # Apply each disambiguation rule
        for rule_name, rule_config in self.rules.items():
            df = self._apply_rule(df, rule_name, rule_config)

        # Log statistics
        self._log_statistics()

        # Filter out excluded papers
        filtered_df = df[df["disambiguation_status"] == "passed"].copy()
        logger.info(f"After disambiguation: {len(filtered_df)} papers remain")

        return filtered_df

    def _apply_rule(
        self, df: pd.DataFrame, rule_name: str, rule_config: dict[str, Any]
    ) -> pd.DataFrame:
        """Apply a single disambiguation rule.

        Args:
            df: DataFrame of papers
            rule_name: Name of the rule
            rule_config: Configuration for the rule

        Returns:
            DataFrame with rule applied
        """
        negative_context = rule_config.get("negative_context", [])
        positive_required = rule_config.get("positive_required", [])

        for idx, row in df.iterrows():
            # Skip if already excluded
            if row["disambiguation_status"] != "passed":
                continue

            # Combine title and abstract for context checking
            text = f"{row.get('title', '')} {row.get('abstract', '')}".lower()

            # Check negative context
            has_negative = any(term.lower() in text for term in negative_context)

            # For generic_surveys rule, also check for positive requirements
            if rule_name == "generic_surveys" and has_negative:
                has_positive = any(term.lower() in text for term in positive_required)
                if not has_positive:
                    df.at[idx, "disambiguation_status"] = "excluded"
                    df.at[idx, "disambiguation_reason"] = (
                        f"{rule_name}: negative context without required positive terms"
                    )
                    self.stats[f"{rule_name}_filtered"] += 1
            elif has_negative:
                # For other rules, exclude if negative context found
                df.at[idx, "disambiguation_status"] = "excluded"
                df.at[idx, "disambiguation_reason"] = (
                    f"{rule_name}: negative context detected"
                )
                self.stats[f"{rule_name}_filtered"] += 1

        return df

    def _tag_grey_literature(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tag papers from grey literature sources.

        Args:
            df: DataFrame of papers

        Returns:
            DataFrame with grey_lit_flag set
        """
        for idx, row in df.iterrows():
            url = str(row.get("url", ""))
            source = str(row.get("source_db", ""))

            # Check if URL contains grey lit domains
            for grey_source in self.grey_lit_sources:
                if grey_source in url or grey_source in source:
                    df.at[idx, "grey_lit_flag"] = True
                    self.stats["grey_lit_tagged"] += 1
                    break

        return df

    def apply_near_operator(self, text: str, pattern: str) -> bool:
        """Apply NEAR operator for proximity search.

        Args:
            text: Text to search in
            pattern: Pattern with NEAR operator (e.g., '"term1" NEAR/5 (term2 OR term3)')

        Returns:
            True if pattern matches
        """
        # Parse NEAR pattern
        near_match = re.match(r'"([^"]+)"\s*NEAR/(\d+)\s*\(([^)]+)\)', pattern)
        if not near_match:
            # If not a NEAR pattern, do simple substring search
            return pattern.lower() in text.lower()

        term1 = near_match.group(1)
        distance = int(near_match.group(2))
        term2_group = near_match.group(3)

        # Parse OR terms in the second group
        term2_options = [t.strip().strip('"') for t in term2_group.split(" OR ")]

        # Find all occurrences of term1
        term1_positions = [
            m.start() for m in re.finditer(re.escape(term1.lower()), text.lower())
        ]

        # Check if any term2 option is within distance
        for pos1 in term1_positions:
            for term2 in term2_options:
                term2_positions = [
                    m.start()
                    for m in re.finditer(re.escape(term2.lower()), text.lower())
                ]
                for pos2 in term2_positions:
                    # Count words between positions
                    between_text = text[min(pos1, pos2) : max(pos1, pos2)]
                    word_count = len(between_text.split())
                    if word_count <= distance:
                        return True

        return False

    def get_excluded_papers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get papers that were excluded by disambiguation.

        Args:
            df: DataFrame after disambiguation

        Returns:
            DataFrame of excluded papers
        """
        return df[df["disambiguation_status"] == "excluded"].copy()

    def _log_statistics(self) -> None:
        """Log disambiguation statistics."""
        logger.info("Disambiguation statistics:")
        logger.info(f"  Total papers: {self.stats['total_papers']}")
        logger.info(f"  Matrix game filtered: {self.stats['matrix_game_filtered']}")
        logger.info(f"  Red teaming filtered: {self.stats['red_teaming_filtered']}")
        logger.info(f"  RL board game filtered: {self.stats['rl_board_game_filtered']}")
        logger.info(
            f"  Generic surveys filtered: {self.stats['generic_surveys_filtered']}"
        )
        logger.info(f"  Grey literature tagged: {self.stats['grey_lit_tagged']}")

        total_filtered = sum(
            self.stats[key] for key in self.stats if key.endswith("_filtered")
        )
        logger.info(f"  Total filtered: {total_filtered}")

    def create_disambiguation_report(self, df: pd.DataFrame) -> dict[str, Any]:
        """Create a detailed disambiguation report.

        Args:
            df: DataFrame after disambiguation

        Returns:
            Dictionary with disambiguation statistics and examples
        """
        excluded_df = self.get_excluded_papers(df)

        report = {
            "statistics": self.stats.copy(),
            "exclusion_reasons": {},
            "grey_literature": {
                "count": len(df[df["grey_lit_flag"] == True]),
                "sources": (
                    df[df["grey_lit_flag"] == True]["source_db"]
                    .value_counts()
                    .to_dict()
                    if "source_db" in df.columns
                    else {}
                ),
            },
        }

        # Group by exclusion reason
        if len(excluded_df) > 0:
            reason_counts = excluded_df["disambiguation_reason"].value_counts()
            report["exclusion_reasons"] = reason_counts.to_dict()

            # Add examples for each reason
            report["examples"] = {}
            for reason in reason_counts.index[:5]:  # Top 5 reasons
                examples = excluded_df[
                    excluded_df["disambiguation_reason"] == reason
                ].head(2)
                report["examples"][reason] = [
                    {
                        "title": str(row["title"]),
                        "year": str(row.get("year", "")),
                        "abstract_snippet": str(row.get("abstract", ""))[:200] + "...",
                    }
                    for _, row in examples.iterrows()
                ]

        return report
