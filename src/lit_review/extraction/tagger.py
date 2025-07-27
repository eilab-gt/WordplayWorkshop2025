"""Rule-based tagger for failure modes and metadata extraction."""

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)


class Tagger:
    """Tags papers with failure modes and other metadata using rules."""

    def __init__(self, config):
        """Initialize tagger with configuration.

        Args:
            config: Configuration object
        """
        self.config = config
        self.failure_vocab = config.failure_vocab

        # Build regex patterns for failure modes
        self.failure_patterns = self._build_failure_patterns()

        # Build patterns for other metadata
        self.metadata_patterns = self._build_metadata_patterns()

        # Track statistics
        self.stats = {
            "total_processed": 0,
            "papers_with_failures": 0,
            "total_failure_tags": 0,
            "pattern_matches": {},
        }

    def tag_papers(
        self, df: pd.DataFrame, use_llm_results: bool = True
    ) -> pd.DataFrame:
        """Tag all papers with failure modes and metadata.

        Args:
            df: DataFrame with paper information
            use_llm_results: Whether to use existing LLM results as base

        Returns:
            DataFrame with added/updated tags
        """
        logger.info(f"Starting tagging for {len(df)} papers")
        self.stats["total_processed"] = len(df)

        # Initialize columns if not present
        if "failure_modes" not in df.columns:
            df["failure_modes"] = ""
        if "failure_modes_regex" not in df.columns:
            df["failure_modes_regex"] = ""

        # Process each paper
        for idx, paper in df.iterrows():
            # Get existing failure modes from LLM
            existing_modes = set()
            if use_llm_results and paper.get("failure_modes"):
                existing_modes = set(str(paper["failure_modes"]).split("|"))

            # Extract failure modes from text
            text_to_search = self._prepare_search_text(paper)
            regex_modes = self._extract_failure_modes(text_to_search)

            # Combine modes
            all_modes = Union[existing_modes, regex_modes]

            # Update DataFrame
            df.at[idx, "failure_modes"] = (
                "|".join(sorted(all_modes)) if all_modes else ""
            )
            df.at[idx, "failure_modes_regex"] = (
                "|".join(sorted(regex_modes)) if regex_modes else ""
            )

            if all_modes:
                self.stats["papers_with_failures"] += 1
                self.stats["total_failure_tags"] += len(all_modes)

        # Apply additional metadata tagging
        df = self._tag_metadata(df)

        self._log_statistics()
        return df

    def _build_failure_patterns(self) -> dict[str, list[re.Pattern]]:
        """Build regex patterns for failure mode detection.

        Returns:
            Dictionary mapping failure categories to regex patterns
        """
        patterns = {}

        # Content-related failures
        patterns["bias"] = [
            re.compile(r"\bbias(?:ed)?\b", re.IGNORECASE),
            re.compile(r"\bstereotyp\w+\b", re.IGNORECASE),
            re.compile(r"\bdiscriminat\w+\b", re.IGNORECASE),
            re.compile(r"\bunfair\w*\b", re.IGNORECASE),
            re.compile(r"\bprejudic\w+\b", re.IGNORECASE),
        ]

        patterns["hallucination"] = [
            re.compile(r"\bhallucinat\w+\b", re.IGNORECASE),
            re.compile(r"\bconfabulat\w+\b", re.IGNORECASE),
            re.compile(r"\bfabricat\w+\b", re.IGNORECASE),
            re.compile(r"\bmade[\s-]?up\b", re.IGNORECASE),
            re.compile(r"\bfalse\s+(?:Union[information, facts]?|claims?)\b", re.IGNORECASE),
        ]

        patterns["factual_error"] = [
            re.compile(
                r"\bfactual(?:ly)?\s+(?:Union[error, incorrect]|wrong)\b", re.IGNORECASE
            ),
            re.compile(r"\binaccura\w+\b", re.IGNORECASE),
            re.compile(r"\bincorrect\s+(?:facts?|information)\b", re.IGNORECASE),
            re.compile(r"\bmisinformation\b", re.IGNORECASE),
        ]

        patterns["inconsistency"] = [
            re.compile(r"\binconsisten\w+\b", re.IGNORECASE),
            re.compile(r"\bcontradict\w+\b", re.IGNORECASE),
            re.compile(r"\bincoher\w+\b", re.IGNORECASE),
            re.compile(r"\bunstable\s+(?:Union[behavior, output])\b", re.IGNORECASE),
        ]

        # Interactive failures
        patterns["escalation"] = [
            re.compile(r"\bescalat\w+\b", re.IGNORECASE),
            re.compile(r"\bspiral(?:ing)?\b", re.IGNORECASE),
            re.compile(r"\baggressive\s+(?:Union[behavior, response])\b", re.IGNORECASE),
            re.compile(r"\bconflict\s+(?:Union[escalation, spiral])\b", re.IGNORECASE),
        ]

        patterns["deception"] = [
            re.compile(r"\bdecept\w+\b", re.IGNORECASE),
            re.compile(r"\bdeceiv\w+\b", re.IGNORECASE),
            re.compile(r"\bmislead\w+\b", re.IGNORECASE),
            re.compile(r"\bmanipulat\w+\b", re.IGNORECASE),
            re.compile(r"\bdishonest\w*\b", re.IGNORECASE),
        ]

        patterns["prompt_sensitivity"] = [
            re.compile(r"\bprompt[\s-]?(?:Union[sensitiv, engineer]|hack)\w*\b", re.IGNORECASE),
            re.compile(r"\bprompt[\s-]?injection\b", re.IGNORECASE),
            re.compile(r"\bsensitive\s+to\s+(?:Union[prompt, input])\b", re.IGNORECASE),
            re.compile(r"\bunstable\s+(?:Union[to, with])\s+prompt\b", re.IGNORECASE),
        ]

        # Security failures
        patterns["data_leakage"] = [
            re.compile(r"\bdata[\s-]?leak\w*\b", re.IGNORECASE),
            re.compile(r"\bprivacy[\s-]?(?:Union[breach, violation]|leak)\b", re.IGNORECASE),
            re.compile(r"\bunauthorized\s+(?:Union[access, disclosure])\b", re.IGNORECASE),
            re.compile(r"\bexpos(?:Union[e, ing])\s+(?:Union[private, sensitive])\b", re.IGNORECASE),
        ]

        patterns["jailbreak"] = [
            re.compile(r"\bjailbreak\w*\b", re.IGNORECASE),
            re.compile(
                r"\bbypass\w*\s+(?:Union[safety, security]|guardrails)\b", re.IGNORECASE
            ),
            re.compile(r"\bcircumvent\w*\s+(?:Union[restrictions, controls])\b", re.IGNORECASE),
            re.compile(r"\bbreak\w*\s+(?:Union[out, free])\b", re.IGNORECASE),
        ]

        return patterns

    def _build_metadata_patterns(self) -> dict[str, re.Pattern]:
        """Build regex patterns for metadata extraction.

        Returns:
            Dictionary of metadata patterns
        """
        patterns = {
            # LLM model detection
            "gpt4": re.compile(r"\bgpt-?4\b", re.IGNORECASE),
            "gpt35": re.compile(r"\bgpt-?3\.5\b", re.IGNORECASE),
            "claude": re.compile(r"\bclaude\b", re.IGNORECASE),
            "palm": re.compile(r"\bpalm\b", re.IGNORECASE),
            "llama": re.compile(r"\bllama\b", re.IGNORECASE),
            "bert": re.compile(r"\bbert\b", re.IGNORECASE),
            # Game type detection
            "matrix_game": re.compile(r"\bmatrix\s+(?:Union[game, wargame])\b", re.IGNORECASE),
            "seminar_game": re.compile(
                r"\bseminar\s+(?:Union[game, wargame])\b", re.IGNORECASE
            ),
            "digital_game": re.compile(
                r"\bdigital\s+(?:Union[game, wargame]|simulation)\b", re.IGNORECASE
            ),
            # Evaluation metrics
            "win_rate": re.compile(r"\bwin[\s-]?rate\b", re.IGNORECASE),
            "accuracy": re.compile(r"\baccuracy\b", re.IGNORECASE),
            "f1_score": re.compile(r"\bf1[\s-]?score\b", re.IGNORECASE),
            "human_evaluation": re.compile(
                r"\bhuman\s+(?:Union[evaluation, assessment]|rating)\b", re.IGNORECASE
            ),
            # Code availability
            "github": re.compile(r"github\.com/[\w-]+/[\w-]+", re.IGNORECASE),
            "code_available": re.compile(
                r"\bcode\s+(?:is\s+)?available\b", re.IGNORECASE
            ),
            "open_source": re.compile(r"\bopen[\s-]?source\b", re.IGNORECASE),
        }

        return patterns

    def _prepare_search_text(self, paper: pd.Series) -> str:
        """Prepare text for pattern searching.

        Args:
            paper: Paper data

        Returns:
            Combined text to search
        """
        # Combine title, abstract, and any extracted text
        parts = []

        if paper.get("title"):
            parts.append(str(paper["title"]))

        if paper.get("abstract"):
            parts.append(str(paper["abstract"]))

        # If we have eval_metrics from LLM extraction, include it
        if paper.get("eval_metrics"):
            parts.append(str(paper["eval_metrics"]))

        return " ".join(parts)

    def _extract_failure_modes(self, text: str) -> set[str]:
        """Extract failure modes from text using patterns.

        Args:
            text: Text to search

        Returns:
            Set of detected failure modes
        """
        detected_modes = set()

        for category, patterns in self.failure_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    # Map to controlled vocabulary
                    vocab_term = self._map_to_vocab(category)
                    if vocab_term:
                        detected_modes.add(vocab_term)

                        # Track statistics
                        pattern_key = f"{category}:{pattern.pattern}"
                        self.stats["pattern_matches"][pattern_key] = (
                            self.stats["pattern_matches"].get(pattern_key, 0) + 1
                        )
                    break  # One match per category is enough

        return detected_modes

    def _map_to_vocab(self, category: str) -> Optional[str]:
        """Map category to controlled vocabulary term.

        Args:
            category: Category name

        Returns:
            Vocabulary term or None
        """
        # Check all vocabulary categories
        for _vocab_cat, terms in self.failure_vocab.items():
            if category in terms:
                return category

        # Handle special mappings
        if category == "factual_error" or category == "inconsistency":
            return "hallucination"  # Group under hallucination
        elif category == "jailbreak":
            return "prompt_sensitivity"  # Group under prompt sensitivity

        # Default: return if it's in any vocab list
        all_terms = []
        for terms in self.failure_vocab.values():
            all_terms.extend(terms)

        if category in all_terms:
            return category

        return "other"

    def _tag_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tag papers with additional metadata.

        Args:
            df: DataFrame to tag

        Returns:
            Tagged DataFrame
        """
        # Add columns for detected metadata
        for key in [
            "llm_detected",
            "game_type_detected",
            "metrics_detected",
            "code_detected",
        ]:
            if key not in df.columns:
                df[key] = ""

        for idx, paper in df.iterrows():
            text = self._prepare_search_text(paper)

            # Detect LLM models
            llm_models = []
            for model, pattern in self.metadata_patterns.items():
                if (
                    model.startswith("gpt")
                    or model
                    in [
                        "claude",
                        "palm",
                        "llama",
                        "bert",
                    ]
                ) and pattern.search(text):
                    llm_models.append(model)

            if llm_models:
                df.at[idx, "llm_detected"] = "|".join(llm_models)

            # Detect game types
            game_types = []
            for game_type in ["matrix_game", "seminar_game", "digital_game"]:
                if self.metadata_patterns[game_type].search(text):
                    game_types.append(game_type.replace("_game", ""))

            if game_types:
                df.at[idx, "game_type_detected"] = "|".join(game_types)

            # Detect evaluation metrics
            metrics = []
            for metric in ["win_rate", "accuracy", "f1_score", "human_evaluation"]:
                if self.metadata_patterns[metric].search(text):
                    metrics.append(metric)

            if metrics:
                df.at[idx, "metrics_detected"] = "|".join(metrics)

            # Detect code availability
            github_match = self.metadata_patterns["github"].search(text)
            if github_match:
                df.at[idx, "code_detected"] = github_match.group(0)
            elif self.metadata_patterns["code_available"].search(
                text
            ) or self.metadata_patterns["open_source"].search(text):
                df.at[idx, "code_detected"] = "mentioned"

        return df

    def _log_statistics(self):
        """Log tagging statistics."""
        logger.info("Tagging statistics:")
        logger.info(f"  Total papers processed: {self.stats['total_processed']}")
        logger.info(
            f"  Papers with failure modes: {self.stats['papers_with_failures']}"
        )
        logger.info(f"  Total failure tags applied: {self.stats['total_failure_tags']}")

        # Log top pattern matches
        if self.stats["pattern_matches"]:
            logger.info("  Top pattern matches:")
            sorted_patterns = sorted(
                self.stats["pattern_matches"].items(), key=lambda x: x[1], reverse=True
            )[:10]

            for pattern, count in sorted_patterns:
                logger.info(f"    {pattern}: {count}")

    def get_failure_mode_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get summary of failure modes across papers.

        Args:
            df: DataFrame with tagged papers

        Returns:
            Summary DataFrame
        """
        # Collect all failure modes
        all_modes = []

        for modes_str in df["failure_modes"].dropna():
            if modes_str:
                modes = modes_str.split("|")
                all_modes.extend(modes)

        # Count occurrences
        from collections import Counter

        mode_counts = Counter(all_modes)

        # Create summary DataFrame
        summary = pd.DataFrame(
            list(mode_counts.items()), columns=["failure_mode", "count"]
        ).sort_values("count", ascending=False)

        # Add percentage
        total_papers = len(
            df[df["failure_modes"].notna() & (df["failure_modes"] != "")]
        )
        summary["percentage"] = (summary["count"] / total_papers * 100).round(1)

        return summary
