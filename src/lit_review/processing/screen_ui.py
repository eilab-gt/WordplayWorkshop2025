"""Screening UI generator for manual paper review."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class ScreenUI:
    """Generates screening sheets for manual paper review."""

    def __init__(self, config):
        """Initialize screening UI generator.

        Args:
            config: Configuration object
        """
        self.config = config
        self.output_path = config.screening_progress_path

    def prepare_screening_sheet(
        self,
        df: pd.DataFrame,
        output_path: Path | None = None,
        include_asreview: bool = False,
    ) -> pd.DataFrame:
        """Prepare a screening sheet from paper data.

        Args:
            df: DataFrame with paper information
            output_path: Path to save screening sheet (uses config default if None)
            include_asreview: Whether to prepare for ASReview integration

        Returns:
            DataFrame with screening columns added
        """
        logger.info(f"Preparing screening sheet for {len(df)} papers")

        # Copy DataFrame to avoid modifying original
        screening_df = df.copy()

        # Add screening columns
        screening_df = self._add_screening_columns(screening_df)

        # Sort for easier screening
        screening_df = self._sort_for_screening(screening_df)

        # Add metadata columns
        screening_df = self._add_metadata(screening_df)

        # Save to file
        if output_path is None:
            output_path = self.output_path

        self._save_screening_sheet(screening_df, output_path)

        # Optionally prepare ASReview format
        if include_asreview:
            self._prepare_asreview_format(screening_df, output_path)

        return screening_df

    def _add_screening_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add columns for screening decisions.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with screening columns
        """
        # Title/Abstract screening columns
        df["include_ta"] = ""  # yes/no/maybe
        df["reason_ta"] = ""  # Exclusion reason
        df["notes_ta"] = ""  # Additional notes

        # Full-text screening columns
        df["include_ft"] = ""  # yes/no
        df["reason_ft"] = ""  # Exclusion reason
        df["notes_ft"] = ""  # Additional notes

        # Screening metadata
        df["screener_ta"] = ""  # Who screened title/abstract
        df["screened_ta_date"] = ""  # When screened
        df["screener_ft"] = ""  # Who screened full text
        df["screened_ft_date"] = ""  # When screened

        # Quality/relevance scores
        df["relevance_score"] = ""  # 1-5 scale
        df["quality_score"] = ""  # 1-5 scale

        return df

    def _sort_for_screening(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort papers for efficient screening.

        Args:
            df: Input DataFrame

        Returns:
            Sorted DataFrame
        """
        # Create a composite score for sorting
        df["sort_score"] = 0

        # Boost papers with certain keywords in title/abstract
        priority_keywords = [
            "wargame",
            "war game",
            "crisis simulation",
            "llm",
            "large language model",
            "gpt",
            "claude",
        ]

        for keyword in priority_keywords:
            # Check title
            df.loc[
                df["title"].str.lower().str.contains(keyword, na=False), "sort_score"
            ] += 2
            # Check abstract
            df.loc[
                df["abstract"].str.lower().str.contains(keyword, na=False), "sort_score"
            ] += 1

        # Boost recent papers
        current_year = datetime.now().year
        df["year_score"] = df["year"].apply(lambda y: max(0, 5 - (current_year - y)))
        df["sort_score"] += df["year_score"]

        # Boost papers with more citations
        if "citations" in df.columns:
            df["citation_score"] = pd.qcut(
                df["citations"].fillna(0),
                q=5,
                labels=[1, 2, 3, 4, 5],
                duplicates="drop",
            )
            df["sort_score"] += df["citation_score"]

        # Sort by score (descending), then year (descending)
        df = df.sort_values(["sort_score", "year"], ascending=[False, False])

        # Remove temporary columns
        df = df.drop(["sort_score", "year_score"], axis=1)
        if "citation_score" in df.columns:
            df = df.drop("citation_score", axis=1)

        # Reset index
        df = df.reset_index(drop=True)

        return df

    def _add_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add metadata columns for tracking.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with metadata
        """
        # Add screening ID
        df["screening_id"] = [f"SCREEN_{i:04d}" for i in range(1, len(df) + 1)]

        # Add source tracking
        if "source_db" not in df.columns:
            df["source_db"] = "unknown"

        # Add duplicate check flag
        df["potential_duplicate"] = ""

        # Add language flag (for future ML detection)
        df["language_detected"] = "en"  # Default to English

        # Reorder columns for better viewing
        screening_cols = [
            "screening_id",
            "title",
            "authors",
            "year",
            "venue",
            "abstract",
            "doi",
            "url",
            "source_db",
            "include_ta",
            "reason_ta",
            "notes_ta",
            "include_ft",
            "reason_ft",
            "notes_ft",
            "relevance_score",
            "quality_score",
            "pdf_path",
            "pdf_status",
        ]

        # Only include columns that exist
        cols = [col for col in screening_cols if col in df.columns]
        # Add any remaining columns
        remaining_cols = [col for col in df.columns if col not in cols]
        cols.extend(remaining_cols)

        return df[cols]

    def _save_screening_sheet(self, df: pd.DataFrame, output_path: Path):
        """Save screening sheet to CSV.

        Args:
            df: Screening DataFrame
            output_path: Path to save file
        """
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Saved screening sheet to {output_path}")

        # Also create an Excel version for easier manual screening
        excel_path = output_path.with_suffix(".xlsx")
        self._save_excel_version(df, excel_path)

    def _save_excel_version(self, df: pd.DataFrame, excel_path: Path):
        """Save an Excel version with formatting.

        Args:
            df: Screening DataFrame
            excel_path: Path to save Excel file
        """
        try:
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                # Write main screening sheet
                df.to_excel(writer, sheet_name="Screening", index=False)

                # Get workbook and worksheet
                worksheet = writer.sheets["Screening"]

                # Auto-adjust column widths
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter

                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass

                    # Set reasonable limits
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width

                # Add instructions sheet
                instructions_df = self._create_instructions()
                instructions_df.to_excel(writer, sheet_name="Instructions", index=False)

                # Add exclusion reasons sheet
                reasons_df = self._create_exclusion_reasons()
                reasons_df.to_excel(writer, sheet_name="Exclusion_Reasons", index=False)

            logger.info(f"Saved Excel screening sheet to {excel_path}")

        except Exception as e:
            logger.warning(f"Could not create Excel version: {e}")

    def _create_instructions(self) -> pd.DataFrame:
        """Create instructions DataFrame.

        Returns:
            DataFrame with screening instructions
        """
        instructions = [
            {
                "Step": "Title/Abstract Screening",
                "Instructions": 'Review title and abstract. Mark include_ta as "yes", "no", or "maybe".',
                "Notes": 'If "no", provide reason in reason_ta column',
            },
            {
                "Step": "Inclusion Criteria",
                "Instructions": "Paper must: (1) Use LLM ≥100M params, (2) Involve wargaming/conflict simulation, (3) Have natural language interaction",
                "Notes": "See review protocol for detailed criteria",
            },
            {
                "Step": "Full-Text Screening",
                "Instructions": 'For papers marked "yes" or "maybe" in title/abstract, review full text',
                "Notes": 'Mark include_ft as "yes" or "no" with reason',
            },
            {
                "Step": "Quality Scoring",
                "Instructions": "Rate relevance (1-5) and quality (1-5) for included papers",
                "Notes": "5 = highly relevant/excellent quality",
            },
        ]

        return pd.DataFrame(instructions)

    def _create_exclusion_reasons(self) -> pd.DataFrame:
        """Create exclusion reasons reference.

        Returns:
            DataFrame with standard exclusion reasons
        """
        reasons = [
            {
                "Code": "E1",
                "Reason": "Not a wargame/conflict simulation",
                "Stage": "Title/Abstract",
            },
            {
                "Code": "E2",
                "Reason": "No LLM or LLM <100M params",
                "Stage": "Title/Abstract",
            },
            {
                "Code": "E3",
                "Reason": "No natural language interaction",
                "Stage": "Title/Abstract",
            },
            {
                "Code": "E4",
                "Reason": "Not empirical (opinion/editorial)",
                "Stage": "Title/Abstract",
            },
            {
                "Code": "E5",
                "Reason": "Wrong publication type",
                "Stage": "Title/Abstract",
            },
            {"Code": "E6", "Reason": "Full text not available", "Stage": "Full Text"},
            {
                "Code": "E7",
                "Reason": "Not in English (no translation)",
                "Stage": "Full Text",
            },
            {"Code": "E8", "Reason": "Duplicate publication", "Stage": "Any"},
        ]

        return pd.DataFrame(reasons)

    def _prepare_asreview_format(self, df: pd.DataFrame, base_path: Path):
        """Prepare data for ASReview active learning.

        Args:
            df: Screening DataFrame
            base_path: Base path for output
        """
        try:
            # ASReview expects specific column names
            asreview_df = pd.DataFrame(
                {
                    "title": df["title"],
                    "abstract": df["abstract"],
                    "authors": df["authors"],
                    "year": df["year"],
                    "doi": df["doi"].fillna(""),
                    "url": df["url"].fillna(""),
                    "included": df["include_ta"].apply(
                        lambda x: 1 if x == "yes" else 0 if x == "no" else -1
                    ),
                }
            )

            # Remove papers without labels for initial training
            labeled_df = asreview_df[asreview_df["included"] != -1]
            unlabeled_df = asreview_df[asreview_df["included"] == -1]

            # Save ASReview format
            asreview_path = base_path.parent / "asreview_data.csv"
            asreview_df.to_csv(asreview_path, index=False)

            logger.info(
                f"Prepared ASReview data: {len(labeled_df)} labeled, {len(unlabeled_df)} unlabeled"
            )

        except Exception as e:
            logger.warning(f"Could not prepare ASReview format: {e}")

    def load_screening_progress(self, path: Path | None = None) -> pd.DataFrame:
        """Load existing screening progress.

        Args:
            path: Path to screening file (uses config default if None)

        Returns:
            DataFrame with screening progress
        """
        if path is None:
            path = self.output_path

        if not path.exists():
            logger.warning(f"No screening file found at {path}")
            return pd.DataFrame()

        # Try Excel first, then CSV
        excel_path = path.with_suffix(".xlsx")
        if excel_path.exists():
            df = pd.read_excel(excel_path, sheet_name="Screening")
        else:
            df = pd.read_csv(path)

        logger.info(f"Loaded {len(df)} papers from screening sheet")

        return df

    def get_screening_statistics(self, df: pd.DataFrame) -> dict[str, Any]:
        """Calculate screening statistics.

        Args:
            df: Screening DataFrame

        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_papers": len(df),
            "title_abstract": {
                "screened": len(df[df["include_ta"] != ""]),
                "included": len(df[df["include_ta"] == "yes"]),
                "excluded": len(df[df["include_ta"] == "no"]),
                "maybe": len(df[df["include_ta"] == "maybe"]),
            },
            "full_text": {
                "screened": len(df[df["include_ft"] != ""]),
                "included": len(df[df["include_ft"] == "yes"]),
                "excluded": len(df[df["include_ft"] == "no"]),
            },
        }

        # Add exclusion reason breakdown
        if "reason_ta" in df.columns:
            ta_reasons = df[df["reason_ta"] != ""]["reason_ta"].value_counts().to_dict()
            stats["title_abstract"]["exclusion_reasons"] = ta_reasons

        if "reason_ft" in df.columns:
            ft_reasons = df[df["reason_ft"] != ""]["reason_ft"].value_counts().to_dict()
            stats["full_text"]["exclusion_reasons"] = ft_reasons

        return stats
