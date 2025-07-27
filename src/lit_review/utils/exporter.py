"""Exporter module for packaging and exporting results."""

import json
import logging
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Exporter:
    """Exports results in various formats and to external services."""

    def __init__(self, config):
        """Initialize exporter with configuration.

        Args:
            config: Configuration object
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.compression = config.export_compression
        self.include_pdfs = config.export_include_pdfs
        self.include_logs = config.export_include_logs

        # Zenodo settings
        self.zenodo_enabled = config.zenodo_enabled
        self.zenodo_token = config.zenodo_token
        self.zenodo_community = getattr(config, "zenodo_community", "llm-wargames")

    def export_full_package(
        self,
        extraction_df: pd.DataFrame,
        figures: list[Path],
        summary: dict[str, Any],
        output_name: str | None = None,
        excluded_df: pd.DataFrame | None = None,
        disambiguation_report: dict[str, Any] | None = None,
    ) -> Path:
        """Create a complete export package.

        Args:
            extraction_df: DataFrame with extracted data
            figures: List of figure paths
            summary: Summary statistics dictionary
            output_name: Name for the output file (without extension)
            excluded_df: DataFrame with excluded papers (optional)
            disambiguation_report: Disambiguation statistics report (optional)

        Returns:
            Path to the created package
        """
        if output_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"lit_review_export_{timestamp}"

        logger.info(f"Creating export package: {output_name}")

        # Create temporary directory for organizing files
        temp_dir = self.output_dir / f"{output_name}_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Save main data files
            self._save_data_files(extraction_df, temp_dir)

            # 2. Copy figures
            self._copy_figures(figures, temp_dir)

            # 3. Save summary report
            self._save_summary(summary, temp_dir)

            # 4. Create metadata file
            self._create_metadata(extraction_df, temp_dir)

            # 5. Optionally include PDFs
            if self.include_pdfs:
                self._copy_pdfs(extraction_df, temp_dir)

            # 6. Optionally include logs
            if self.include_logs:
                self._copy_logs(temp_dir)

            # 7. Save excluded papers if provided
            if excluded_df is not None:
                self._save_excluded_papers(excluded_df, temp_dir)

            # 8. Save disambiguation report if provided
            if disambiguation_report is not None:
                self._save_disambiguation_report(disambiguation_report, temp_dir)

            # 9. Create README
            self._create_readme(
                extraction_df, temp_dir, excluded_df, disambiguation_report
            )

            # 10. Create archive
            if self.compression == "zip":
                archive_path = self._create_zip_archive(temp_dir, output_name)
            else:
                # Could add support for tar.gz, etc.
                archive_path = self._create_zip_archive(temp_dir, output_name)

            # 11. Clean up temp directory
            import shutil

            shutil.rmtree(temp_dir)

            logger.info(f"Export package created: {archive_path}")

            # 12. Optionally upload to Zenodo
            if self.zenodo_enabled and self.zenodo_token:
                self._upload_to_zenodo(archive_path, extraction_df)

            return archive_path

        except Exception as e:
            logger.error(f"Error creating export package: {e}")
            # Clean up on error
            if temp_dir.exists():
                import shutil

                shutil.rmtree(temp_dir)
            raise

    def _save_data_files(self, df: pd.DataFrame, output_dir: Path):
        """Save data files in multiple formats.

        Args:
            df: DataFrame to save
            output_dir: Directory to save files
        """
        data_dir = output_dir / "data"
        data_dir.mkdir(exist_ok=True)

        # Save as CSV
        csv_path = data_dir / "extraction_results.csv"
        df.to_csv(csv_path, index=False)
        logger.debug(f"Saved CSV: {csv_path}")

        # Save as JSON
        json_path = data_dir / "extraction_results.json"
        df.to_json(json_path, orient="records", indent=2)
        logger.debug(f"Saved JSON: {json_path}")

        # Save as Excel with formatting
        excel_path = data_dir / "extraction_results.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Results", index=False)

            # Add summary sheet
            summary_data = {
                "Metric": [
                    "Total Papers",
                    "Papers with PDFs",
                    "Papers Extracted",
                    "Papers with Failure Modes",
                    "Unique LLM Families",
                ],
                "Value": [
                    len(df),
                    (
                        len(
                            df[
                                df["pdf_status"].notna()
                                & (
                                    df["pdf_status"].str.startswith("downloaded")
                                    | df["pdf_status"].str.startswith("cached")
                                )
                            ]
                        )
                        if "pdf_status" in df.columns
                        else 0
                    ),
                    (
                        len(df[df["extraction_status"] == "success"])
                        if "extraction_status" in df.columns
                        else 0
                    ),
                    (
                        len(
                            df[
                                df["failure_modes"].notna()
                                & (df["failure_modes"] != "")
                            ]
                        )
                        if "failure_modes" in df.columns
                        else 0
                    ),
                    df["llm_family"].nunique() if "llm_family" in df.columns else 0,
                ],
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

        logger.debug(f"Saved Excel: {excel_path}")

    def _copy_figures(self, figures: list[Path], output_dir: Path):
        """Copy figures to export directory.

        Args:
            figures: List of figure paths
            output_dir: Output directory
        """
        if not figures:
            return

        figures_dir = output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        for fig_path in figures:
            if fig_path.exists():
                import shutil

                dest_path = figures_dir / fig_path.name
                shutil.copy2(fig_path, dest_path)
                logger.debug(f"Copied figure: {fig_path.name}")

    def _save_summary(self, summary: dict[str, Any], output_dir: Path):
        """Save summary report.

        Args:
            summary: Summary dictionary
            output_dir: Output directory
        """
        summary_path = output_dir / "summary_report.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.debug(f"Saved summary: {summary_path}")

    def _create_metadata(self, df: pd.DataFrame, output_dir: Path):
        """Create metadata file for the export.

        Args:
            df: DataFrame with data
            output_dir: Output directory
        """
        metadata = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "pipeline_version": "1.0.0",
                "total_papers": len(df),
                "config_used": {
                    "search_years": list(self.config.search_years),
                    "llm_model": self.config.llm_model,
                    "inclusion_flags": self.config.inclusion_flags,
                },
            },
            "data_summary": {
                "columns": list(df.columns),
                "sources": (
                    df["source_db"].value_counts().to_dict()
                    if "source_db" in df.columns
                    else {}
                ),
                "year_range": {
                    "min": int(df["year"].min()) if "year" in df.columns else None,
                    "max": int(df["year"].max()) if "year" in df.columns else None,
                },
            },
            "quality_metrics": {
                "papers_with_abstracts": len(
                    df[df["abstract"].notna() & (df["abstract"] != "")]
                ),
                "papers_with_dois": len(df[df["doi"].notna() & (df["doi"] != "")]),
                "extraction_success_rate": (
                    len(df[df["extraction_status"] == "success"]) / len(df)
                    if "extraction_status" in df.columns
                    else 0
                ),
            },
        }

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.debug(f"Created metadata: {metadata_path}")

    def _copy_pdfs(self, df: pd.DataFrame, output_dir: Path):
        """Copy PDFs to export directory.

        Args:
            df: DataFrame with PDF paths
            output_dir: Output directory
        """
        if "pdf_path" not in df.columns:
            return

        pdfs_dir = output_dir / "pdfs"
        pdfs_dir.mkdir(exist_ok=True)

        pdf_count = 0
        for pdf_path_str in tqdm(df["pdf_path"].dropna(), desc="Copying PDFs"):
            if pdf_path_str:
                pdf_path = Path(pdf_path_str)
                if pdf_path.exists():
                    import shutil

                    dest_path = pdfs_dir / pdf_path.name
                    shutil.copy2(pdf_path, dest_path)
                    pdf_count += 1

        logger.info(f"Copied {pdf_count} PDFs to export")

    def _copy_logs(self, output_dir: Path):
        """Copy log files to export.

        Args:
            output_dir: Output directory
        """
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        # Copy SQLite database if exists
        db_path = self.config.logging_db_path
        if db_path.exists():
            import shutil

            dest_path = logs_dir / db_path.name
            shutil.copy2(db_path, dest_path)
            logger.debug(f"Copied log database: {db_path.name}")

        # Copy any text logs
        log_dir = Path(self.config.log_dir)
        if log_dir.exists():
            for log_file in log_dir.glob("*.log"):
                import shutil

                dest_path = logs_dir / log_file.name
                shutil.copy2(log_file, dest_path)

    def _save_excluded_papers(self, excluded_df: pd.DataFrame, output_dir: Path):
        """Save excluded papers data.

        Args:
            excluded_df: DataFrame with excluded papers
            output_dir: Output directory
        """
        excluded_dir = output_dir / "excluded_papers"
        excluded_dir.mkdir(exist_ok=True)

        # Save as CSV
        csv_path = excluded_dir / "excluded_papers.csv"
        excluded_df.to_csv(csv_path, index=False)
        logger.debug(f"Saved excluded papers CSV: {csv_path}")

        # Save summary statistics
        summary_data = {
            "total_excluded": len(excluded_df),
            "exclusion_reasons": (
                excluded_df["disambiguation_reason"].value_counts().to_dict()
                if "disambiguation_reason" in excluded_df.columns
                else {}
            ),
            "by_year": (
                excluded_df["year"].value_counts().sort_index().to_dict()
                if "year" in excluded_df.columns
                else {}
            ),
            "by_source": (
                excluded_df["source_db"].value_counts().to_dict()
                if "source_db" in excluded_df.columns
                else {}
            ),
        }

        summary_path = excluded_dir / "excluded_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)
        logger.debug(f"Saved excluded papers summary: {summary_path}")

    def _save_disambiguation_report(self, report: dict[str, Any], output_dir: Path):
        """Save disambiguation report.

        Args:
            report: Disambiguation report dictionary
            output_dir: Output directory
        """
        report_path = output_dir / "disambiguation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.debug(f"Saved disambiguation report: {report_path}")

    def _create_readme(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        excluded_df: pd.DataFrame | None = None,
        disambiguation_report: dict[str, Any] | None = None,
    ):
        """Create README file for the export.

        Args:
            df: DataFrame with data
            output_dir: Output directory
            excluded_df: DataFrame with excluded papers (optional)
            disambiguation_report: Disambiguation statistics report (optional)
        """
        readme_content = f"""# Literature Review Export

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Contents

- `data/`: Extracted data in CSV, JSON, and Excel formats
- `figures/`: Visualization plots
- `summary_report.json`: Summary statistics
- `metadata.json`: Export metadata and configuration
"""

        if excluded_df is not None:
            readme_content += (
                "- `excluded_papers/`: Papers excluded by disambiguation filters\n"
            )

        if disambiguation_report is not None:
            readme_content += (
                "- `disambiguation_report.json`: Detailed disambiguation statistics\n"
            )

        if self.include_pdfs:
            readme_content += "- `pdfs/`: PDF files of papers\n"

        if self.include_logs:
            readme_content += "- `logs/`: Processing logs and database\n"

        readme_content += f"""
## Summary Statistics

- Total papers: {len(df)}
- Date range: {df["year"].min()}-{df["year"].max()}
- Sources: {", ".join(df["source_db"].unique()) if "source_db" in df.columns else "N/A"}"""

        if excluded_df is not None:
            readme_content += f"""
- Excluded papers: {len(excluded_df)}
- Exclusion rate: {len(excluded_df) / (len(df) + len(excluded_df)) * 100:.1f}%"""

        if disambiguation_report is not None:
            readme_content += f"""
- Grey literature papers: {disambiguation_report.get('grey_literature', {}).get('count', 0)}"""

        readme_content += """

## Data Dictionary

### Main Fields
- `title`: Paper title
- `authors`: Semi-colon separated author names
- `year`: Publication year
- `abstract`: Paper abstract
- `doi`: Digital Object Identifier
- `url`: Paper URL
- `source_db`: Database source

### Extracted Fields
- `venue_type`: Type of publication venue
- `game_type`: Type of wargame
- `open_ended`: Whether game allows open-ended moves (yes/no)
- `quantitative`: Whether game tracks numeric scores (yes/no)
- `llm_family`: LLM model family used
- `llm_role`: Role of LLM (player/generator/analyst)
- `eval_metrics`: Evaluation metrics used
- `failure_modes`: Detected failure modes (pipe-separated)
- `awscale`: AWScale rating (1-5)
- `code_release`: GitHub URL or "none"

## Citation

If you use this data, please cite:
[Add citation information here]
"""

        readme_path = output_dir / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme_content)
        logger.debug("Created README file")

    def _create_zip_archive(self, source_dir: Path, output_name: str) -> Path:
        """Create ZIP archive of directory.

        Args:
            source_dir: Directory to archive
            output_name: Name for output file

        Returns:
            Path to created archive
        """
        archive_path = self.output_dir / f"{output_name}.zip"

        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in tqdm(list(source_dir.rglob("*")), desc="Creating archive"):
                if file_path.is_file():
                    arcname = file_path.relative_to(source_dir.parent)
                    zipf.write(file_path, arcname)

        # Get size
        size_mb = archive_path.stat().st_size / 1024 / 1024
        logger.info(f"Created archive: {archive_path} ({size_mb:.1f} MB)")

        return archive_path

    def _upload_to_zenodo(self, archive_path: Path, df: pd.DataFrame):
        """Upload archive to Zenodo.

        Args:
            archive_path: Path to archive file
            df: DataFrame with data (for metadata)
        """
        logger.info("Uploading to Zenodo...")

        try:
            # Create deposition
            headers = {"Authorization": f"Bearer {self.zenodo_token}"}

            # Check if this is sandbox or production
            base_url = (
                "https://zenodo.org/api"
                if "sandbox" not in str(self.zenodo_token)
                else "https://sandbox.zenodo.org/api"
            )

            # Create new deposition
            r = requests.post(
                f"{base_url}/deposit/depositions", headers=headers, json={}
            )

            if r.status_code != 201:
                logger.error(
                    f"Failed to create Zenodo deposition: {r.status_code} {r.text}"
                )
                return

            deposition = r.json()
            deposition_id = deposition["id"]
            bucket_url = deposition["links"]["bucket"]

            logger.info(f"Created Zenodo deposition: {deposition_id}")

            # Upload file
            with open(archive_path, "rb") as f:
                r = requests.put(
                    f"{bucket_url}/{archive_path.name}", data=f, headers=headers
                )

            if r.status_code != 200:
                logger.error(f"Failed to upload file to Zenodo: {r.status_code}")
                return

            # Add metadata
            metadata = {
                "metadata": {
                    "title": f"LLM Wargaming Literature Review Export - {datetime.now().strftime('%Y-%m-%d')}",
                    "upload_type": "dataset",
                    "description": f"Systematic literature review data on LLM-powered wargames. "
                    f"Contains {len(df)} papers from {df['year'].min()}-{df['year'].max()}.",
                    "creators": [{"name": "Literature Review Pipeline"}],
                    "keywords": [
                        "LLM",
                        "wargaming",
                        "systematic review",
                        "natural language processing",
                    ],
                    "communities": (
                        [{"identifier": self.zenodo_community}]
                        if self.zenodo_community
                        else []
                    ),
                    "language": "eng",
                    "license": "cc-by-4.0",
                }
            }

            r = requests.put(
                f"{base_url}/deposit/depositions/{deposition_id}",
                headers=headers,
                json=metadata,
            )

            if r.status_code == 200:
                logger.info(f"Zenodo upload complete. Deposition ID: {deposition_id}")
                logger.info(
                    "Note: Deposition is in draft state. Publish manually via Zenodo website."
                )
            else:
                logger.error(f"Failed to add metadata to Zenodo: {r.status_code}")

        except Exception as e:
            logger.error(f"Zenodo upload error: {e}")

    def export_bibtex(self, df: pd.DataFrame, output_path: Path | None = None) -> Path:
        """Export results as BibTeX file.

        Args:
            df: DataFrame with paper data
            output_path: Path for output file

        Returns:
            Path to created BibTeX file
        """
        if output_path is None:
            output_path = self.output_dir / "papers.bib"

        entries = []

        for _idx, paper in df.iterrows():
            # Determine entry type
            venue_type = paper.get("venue_type", "").lower()
            if "conference" in venue_type or "workshop" in venue_type:
                entry_type = "inproceedings"
            elif "journal" in venue_type:
                entry_type = "article"
            elif "tech" in venue_type and "report" in venue_type:
                entry_type = "techreport"
            else:
                entry_type = "misc"

            # Create citation key
            first_author = (
                paper["authors"].split(";")[0].split()[-1]
                if paper.get("authors")
                else "Unknown"
            )
            year = paper.get("year", 0)
            title_words = paper.get("title", "").split()[:2]
            cite_key = f"{first_author}{year}{''.join(title_words)}"
            cite_key = "".join(c for c in cite_key if c.isalnum())

            # Build entry
            entry = f"@{entry_type}{{{cite_key},\n"
            entry += f"  title = {{{paper.get('title', 'Unknown')}}},\n"

            # Authors
            if paper.get("authors"):
                authors = " and ".join(paper["authors"].split(";"))
                entry += f"  author = {{{authors}}},\n"

            # Year
            if year > 0:
                entry += f"  year = {{{year}}},\n"

            # Venue
            if paper.get("venue"):
                if entry_type == "article":
                    entry += f"  journal = {{{paper['venue']}}},\n"
                elif entry_type == "inproceedings":
                    entry += f"  booktitle = {{{paper['venue']}}},\n"

            # DOI
            if paper.get("doi"):
                entry += f"  doi = {{{paper['doi']}}},\n"

            # URL
            if paper.get("url"):
                entry += f"  url = {{{paper['url']}}},\n"

            # Abstract
            if paper.get("abstract"):
                abstract = (
                    paper["abstract"]
                    .replace("\n", " ")
                    .replace("{", "\\{")
                    .replace("}", "\\}")
                )
                entry += f"  abstract = {{{abstract}}},\n"

            entry = entry.rstrip(",\n") + "\n}\n"
            entries.append(entry)

        # Write BibTeX file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(entries))

        logger.info(f"Exported {len(entries)} BibTeX entries to {output_path}")
        return output_path
