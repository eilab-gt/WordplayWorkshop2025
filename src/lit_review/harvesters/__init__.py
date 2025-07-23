"""Literature harvesters for various academic databases."""

from .arxiv_harvester import ArxivHarvester
from .base import BaseHarvester, Paper
from .crossref import CrossrefHarvester
from .google_scholar import GoogleScholarHarvester
from .search_harvester import SearchHarvester
from .semantic_scholar import SemanticScholarHarvester

__all__ = [
    "ArxivHarvester",
    "BaseHarvester",
    "CrossrefHarvester",
    "GoogleScholarHarvester",
    "Paper",
    "SearchHarvester",
    "SemanticScholarHarvester",
]
