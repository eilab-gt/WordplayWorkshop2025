"""Literature harvesters for various academic databases."""

from .base import BaseHarvester, Paper
from .google_scholar import GoogleScholarHarvester
from .arxiv_harvester import ArxivHarvester
from .semantic_scholar import SemanticScholarHarvester
from .crossref import CrossrefHarvester
from .search_harvester import SearchHarvester

__all__ = [
    'BaseHarvester',
    'Paper',
    'GoogleScholarHarvester',
    'ArxivHarvester', 
    'SemanticScholarHarvester',
    'CrossrefHarvester',
    'SearchHarvester'
]