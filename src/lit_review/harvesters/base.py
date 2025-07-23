"""Base harvester class for literature search."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging


logger = logging.getLogger(__name__)


@dataclass
class Paper:
    """Represents a paper from any source."""
    
    title: str
    authors: List[str]
    year: int
    abstract: str
    source_db: str
    url: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    venue: Optional[str] = None
    citations: Optional[int] = None
    pdf_url: Optional[str] = None
    keywords: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            'title': self.title,
            'authors': '; '.join(self.authors) if self.authors else '',
            'year': self.year,
            'abstract': self.abstract,
            'source_db': self.source_db,
            'url': self.url,
            'doi': self.doi,
            'arxiv_id': self.arxiv_id,
            'venue': self.venue,
            'citations': self.citations,
            'pdf_url': self.pdf_url,
            'keywords': '; '.join(self.keywords) if self.keywords else ''
        }


class BaseHarvester(ABC):
    """Abstract base class for all harvesters."""
    
    def __init__(self, config: Any):
        """Initialize harvester with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.source_name = self.__class__.__name__.replace('Harvester', '').lower()
        self.results: List[Paper] = []
        
    @abstractmethod
    def search(self, query: str, max_results: int = 100) -> List[Paper]:
        """Execute search and return papers.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to retrieve
            
        Returns:
            List of Paper objects
        """
        pass
    
    def build_query(self) -> str:
        """Build query string from configuration.
        
        Returns:
            Formatted query string
        """
        # Build query components
        wargame_part = f"({' OR '.join(f'"{term}"' for term in self.config.wargame_terms)})"
        llm_part = f"({' OR '.join(f'"{term}"' for term in self.config.llm_terms)})"
        action_part = f"({' OR '.join(f'"{term}"' for term in self.config.action_terms)})"
        
        # Build exclusions
        exclusions = ' '.join(f'NOT "{term}"' for term in self.config.exclusion_terms)
        
        # Combine
        query = f"{wargame_part} AND {llm_part} AND {action_part}"
        if exclusions:
            query = f"{query} {exclusions}"
            
        return query
    
    def filter_by_year(self, papers: List[Paper]) -> List[Paper]:
        """Filter papers by configured year range.
        
        Args:
            papers: List of papers to filter
            
        Returns:
            Filtered list of papers
        """
        start_year, end_year = self.config.search_years
        filtered = [
            p for p in papers 
            if p.year and start_year <= p.year <= end_year
        ]
        
        logger.info(
            f"{self.source_name}: Filtered {len(papers)} papers to {len(filtered)} "
            f"by year range {start_year}-{end_year}"
        )
        
        return filtered
    
    def clean_text(self, text: Optional[str]) -> str:
        """Clean and normalize text fields.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char.isspace())
        
        return text.strip()
    
    def extract_doi(self, text: str) -> Optional[str]:
        """Extract DOI from text if present.
        
        Args:
            text: Text that may contain a DOI
            
        Returns:
            DOI string or None
        """
        import re
        
        # DOI pattern
        doi_pattern = r'10\.\d{4,}/[-._;()/:\w]+'
        match = re.search(doi_pattern, text)
        
        return match.group(0) if match else None