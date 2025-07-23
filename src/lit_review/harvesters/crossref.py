"""Crossref harvester implementation."""

import time
import logging
from typing import List, Optional, Dict, Any
import requests
from urllib.parse import quote

from .base import BaseHarvester, Paper


logger = logging.getLogger(__name__)


class CrossrefHarvester(BaseHarvester):
    """Harvester for Crossref API."""
    
    BASE_URL = "https://api.crossref.org"
    
    def __init__(self, config):
        """Initialize Crossref harvester.
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        self.rate_limits = config.rate_limits.get('crossref', {})
        self.delay_milliseconds = self.rate_limits.get('delay_milliseconds', 20)
        self.email = config.unpaywall_email  # Crossref appreciates contact info
        
        # Set up headers
        self.headers = {
            'User-Agent': f'LitReviewPipeline/1.0 (mailto:{self.email})' if self.email else 'LitReviewPipeline/1.0'
        }
        
    def search(self, query: str, max_results: int = 100) -> List[Paper]:
        """Search Crossref for papers.
        
        Args:
            query: Search query string
            max_results: Maximum number of results
            
        Returns:
            List of Paper objects
        """
        papers = []
        
        try:
            logger.info(f"Crossref: Starting search with query: {query}")
            
            # Crossref allows up to 1000 results per request, but we'll use smaller batches
            batch_size = min(100, max_results)
            offset = 0
            
            while len(papers) < max_results:
                # Execute search
                results, total = self._search_batch(query, batch_size, offset)
                
                if not results:
                    break
                
                # Extract papers
                for result in results:
                    paper = self._extract_paper(result)
                    if paper:
                        papers.append(paper)
                        
                        if len(papers) >= max_results:
                            break
                
                # Check if more results available
                if offset + batch_size >= total:
                    break
                
                offset += batch_size
                
                # Rate limiting
                time.sleep(self.delay_milliseconds / 1000.0)
            
            logger.info(f"Crossref: Found {len(papers)} papers")
            
        except Exception as e:
            logger.error(f"Crossref: Error during search: {e}")
        
        # Filter by year
        papers = self.filter_by_year(papers)
        
        return papers
    
    def _search_batch(self, query: str, rows: int, offset: int) -> tuple[List[Dict[str, Any]], int]:
        """Search for a batch of papers.
        
        Args:
            query: Search query
            rows: Number of results to retrieve
            offset: Starting offset
            
        Returns:
            Tuple of (results list, total count)
        """
        try:
            # Build search URL
            url = f"{self.BASE_URL}/works"
            
            # Parameters
            params = {
                'query': query,
                'rows': rows,
                'offset': offset,
                'select': 'DOI,title,author,published-print,published-online,abstract,container-title,type,is-referenced-by-count,URL,link'
            }
            
            # Add year filter
            start_year, end_year = self.config.search_years
            params['filter'] = f'from-pub-date:{start_year},until-pub-date:{end_year}'
            
            # Make request
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            message = data.get('message', {})
            
            items = message.get('items', [])
            total = message.get('total-results', 0)
            
            return items, total
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Crossref: Request error: {e}")
            return [], 0
    
    def _extract_paper(self, result: Dict[str, Any]) -> Optional[Paper]:
        """Extract Paper object from Crossref result.
        
        Args:
            result: Crossref result dictionary
            
        Returns:
            Paper object or None if extraction fails
        """
        try:
            # Extract title
            title_list = result.get('title', [])
            title = title_list[0] if title_list else ''
            if not title:
                return None
            
            # Extract authors
            authors = []
            for author in result.get('author', []):
                given = author.get('given', '')
                family = author.get('family', '')
                if given and family:
                    authors.append(f"{given} {family}")
                elif family:
                    authors.append(family)
            
            # Extract year
            year = 0
            published = result.get('published-print') or result.get('published-online')
            if published:
                date_parts = published.get('date-parts', [[]])
                if date_parts and date_parts[0]:
                    year = date_parts[0][0] if date_parts[0] else 0
            
            # Get abstract
            abstract = result.get('abstract', '')
            
            # Clean HTML from abstract if present
            if abstract and '<' in abstract:
                from bs4 import BeautifulSoup
                abstract = BeautifulSoup(abstract, 'html.parser').get_text()
            
            # Create Paper object
            paper = Paper(
                title=self.clean_text(title),
                authors=authors,
                year=year,
                abstract=self.clean_text(abstract),
                source_db='crossref',
                url=result.get('URL', ''),
                doi=result.get('DOI'),
                venue=result.get('container-title', [''])[0] if result.get('container-title') else '',
                citations=result.get('is-referenced-by-count', 0)
            )
            
            # Try to get PDF URL from links
            for link in result.get('link', []):
                if link.get('content-type') == 'application/pdf':
                    paper.pdf_url = link.get('URL')
                    break
            
            # Add publication type as keyword
            pub_type = result.get('type', '')
            if pub_type:
                paper.keywords = [pub_type]
            
            return paper
            
        except Exception as e:
            logger.error(f"Crossref: Failed to extract paper: {e}")
            return None
    
    def get_paper_by_doi(self, doi: str) -> Optional[Paper]:
        """Get a specific paper by DOI.
        
        Args:
            doi: Digital Object Identifier
            
        Returns:
            Paper object or None if not found
        """
        try:
            url = f"{self.BASE_URL}/works/{doi}"
            
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 404:
                return None
                
            response.raise_for_status()
            data = response.json()
            
            result = data.get('message', {})
            return self._extract_paper(result)
            
        except Exception as e:
            logger.error(f"Crossref: Error fetching DOI {doi}: {e}")
            return None
    
    def validate_doi(self, doi: str) -> bool:
        """Check if a DOI exists in Crossref.
        
        Args:
            doi: Digital Object Identifier
            
        Returns:
            True if DOI exists, False otherwise
        """
        try:
            url = f"{self.BASE_URL}/works/{doi}"
            response = requests.head(url, headers=self.headers)
            return response.status_code == 200
            
        except Exception:
            return False