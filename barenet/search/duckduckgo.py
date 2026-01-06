"""
DuckDuckGo HTML search engine for BareNet.

This module implements a passive, ethical HTML-based search engine
that queries DuckDuckGo's public HTML interface to discover URLs
for research purposes.

Design Principles:
- Passive data collection only (no aggressive scraping)
- Single request per search (no pagination)
- No JavaScript execution or browser automation
- Clear identification via User-Agent
- Conservative rate limiting

Important Constraints:
- This is a discovery tool only, not a crawler
- Results are not validated or filtered here
- No API keys or authentication required
- Respects DuckDuckGo's Terms of Service

Note: HTML parsing is inherently fragile. This implementation
includes multiple fallback strategies and graceful degradation
to handle changes in DuckDuckGo's HTML structure.
"""

import time
import urllib.parse
from typing import List, Optional
import logging
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Tag

from barenet.search.base import SearchEngine, SearchResult, SearchEngineError
from barenet.httpclient import BareNetHttpClient

# Configure module-level logger
logger = logging.getLogger(__name__)


class DuckDuckGoHtmlEngine(SearchEngine):
    """
    HTML-based search engine for DuckDuckGo.
    
    This engine performs single-page searches against DuckDuckGo's
    HTML interface and extracts URLs, titles, and snippets from
    the search results page.
    
    Usage Ethics:
    - Only public, non-authenticated searches
    - Conservative rate limiting (one search per few seconds)
    - No attempts to bypass rate limiting or CAPTCHAs
    - No collection of personal data or PII
    """
    
    # Constants for HTML parsing
    BASE_URL = "https://duckduckgo.com/html/"
    SEARCH_DELAY = 3.0  # seconds between searches (conservative)
    DEFAULT_MAX_RESULTS = 20
    
    # CSS selectors for DuckDuckGo's HTML structure
    # Multiple selectors for robustness against changes
    RESULT_SELECTORS = [
        'div.result.results_links',
        'div.web-result',
        'div.result',
        '.result__body'
    ]
    
    TITLE_SELECTORS = [
        'a.result__url',
        'a.result__title',
        '.result__title a',
        '.web-result-title a'
    ]
    
    SNIPPET_SELECTORS = [
        'a.result__snippet',
        '.result__snippet',
        '.web-result-snippet'
    ]
    
    def __init__(self, http_client: Optional[BareNetHttpClient] = None) -> None:
        """
        Initialize the DuckDuckGo search engine.
        
        Args:
            http_client: Optional shared HTTP client instance.
                        If None, a new instance will be created.
        
        Raises:
            SearchEngineError: If initialization fails
        """
        super().__init__()
        
        try:
            self.http_client = http_client or BareNetHttpClient(
                rate_limit=0.5,  # More conservative for search
                cache_ttl=3600,  # Cache search results for 1 hour
                timeout=15.0,    # Slightly longer timeout for search
            )
            
            # Track last search time for rate limiting
            self._last_search_time = 0.0
            
            self._logger.info(f"Initialized DuckDuckGo search engine")
            
        except Exception as e:
            raise SearchEngineError(f"Failed to initialize DuckDuckGo engine: {e}")
    
    @property
    def name(self) -> str:
        """Return the engine's unique identifier."""
        return "duckduckgo"
    
    def _enforce_search_delay(self) -> None:
        """Enforce a minimum delay between searches to be respectful."""
        current_time = time.time()
        time_since_last = current_time - self._last_search_time
        
        if time_since_last < self.SEARCH_DELAY:
            sleep_time = self.SEARCH_DELAY - time_since_last
            self._logger.debug(f"Enforcing search delay: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self._last_search_time = time.time()
    
    def _build_search_url(self, query: str) -> str:
        """
        Build a DuckDuckGo HTML search URL for the given query.
        
        Args:
            query: The search query string
            
        Returns:
            Complete DuckDuckGo HTML search URL
            
        Note:
            Uses the HTML interface (not JavaScript) and sets conservative
            parameters to minimize server load.
        """
        # Clean and encode the query
        cleaned_query = query.strip()
        
        # Build query parameters
        params = {
            'q': cleaned_query,
            'kl': 'us-en',  # Language/region
            'kz': '-1',     # No safe search bypass (ethical use only)
            'k1': '-1',     # No JavaScript
            'kd': '-1',     # No DuckDuckGo redirects
            'kp': '-2',     # Moderate safe search
        }
        
        # Build URL with parameters
        query_string = '&'.join(f"{k}={v}" for k, v in params.items())
        search_url = f"{self.BASE_URL}?{query_string}"
        
        self._logger.debug(f"Built search URL for query: '{cleaned_query}'")
        return search_url
    
    def _parse_search_results(self, html: str, max_results: int) -> List[SearchResult]:
        """
        Parse HTML search results from DuckDuckGo.
        
        Args:
            html: The HTML content of the search results page
            max_results: Maximum number of results to extract
            
        Returns:
            List of SearchResult objects
            
        Raises:
            SearchEngineError: If parsing fails or no results found
        """
        results: List[SearchResult] = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Try multiple selectors for robustness
            result_elements = None
            for selector in self.RESULT_SELECTORS:
                result_elements = soup.select(selector)
                if result_elements:
                    self._logger.debug(f"Found {len(result_elements)} results with selector: {selector}")
                    break
            
            if not result_elements:
                self._logger.warning("No search results found with any selector")
                # Check if we got a CAPTCHA or error page
                title_element = soup.find('title')
                if title_element and 'CAPTCHA' in title_element.get_text():
                    raise SearchEngineError(
                        "DuckDuckGo is showing a CAPTCHA. Please wait before trying again."
                    )
                return results  # Return empty list for no results
            
            for i, result_element in enumerate(result_elements[:max_results]):
                try:
                    result = self._extract_single_result(result_element, i + 1)
                    if result:
                        results.append(result)
                except Exception as e:
                    self._logger.warning(f"Failed to extract result {i+1}: {e}")
                    continue
            
            self._logger.info(f"Successfully extracted {len(results)} results")
            return results
            
        except Exception as e:
            raise SearchEngineError(f"Failed to parse search results: {e}")
    
    def _extract_single_result(self, result_element: Tag, rank: int) -> Optional[SearchResult]:
        """
        Extract a single search result from a result element.
        
        Args:
            result_element: BeautifulSoup Tag for a single result
            rank: The rank/position of this result
            
        Returns:
            SearchResult object or None if extraction fails
        """
        try:
            # Extract URL
            url = self._extract_url(result_element)
            if not url:
                self._logger.debug(f"No URL found for result at rank {rank}")
                return None
            
            # Extract title
            title = self._extract_title(result_element) or f"Result {rank}"
            
            # Extract snippet
            snippet = self._extract_snippet(result_element)
            
            # Create and return the result
            return SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                source=self.name,
                rank=rank
            )
            
        except Exception as e:
            self._logger.debug(f"Error extracting result at rank {rank}: {e}")
            return None
    
    def _extract_url(self, result_element: Tag) -> Optional[str]:
        """
        Extract the URL from a result element.
        
        Args:
            result_element: BeautifulSoup Tag for a single result
            
        Returns:
            URL string or None if not found
        
        Note:
            DuckDuckGo often uses redirect URLs. We try to extract
            the actual destination URL when possible.
        """
        # Try multiple title selectors
        for selector in self.TITLE_SELECTORS:
            title_link = result_element.select_one(selector)
            if title_link and title_link.get('href'):
                href = title_link.get('href', '').strip()
                
                # Handle DuckDuckGo redirects
                if href.startswith('//duckduckgo.com/l/?'):
                    # Extract the actual URL from redirect
                    parsed = urllib.parse.urlparse(href)
                    query_params = urllib.parse.parse_qs(parsed.query)
                    if 'uddg' in query_params:
                        # URL is double-encoded in some cases
                        encoded_url = query_params['uddg'][0]
                        decoded_url = urllib.parse.unquote(encoded_url)
                        
                        # Validate it's a proper URL
                        parsed_url = urlparse(decoded_url)
                        if parsed_url.scheme and parsed_url.netloc:
                            return decoded_url
                
                # If not a redirect, check if it's a full URL
                parsed_href = urlparse(href)
                if parsed_href.scheme and parsed_href.netloc:
                    return href
                elif href.startswith('/'):
                    # Relative URL, convert to absolute
                    return urljoin(self.BASE_URL, href)
        
        # Fallback: Look for any link in the result
        links = result_element.find_all('a', href=True)
        for link in links:
            href = link.get('href', '').strip()
            if href and (href.startswith('http://') or href.startswith('https://')):
                return href
        
        return None
    
    def _extract_title(self, result_element: Tag) -> Optional[str]:
        """
        Extract the title from a result element.
        
        Args:
            result_element: BeautifulSoup Tag for a single result
            
        Returns:
            Title string or None if not found
        """
        # Try multiple title selectors
        for selector in self.TITLE_SELECTORS:
            title_element = result_element.select_one(selector)
            if title_element:
                # Get text and clean it
                title_text = title_element.get_text(strip=True)
                if title_text:
                    # Remove domain info that sometimes appears in the title
                    title_text = title_text.split(' › ')[0]
                    return title_text[:500]  # Limit length
        
        # Fallback: Look for any h2 or h3 with links
        for tag_name in ['h2', 'h3']:
            heading = result_element.find(tag_name)
            if heading:
                text = heading.get_text(strip=True)
                if text:
                    return text[:500]
        
        return None
    
    def _extract_snippet(self, result_element: Tag) -> Optional[str]:
        """
        Extract the snippet/description from a result element.
        
        Args:
            result_element: BeautifulSoup Tag for a single result
            
        Returns:
            Snippet string or None if not found
        """
        # Try multiple snippet selectors
        for selector in self.SNIPPET_SELECTORS:
            snippet_element = result_element.select_one(selector)
            if snippet_element:
                snippet_text = snippet_element.get_text(strip=True)
                if snippet_text:
                    return snippet_text[:1000]  # Limit length
        
        # Fallback: Look for any paragraph with substantial text
        paragraphs = result_element.find_all('p')
        for p in paragraphs:
            text = p.get_text(strip=True)
            if text and len(text) > 20:  # Reasonable length for a snippet
                return text[:1000]
        
        return None
    
    def search(self, query: str, max_results: int = DEFAULT_MAX_RESULTS) -> List[SearchResult]:
        """
        Execute a search on DuckDuckGo and return discovered URLs.
        
        Args:
            query: The search query string
            max_results: Maximum number of results to return (1-50)
            
        Returns:
            List of SearchResult objects
            
        Raises:
            ValueError: If query is invalid or max_results out of bounds
            SearchEngineError: For network, parsing, or engine failures
        
        Note:
            This method performs exactly one HTTP request and does not
            follow pagination. Results are limited to the first page.
        """
        # Validate inputs using parent class method
        self.validate_search_params(query, max_results)
        
        # Additional validation specific to this engine
        if max_results > 50:
            self._logger.warning(f"max_results {max_results} exceeds recommended limit, capping at 50")
            max_results = 50
        
        # Enforce rate limiting between searches
        self._enforce_search_delay()
        
        self._logger.info(f"Searching DuckDuckGo for: '{query}' (max_results: {max_results})")
        
        try:
            # Build search URL
            search_url = self._build_search_url(query)
            
            # Fetch the search results page HTML content
            self._logger.debug(f"Fetching search results from: {search_url}")
            html_content = self.http_client.fetch_html(search_url)
            
            # Parse results from HTML
            results = self._parse_search_results(html_content, max_results)
            
            self._logger.info(f"Search completed: found {len(results)} results for '{query}'")
            return results
            
        except SearchEngineError:
            raise  # Re-raise our own errors
        except Exception as e:
            raise SearchEngineError(f"DuckDuckGo search failed: {e}")
    
    def close(self) -> None:
        """Close the HTTP client and release resources."""
        if hasattr(self, 'http_client') and self.http_client:
            self.http_client.close()
            self._logger.debug("HTTP client closed")
    
    def __enter__(self) -> 'DuckDuckGoHtmlEngine':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensure cleanup."""
        self.close()
    
    def __del__(self) -> None:
        """Destructor - ensure cleanup."""
        self.close()


# Convenience function for single search
def search_duckduckgo(query: str, max_results: int = 20) -> List[SearchResult]:
    """
    Convenience function to search DuckDuckGo.
    
    Args:
        query: Search query string
        max_results: Maximum number of results
        
    Returns:
        List of search results
    """
    with DuckDuckGoHtmlEngine() as engine:
        return engine.search(query, max_results)
