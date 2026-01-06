"""
Browser orchestration layer for BareNet.

This module provides the core orchestration logic for BareNet, coordinating
search engines, filters, and HTTP validation to discover web resources
in a passive, ethical manner.

Design Principles:
- Orchestration only: No CLI, no printing, no user interaction
- Deterministic execution: Same inputs produce same outputs
- Graceful degradation: Individual failures don't halt the pipeline
- Strong typing: Type hints for all public interfaces
- Passive behavior: No active scanning or exploitation
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set
from datetime import datetime

from barenet.search.base import SearchEngine, SearchResult, SearchEngineError
from barenet.filters.base import Filter, FilterPipeline, FilterResult
from barenet.httpclient import BareNetHttpClient, HttpClientError

# Configure module-level logger
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DiscoveredResource:
    """
    Immutable container for a discovered web resource.
    
    Represents a resource that has passed through the entire discovery
    and validation pipeline.
    
    Attributes:
        original_url: The URL as originally discovered by the search engine
        final_url: The URL after any redirects or normalization
        final_scheme: The scheme/protocol of the final URL (http, https, etc.)
        title: Human-readable title from the search result
        snippet: Brief description from the search result
        source: Name of the search engine that discovered this resource
        rank: Position in the search results (1-indexed)
        status_code: HTTP status code from validation, if available
        headers: HTTP response headers from validation, if available
        filters_passed: List of filter names that accepted this resource
        validated_at: Timestamp when the resource was validated (None if not validated)
        from_cache: Whether the validation result came from cache
    """
    original_url: str
    final_url: str
    final_scheme: str
    title: Optional[str] = None
    snippet: Optional[str] = None
    source: str = "unknown"
    rank: int = 0
    status_code: Optional[int] = None
    headers: Optional[Dict[str, str]] = field(default_factory=dict)
    filters_passed: List[str] = field(default_factory=list)
    validated_at: Optional[datetime] = None
    from_cache: bool = False
    
    def __post_init__(self) -> None:
        """Validate the discovered resource after initialization."""
        # Validate URLs are non-empty
        if not self.original_url or not self.final_url:
            raise ValueError("URLs cannot be empty")
        
        # Validate scheme is valid
        if self.final_scheme not in ("http", "https"):
            logger.warning(f"Unusual scheme for resource: {self.final_scheme}")
        
        # Validate rank is non-negative
        if self.rank < 0:
            raise ValueError(f"Rank must be non-negative, got: {self.rank}")


class BrowserError(Exception):
    """Base exception for browser orchestration failures."""
    pass


class Browser:
    """
    Core orchestration layer for BareNet discovery pipeline.
    
    Coordinates search engines, filters, and HTTP validation to discover
    web resources in a deterministic, passive manner.
    
    The browser is designed to be:
    - Stateless (except for injected dependencies)
    - Deterministic
    - Testable via dependency injection
    - Resilient to individual component failures
    
    Usage:
        browser = Browser(
            engine=search_engine,
            filters=[filter1, filter2],
            http_client=http_client
        )
        results = browser.run("site:example.com")
    """
    
    def __init__(
        self,
        engine: SearchEngine,
        filters: Optional[List[Filter]] = None,
        http_client: Optional[BareNetHttpClient] = None,
        validate_urls: bool = True,
        deduplicate: bool = True
    ) -> None:
        """
        Initialize the browser with orchestration components.
        
        Args:
            engine: The search engine to use for discovery
            filters: Optional list of filters to apply to discovered URLs
            http_client: Optional HTTP client for URL validation
            validate_urls: Whether to validate URLs with HTTP client
            deduplicate: Whether to deduplicate results by final URL
            
        Raises:
            BrowserError: If any required component is invalid
        """
        # Validate and store components
        if not isinstance(engine, SearchEngine):
            raise BrowserError(f"engine must be SearchEngine, got: {type(engine)}")
        self.engine = engine
        
        # Initialize filter pipeline
        self.filter_pipeline = FilterPipeline(filters or [])
        
        # Initialize HTTP client if needed
        if validate_urls and http_client is None:
            logger.info("Creating default HTTP client for URL validation")
            http_client = BareNetHttpClient()
        
        self.http_client = http_client
        self.validate_urls = validate_urls and http_client is not None
        self.deduplicate = deduplicate
        
        # Statistics tracking
        self._stats: Dict[str, int] = {
            'searches': 0,
            'results_found': 0,
            'results_filtered': 0,
            'results_validated': 0,
            'errors': 0
        }
        
        logger.info(
            f"Initialized Browser with engine='{engine.name}', "
            f"filters={len(self.filter_pipeline)}, "
            f"validate_urls={self.validate_urls}, "
            f"deduplicate={self.deduplicate}"
        )
    
    def _validate_search_params(self, query: str, max_results: int) -> None:
        """
        Validate search parameters before execution.
        
        Args:
            query: The search query to validate
            max_results: Maximum results parameter to validate
            
        Raises:
            ValueError: If any parameter is invalid
        """
        if not query or not isinstance(query, str):
            raise ValueError(f"Query must be non-empty string, got: {type(query)}")
        
        query_trimmed = query.strip()
        if not query_trimmed:
            raise ValueError("Query cannot be empty or whitespace only")
        
        if not isinstance(max_results, int):
            raise ValueError(f"max_results must be integer, got: {type(max_results)}")
        
        if max_results < 1:
            raise ValueError(f"max_results must be positive, got: {max_results}")
        
        if max_results > 100:
            logger.warning(f"Large max_results value: {max_results}")
    
    def _apply_filters_without_validation(self, search_result: SearchResult) -> List[FilterResult]:
        """
        Apply filters that don't require HTTP validation data.
        
        This allows us to filter out URLs before making HTTP requests,
        which is more efficient for static filters (e.g., http_only, scheme).
        
        Args:
            search_result: The search result to filter
            
        Returns:
            List of FilterResult objects from each filter
        """
        try:
            # Extract basic information for filtering
            final_url = search_result.url
            # Extract scheme from URL (basic extraction)
            if '://' in final_url:
                final_scheme = final_url.split('://', 1)[0]
            else:
                final_scheme = 'http'  # Default assumption
            
            # Apply filter pipeline without headers
            filter_results = self.filter_pipeline.evaluate(
                original_url=search_result.url,
                final_url=final_url,
                final_scheme=final_scheme,
                headers=None  # No headers available yet
            )
            
            return filter_results
            
        except Exception as e:
            logger.warning(
                f"Pre-validation filter evaluation failed for '{search_result.url}': {e}"
            )
            # Return empty list on filter failure
            return []
    
    def _apply_filters_with_validation(self, 
                                     search_result: SearchResult,
                                     validation_data: Dict[str, Any]) -> List[FilterResult]:
        """
        Apply filters that require HTTP validation data.
        
        This is for filters that need headers, status codes, etc.
        
        Args:
            search_result: The search result to filter
            validation_data: HTTP validation data
            
        Returns:
            List of FilterResult objects from each filter
        """
        try:
            # Apply filter pipeline with headers
            filter_results = self.filter_pipeline.evaluate(
                original_url=search_result.url,
                final_url=validation_data['final_url'],
                final_scheme=validation_data['final_scheme'],
                headers=validation_data['headers']
            )
            
            return filter_results
            
        except Exception as e:
            logger.warning(
                f"Post-validation filter evaluation failed for '{search_result.url}': {e}"
            )
            # Return empty list on filter failure
            return []
    
    def _validate_with_http_client(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Validate a URL using the HTTP client if available.
        
        Args:
            url: The URL to validate
            
        Returns:
            Validation data dict or None if validation failed
        """
        if not self.validate_urls or not self.http_client:
            return None
        
        try:
            validation_result = self.http_client.validate_url(url)
            self._stats['results_validated'] += 1
            return validation_result
            
        except HttpClientError as e:
            logger.debug(f"HTTP validation failed for '{url}': {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error during HTTP validation for '{url}': {e}")
            self._stats['errors'] += 1
            return None
    
    def _process_search_result(
        self,
        search_result: SearchResult,
        seen_urls: Set[str]
    ) -> Optional[DiscoveredResource]:
        """
        Process a single search result through the pipeline.
        
        Optimized pipeline:
          1. Apply pre-validation filters (static filters)
          2. If passed, validate with HTTP client (if enabled)
          3. Apply post-validation filters (header-based filters)
          4. Deduplicate and create resource
        
        Args:
            search_result: The search result to process
            seen_urls: Set of already processed final URLs for deduplication
            
        Returns:
            DiscoveredResource if successful, None if filtered or failed
        """
        # Step 1: Apply pre-validation filters (static filters)
        pre_filter_results = self._apply_filters_without_validation(search_result)
        
        # Check if any pre-validation filter rejected
        if pre_filter_results and not all(r.accepted for r in pre_filter_results):
            logger.debug(f"URL filtered out by pre-validation: {search_result.url}")
            self._stats['results_filtered'] += 1
            return None
        
        # Step 2: Validate with HTTP client (if enabled)
        validation_data = self._validate_with_http_client(search_result.url)
        
        # Step 3: Apply post-validation filters (if we have validation data)
        post_filter_results = []
        if validation_data:
            post_filter_results = self._apply_filters_with_validation(search_result, validation_data)
            
            # Check if any post-validation filter rejected
            if post_filter_results and not all(r.accepted for r in post_filter_results):
                logger.debug(f"URL filtered out by post-validation: {search_result.url}")
                self._stats['results_filtered'] += 1
                return None
        
        # Step 4: Determine final URL and scheme
        if validation_data:
            final_url = validation_data['final_url']
            final_scheme = validation_data['final_scheme']
            status_code = validation_data['status_code']
            headers = validation_data['headers']
            from_cache = validation_data['from_cache']
            # Use timestamp from validation if available
            validated_at = datetime.fromtimestamp(validation_data.get('timestamp', 0))
        else:
            final_url = search_result.url
            # Extract scheme from URL
            if '://' in final_url:
                final_scheme = final_url.split('://', 1)[0]
            else:
                final_scheme = 'http'  # Default assumption
            status_code = None
            headers = None
            from_cache = False
            validated_at = None  # No validation was performed
        
        # Step 5: Deduplication
        if self.deduplicate and final_url in seen_urls:
            logger.debug(f"Duplicate URL skipped: {final_url}")
            return None
        seen_urls.add(final_url)
        
        # Step 6: Collect filter names that passed
        filters_passed = (
            [r.filter_name for r in pre_filter_results if r.accepted] +
            [r.filter_name for r in post_filter_results if r.accepted]
        )
        
        # Step 7: Create discovered resource
        return DiscoveredResource(
            original_url=search_result.url,
            final_url=final_url,
            final_scheme=final_scheme,
            title=search_result.title,
            snippet=search_result.snippet,
            source=search_result.source,
            rank=search_result.rank,
            status_code=status_code,
            headers=headers,
            filters_passed=filters_passed,
            validated_at=validated_at,
            from_cache=from_cache
        )
    
    def run(self, query: str, max_results: int = 20) -> List[DiscoveredResource]:
        """
        Execute the discovery pipeline for a search query.
        
        Pipeline:
          1. Validate inputs
          2. Execute search engine
          3. For each result:
             a. Apply pre-validation filters (static)
             b. Optionally validate with HTTP client
             c. Apply post-validation filters (header-based)
             d. Deduplicate
          4. Return discovered resources
        
        Args:
            query: Search query string
            max_results: Maximum number of results to discover
            
        Returns:
            List of discovered resources that passed all filters
            
        Raises:
            BrowserError: For orchestration failures
            ValueError: For invalid input parameters
        """
        logger.info(f"Starting discovery for query: '{query}' (max_results: {max_results})")
        
        # Reset statistics for this run
        self._stats = {k: 0 for k in self._stats}
        self._stats['searches'] = 1
        
        discovered_resources: List[DiscoveredResource] = []
        seen_urls: Set[str] = set()
        
        try:
            # Step 1: Validate search parameters
            self._validate_search_params(query, max_results)
            
            # Step 2: Execute search engine
            logger.debug(f"Executing search engine: {self.engine.name}")
            try:
                search_results = self.engine.search(query, max_results)
                self._stats['results_found'] = len(search_results)
                logger.info(f"Search engine found {len(search_results)} results")
                
            except SearchEngineError as e:
                logger.error(f"Search engine failed: {e}")
                raise BrowserError(f"Search engine failed: {e}")
            except Exception as e:
                logger.error(f"Unexpected search engine error: {e}")
                self._stats['errors'] += 1
                # Return empty list rather than fail completely
                return discovered_resources
            
            # Step 3: Process each search result
            for search_result in search_results:
                try:
                    resource = self._process_search_result(search_result, seen_urls)
                    if resource:
                        discovered_resources.append(resource)
                        
                except Exception as e:
                    logger.warning(f"Failed to process result: {e}")
                    self._stats['errors'] += 1
                    continue
            
            # Step 4: Log completion statistics
            self._log_statistics(discovered_resources)
            
            return discovered_resources
            
        except Exception as e:
            logger.error(f"Discovery pipeline failed: {e}")
            self._stats['errors'] += 1
            raise BrowserError(f"Discovery pipeline failed: {e}") from e
    
    def _log_statistics(self, discovered_resources: List[DiscoveredResource]) -> None:
        """Log summary statistics for the discovery run."""
        logger.info(
            f"Discovery complete. Statistics: "
            f"searches={self._stats['searches']}, "
            f"found={self._stats['results_found']}, "
            f"filtered={self._stats['results_filtered']}, "
            f"validated={self._stats['results_validated']}, "
            f"discovered={len(discovered_resources)}, "
            f"errors={self._stats['errors']}"
        )
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get current statistics for the browser instance.
        
        Returns:
            Dictionary of statistics counters
        """
        return self._stats.copy()
    
    def __repr__(self) -> str:
        """Return a string representation of the browser."""
        return (
            f"Browser(engine='{self.engine.name}', "
            f"filters={len(self.filter_pipeline)}, "
            f"validate_urls={self.validate_urls}, "
            f"deduplicate={self.deduplicate})"
        )
    
    def __enter__(self) -> 'Browser':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensure cleanup of resources."""
        # Close HTTP client if we created it
        if hasattr(self, 'http_client') and self.http_client:
            try:
                self.http_client.close()
            except Exception:
                pass  # Ignore cleanup errors


# Factory function for common use cases
def create_browser(
    engine_name: str,
    filter_names: Optional[List[str]] = None,
    validate_urls: bool = True,
    **browser_kwargs
) -> Browser:
    """
    Create a Browser instance with common configurations.
    
    This is a convenience function that handles the instantiation
    of engines and filters by name.
    
    Args:
        engine_name: Name of the search engine to use
        filter_names: Optional list of filter names to apply
        validate_urls: Whether to enable URL validation
        **browser_kwargs: Additional arguments passed to Browser
        
    Returns:
        Configured Browser instance
        
    Note:
        This function requires engines and filters to be registered
        in their respective registries. For now, it's a placeholder
        for future extension.
    """
    # Placeholder implementation
    # In a real implementation, this would look up engines and filters
    # from registries or use dependency injection
    
    logger.warning(
        "create_browser is a placeholder. "
        "Direct Browser instantiation is recommended."
    )
    
    # For now, raise an error to indicate this needs to be implemented
    raise NotImplementedError(
        "Engine and filter registration not yet implemented. "
        "Please use Browser constructor directly."
    )
