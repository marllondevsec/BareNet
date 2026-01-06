"""
Base classes for search engines in BareNet.

This module provides the foundational abstractions for all search engines
used in the BareNet system. It defines the structure for search results
and the interface that all search engines must implement.

Design Principles:
- Local-first execution where possible
- No external API dependencies by default
- Passive, ethical data collection only
- Defensive programming with strong typing
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List
import logging

# Configure module-level logger
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SearchResult:
    """
    Immutable container for search result data.
    
    Frozen to prevent accidental modification and ensure thread safety.
    Uses slots for memory efficiency when processing large result sets.
    
    Attributes:
        title: Human-readable title of the result, if available
        url: The complete URL of the discovered resource
        snippet: Brief description or context about the result
        source: Name of the search engine that produced this result
        rank: Position of this result in the search engine's ranking
               (1-indexed, where 1 is the top result)
    """
    url: str
    source: str
    rank: int
    title: Optional[str] = None
    snippet: Optional[str] = None
    
    def __post_init__(self) -> None:
        """
        Validate the result data after initialization.
        
        Raises:
            ValueError: If any required validation fails
        """
        # Validate URL is not empty
        if not self.url or not isinstance(self.url, str):
            raise ValueError(f"Invalid URL: {self.url}")
        
        # Validate source is not empty
        if not self.source or not isinstance(self.source, str):
            raise ValueError(f"Invalid source: {self.source}")
        
        # Validate rank is positive
        if self.rank < 1:
            raise ValueError(f"Rank must be positive, got: {self.rank}")
        
        # Normalize URL by stripping whitespace
        # Note: We use object.__setattr__ because dataclass is frozen
        if isinstance(self.url, str):
            normalized_url = self.url.strip()
            if normalized_url != self.url:
                object.__setattr__(self, 'url', normalized_url)


class SearchEngine(ABC):
    """
    Abstract base class for all search engines in BareNet.
    
    All concrete search engines must inherit from this class and implement
    the required abstract methods. This ensures a consistent interface
    across different discovery methods.
    
    Design Considerations:
    - Search engines should operate passively and ethically
    - No external API keys or authenticated access should be required
    - Rate limiting and respectful crawling must be implemented
    - All network requests should be made through BareNet's HTTP client
    """
    
    def __init__(self) -> None:
        """Initialize the search engine with default settings."""
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.debug(f"Initializing search engine: {self.name}")
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this search engine.
        
        Returns:
            str: The engine's name (e.g., 'duckduckgo', 'localfile')
        
        Note:
            This should be a lowercase, URL-safe string without spaces.
            It will be used in configuration files and command-line arguments.
        """
        pass
    
    @abstractmethod
    def search(self, query: str, max_results: int = 20) -> List[SearchResult]:
        """
        Execute a search and return discovered URLs.
        
        Args:
            query: The search query string
            max_results: Maximum number of results to return
            
        Returns:
            List[SearchResult]: Ordered list of search results
            
        Raises:
            ValueError: If query is invalid or max_results is not positive
            SearchEngineError: For engine-specific failures
            ConnectionError: For network-related issues
        
        Note:
            Implementations must validate inputs and handle errors gracefully.
            Results should be ordered by relevance (if applicable).
        """
        pass
    
    def validate_search_params(self, query: str, max_results: int) -> None:
        """
        Validate search parameters before execution.
        
        Args:
            query: The search query to validate
            max_results: Maximum results parameter to validate
            
        Raises:
            ValueError: If any parameter is invalid
        """
        # Validate query
        if not query or not isinstance(query, str):
            raise ValueError(f"Query must be a non-empty string, got: {type(query)}")
        
        query_trimmed = query.strip()
        if not query_trimmed:
            raise ValueError("Query cannot be empty or whitespace only")
        
        # Validate max_results
        if not isinstance(max_results, int):
            raise ValueError(f"max_results must be an integer, got: {type(max_results)}")
        
        if max_results < 1:
            raise ValueError(f"max_results must be positive, got: {max_results}")
        
        # Log validation at debug level
        self._logger.debug(
            f"Validated search params: query='{query_trimmed}', max_results={max_results}"
        )
    
    def __repr__(self) -> str:
        """Return a string representation of the search engine."""
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __str__(self) -> str:
        """Return a human-readable description of the search engine."""
        return f"Search Engine: {self.name}"


class SearchEngineError(Exception):
    """
    Base exception for search engine failures.
    
    All engine-specific exceptions should inherit from this class
    to enable consistent error handling across the application.
    """
    pass


class RateLimitExceededError(SearchEngineError):
    """Raised when a search engine's rate limit is exceeded."""
    pass


class SearchEngineConfigurationError(SearchEngineError):
    """Raised when a search engine is improperly configured."""
    pass


# Type alias for search engine factories or registries
SearchEngineRegistry = dict[str, SearchEngine]
