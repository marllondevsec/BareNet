"""
Base classes for URL filters in BareNet.

This module provides the foundational abstractions for all filtering components
used in the BareNet system. Filters are pure logic modules that evaluate URLs
and HTTP responses to determine whether they should be included in results.

Design Principles:
- Filters are stateless and idempotent
- No network calls or external dependencies
- Composability through pipeline architecture
- Transparent decision logging for auditability
- Defensive programming with strong typing
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, ClassVar
import re
import logging

# Configure module-level logger
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FilterResult:
    """
    Immutable container for filter evaluation results.
    
    Frozen to ensure thread safety and prevent accidental modification
    during pipeline processing. Uses slots for memory efficiency.
    
    Attributes:
        accepted: Whether the URL passed this filter
        reason: Human-readable explanation for the decision
        metadata: Optional additional context about the decision
        filter_name: Name of the filter that produced this result
    """
    accepted: bool
    reason: str
    filter_name: str
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)
    
    def __post_init__(self) -> None:
        """
        Validate the filter result after initialization.
        
        Raises:
            ValueError: If any required validation fails
        """
        # Validate accepted is boolean
        if not isinstance(self.accepted, bool):
            raise ValueError(f"accepted must be boolean, got: {type(self.accepted)}")
        
        # Validate reason is non-empty string
        if not self.reason or not isinstance(self.reason, str):
            raise ValueError(f"reason must be a non-empty string, got: {self.reason}")
        
        # Validate filter_name is non-empty string
        if not self.filter_name or not isinstance(self.filter_name, str):
            raise ValueError(
                f"filter_name must be a non-empty string, got: {self.filter_name}"
            )
        
        # Normalize reason by stripping whitespace
        reason_trimmed = self.reason.strip()
        if reason_trimmed != self.reason:
            object.__setattr__(self, 'reason', reason_trimmed)
    
    @classmethod
    def accept(cls, filter_name: str, reason: str = "", **metadata) -> "FilterResult":
        """
        Factory method for creating acceptance results.
        
        Args:
            filter_name: Name of the filter creating this result
            reason: Optional explanation for acceptance
            **metadata: Additional context about the decision
            
        Returns:
            FilterResult with accepted=True
        """
        return cls(
            accepted=True,
            reason=reason or "URL passed filter criteria",
            filter_name=filter_name,
            metadata=metadata
        )
    
    @classmethod
    def reject(cls, filter_name: str, reason: str, **metadata) -> "FilterResult":
        """
        Factory method for creating rejection results.
        
        Args:
            filter_name: Name of the filter creating this result
            reason: Required explanation for rejection
            **metadata: Additional context about the decision
            
        Returns:
            FilterResult with accepted=False
            
        Raises:
            ValueError: If reason is empty
        """
        if not reason or not reason.strip():
            raise ValueError("Rejection reason must be non-empty")
        
        return cls(
            accepted=False,
            reason=reason.strip(),
            filter_name=filter_name,
            metadata=metadata
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the result to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the filter result
        """
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "filter_name": self.filter_name,
            "metadata": self.metadata.copy()  # Return a copy to prevent mutation
        }


class Filter(ABC):
    """
    Abstract base class for all filters in BareNet.
    
    Filters are pure functions that evaluate URLs and HTTP metadata
    to determine inclusion in results. All concrete filters must
    inherit from this class and implement the required methods.
    
    Design Considerations:
    - Filters must not make network requests
    - Filters should be fast and predictable
    - Filter decisions must be explainable via the reason field
    - Filters are composable in pipelines (AND logic by default)
    
    Implementation Notes:
    - Filters should validate their inputs defensively
    - Filters should log their decisions at appropriate levels
    - Filters can maintain internal state for optimization but
      must remain thread-safe if shared between pipelines
    """
    
    # Class-level constants for common validation patterns
    URL_PATTERN: ClassVar[re.Pattern] = re.compile(
        r'^(https?|ftp)://'  # Protocol
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # Domain
        r'localhost|'  # Localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # IPv4
        r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # IPv6
        r'(?::\d+)?'  # Port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE
    )
    
    SCHEME_PATTERN: ClassVar[re.Pattern] = re.compile(r'^(https?|ftp|file)://', re.IGNORECASE)
    
    def __init__(self) -> None:
        """Initialize the filter with a logger and default settings."""
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.debug(f"Initializing filter: {self.name}")
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this filter.
        
        Returns:
            str: The filter's name (e.g., 'http_only', 'no_hsts')
            
        Note:
            This should be a lowercase, URL-safe string without spaces.
            It will be used in configuration files and command-line arguments.
        """
        pass
    
    @abstractmethod
    def apply(
        self,
        *,
        original_url: str,
        final_url: str,
        final_scheme: str,
        headers: Optional[Dict[str, str]] = None
    ) -> FilterResult:
        """
        Evaluate whether a URL should be included based on filter criteria.
        
        Args:
            original_url: The URL as originally discovered
            final_url: The URL after any redirects or normalization
            final_scheme: The scheme/protocol of the final URL (http, https, etc.)
            headers: HTTP response headers, if available (may be None)
            
        Returns:
            FilterResult: Decision with explanation
            
        Raises:
            ValueError: If input validation fails
            FilterError: For filter-specific evaluation errors
            
        Note:
            - This method must not make network requests
            - Headers may be None for URLs that haven't been fetched yet
            - The decision must be deterministic for the same inputs
        """
        pass
    
    def validate_inputs(
        self,
        *,
        original_url: str,
        final_url: str,
        final_scheme: str,
        headers: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Validate filter inputs before evaluation.
        
        Args:
            original_url: The original URL to validate
            final_url: The final URL to validate
            final_scheme: The scheme to validate
            headers: Headers to validate (optional)
            
        Raises:
            ValueError: If any input is invalid
        """
        # Validate URLs are non-empty strings
        if not isinstance(original_url, str) or not original_url.strip():
            raise ValueError(f"original_url must be non-empty string, got: {original_url}")
        
        if not isinstance(final_url, str) or not final_url.strip():
            raise ValueError(f"final_url must be non-empty string, got: {final_url}")
        
        # Validate scheme is non-empty string
        if not isinstance(final_scheme, str) or not final_scheme.strip():
            raise ValueError(f"final_scheme must be non-empty string, got: {final_scheme}")
        
        # Validate URLs have reasonable format (basic check)
        for url_name, url in [("original_url", original_url), ("final_url", final_url)]:
            if not self.URL_PATTERN.match(url):
                self._logger.warning(
                    f"URL '{url}' may not be well-formed, but proceeding with evaluation"
                )
        
        # Validate scheme format
        if not self.SCHEME_PATTERN.match(final_scheme):
            self._logger.warning(
                f"Scheme '{final_scheme}' may not be valid, but proceeding with evaluation"
            )
        
        # Validate headers if provided
        if headers is not None:
            if not isinstance(headers, dict):
                raise ValueError(f"headers must be dict or None, got: {type(headers)}")
            
            # Validate header keys and values are strings
            for key, value in headers.items():
                if not isinstance(key, str):
                    raise ValueError(f"Header keys must be strings, got: {type(key)}")
                if not isinstance(value, str):
                    raise ValueError(
                        f"Header values must be strings, got: {type(value)} for key '{key}'"
                    )
        
        self._logger.debug(
            f"Validated inputs: original_url='{original_url}', "
            f"final_url='{final_url}', final_scheme='{final_scheme}', "
            f"headers_present={headers is not None}"
        )
    
    def __repr__(self) -> str:
        """Return a string representation of the filter."""
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __str__(self) -> str:
        """Return a human-readable description of the filter."""
        return f"Filter: {self.name}"


class FilterPipeline:
    """
    Composable pipeline for executing multiple filters in sequence.
    
    The pipeline applies filters in the order they are added and
    stops early if any filter rejects a URL (short-circuit evaluation).
    
    Design Considerations:
    - Pipeline is immutable after construction
    - Thread-safe for read operations
    - Provides detailed logging of evaluation flow
    """
    
    def __init__(self, filters: Optional[list[Filter]] = None) -> None:
        """
        Initialize the filter pipeline.
        
        Args:
            filters: Initial list of filters to apply in order
            
        Raises:
            ValueError: If any item in filters is not a Filter instance
        """
        self._filters: list[Filter] = []
        self._logger = logging.getLogger(f"{__name__}.FilterPipeline")
        
        if filters:
            for filter_obj in filters:
                self.add_filter(filter_obj)
        
        self._logger.info(f"Initialized pipeline with {len(self._filters)} filters")
    
    def add_filter(self, filter_obj: Filter) -> None:
        """
        Add a filter to the end of the pipeline.
        
        Args:
            filter_obj: Filter instance to add
            
        Raises:
            ValueError: If filter_obj is not a Filter instance
        """
        if not isinstance(filter_obj, Filter):
            raise ValueError(f"Expected Filter instance, got: {type(filter_obj)}")
        
        self._filters.append(filter_obj)
        self._logger.debug(f"Added filter '{filter_obj.name}' to pipeline")
    
    def evaluate(
        self,
        *,
        original_url: str,
        final_url: str,
        final_scheme: str,
        headers: Optional[Dict[str, str]] = None
    ) -> list[FilterResult]:
        """
        Evaluate a URL through all filters in the pipeline.
        
        Args:
            original_url: The URL as originally discovered
            final_url: The URL after any redirects or normalization
            final_scheme: The scheme/protocol of the final URL
            headers: HTTP response headers, if available
            
        Returns:
            List of FilterResult objects in the order filters were applied
            
        Raises:
            ValueError: If input validation fails
            FilterError: If any filter evaluation fails
        """
        results = []
        
        self._logger.debug(
            f"Starting pipeline evaluation for URL: {final_url} "
            f"(original: {original_url})"
        )
        
        for filter_obj in self._filters:
            try:
                result = filter_obj.apply(
                    original_url=original_url,
                    final_url=final_url,
                    final_scheme=final_scheme,
                    headers=headers
                )
                results.append(result)
                
                # Log the decision
                log_level = logging.DEBUG if result.accepted else logging.INFO
                self._logger.log(
                    log_level,
                    f"Filter '{filter_obj.name}' {'accepted' if result.accepted else 'rejected'} "
                    f"URL '{final_url}': {result.reason}"
                )
                
                # Short-circuit on rejection
                if not result.accepted:
                    self._logger.debug(
                        f"Pipeline stopped early due to rejection by '{filter_obj.name}'"
                    )
                    break
                    
            except Exception as e:
                self._logger.error(
                    f"Filter '{filter_obj.name}' failed for URL '{final_url}': {e}",
                    exc_info=True
                )
                raise FilterError(f"Filter '{filter_obj.name}' evaluation failed") from e
        
        self._logger.debug(
            f"Pipeline evaluation complete for '{final_url}': "
            f"{sum(r.accepted for r in results)}/{len(results)} filters passed"
        )
        
        return results
    
    @property
    def filters(self) -> list[str]:
        """Return the names of filters in the pipeline, in order."""
        return [f.name for f in self._filters]
    
    def __len__(self) -> int:
        """Return the number of filters in the pipeline."""
        return len(self._filters)
    
    def __repr__(self) -> str:
        """Return a string representation of the pipeline."""
        filter_names = ", ".join(self.filters)
        return f"FilterPipeline(filters=[{filter_names}])"


class FilterError(Exception):
    """
    Base exception for filter-related failures.
    
    All filter-specific exceptions should inherit from this class
    to enable consistent error handling across the application.
    """
    pass


class FilterConfigurationError(FilterError):
    """Raised when a filter is improperly configured."""
    pass


class FilterValidationError(FilterError):
    """Raised when filter inputs fail validation."""
    pass


# Type alias for filter factories or registries
FilterRegistry = dict[str, Filter]
