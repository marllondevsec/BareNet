"""
HTTP client for BareNet - A passive, ethical research tool.

This module provides a secure, conservative HTTP client designed for
defensive security research. It focuses on safe URL validation and
metadata collection without aggressive crawling or data extraction.

Design Principles:
- Passive and respectful: Conservative rate limiting and caching
- Safe by default: Timeouts, size limits, and redirect controls
- Ethical usage: Clear identification, no TLS bypass, no MITM
- Defensive programming: Input validation and graceful error handling

Important Constraints:
- No automated crawling or parallel requests
- No bypassing of security controls (HSTS, TLS, etc.)
- No user data collection or fingerprinting
- Suitable for authorized research only
"""

import time
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any
from urllib.parse import urlparse, urlunparse, ParseResult
import logging

import httpx

# Configure module-level logger
logger = logging.getLogger(__name__)


class HttpClientError(Exception):
    """Base exception for HTTP client failures."""
    pass


class InvalidURLError(HttpClientError):
    """Raised when a URL is malformed or invalid."""
    pass


class RateLimitExceededError(HttpClientError):
    """Raised when the rate limit is exceeded."""
    pass


class SecurityPolicyViolationError(HttpClientError):
    """Raised when a request violates security policies."""
    pass


class BareNetHttpClient:
    """
    Conservative HTTP client for ethical security research.
    
    This client is designed to safely validate URLs and collect minimal
    metadata without aggressive scanning or data extraction.
    
    Key Features:
    - URL normalization and validation
    - Conservative rate limiting (1 req/sec default)
    - TTL-based in-memory caching
    - HEAD-first approach with GET fallback
    - Strict safety limits (timeouts, redirects, size limits)
    - Clear identification via User-Agent
    
    Security Considerations:
    - Never bypasses TLS/SSL validation
    - Respects HSTS and other security headers
    - No proxy or MITM capability
    - No authentication or credential handling
    """
    
    # Default safety limits (conservative values for research)
    DEFAULT_TIMEOUT: float = 10.0  # seconds
    DEFAULT_MAX_REDIRECTS: int = 5
    DEFAULT_MAX_GET_SIZE: int = 100 * 1024  # 100 KB
    DEFAULT_CACHE_TTL: int = 3600  # 1 hour
    DEFAULT_RATE_LIMIT: float = 1.0  # requests per second
    
    # User agent clearly identifying the tool
    USER_AGENT: str = "BareNet/1.0 (+https://github.com/yourusername/barenet) SecurityResearch"
    
    def __init__(
        self,
        rate_limit: float = DEFAULT_RATE_LIMIT,
        cache_ttl: int = DEFAULT_CACHE_TTL,
        timeout: float = DEFAULT_TIMEOUT,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        max_get_size: int = DEFAULT_MAX_GET_SIZE,
        verify_ssl: bool = True
    ) -> None:
        """
        Initialize the HTTP client with safety constraints.
        
        Args:
            rate_limit: Requests per second (must be > 0)
            cache_ttl: Cache time-to-live in seconds
            timeout: Request timeout in seconds
            max_redirects: Maximum number of redirects to follow
            max_get_size: Maximum GET response body size in bytes
            verify_ssl: Verify SSL certificates (ALWAYS True for ethical research)
            
        Raises:
            ValueError: If any parameter violates safety constraints
        """
        # Validate safety constraints
        if rate_limit <= 0:
            raise ValueError(f"rate_limit must be positive, got: {rate_limit}")
        if cache_ttl < 0:
            raise ValueError(f"cache_ttl must be non-negative, got: {cache_ttl}")
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got: {timeout}")
        if max_redirects < 0:
            raise ValueError(f"max_redirects must be non-negative, got: {max_redirects}")
        if max_get_size <= 0:
            raise ValueError(f"max_get_size must be positive, got: {max_get_size}")
        
        # Safety enforcement: Always verify SSL in ethical research
        if not verify_ssl:
            raise SecurityPolicyViolationError(
                "SSL verification cannot be disabled in BareNet for ethical research"
            )
        
        self.rate_limit = rate_limit
        self.cache_ttl = cache_ttl
        self.timeout = timeout
        self.max_redirects = max_redirects
        self.max_get_size = max_get_size
        self.verify_ssl = verify_ssl
        
        # Rate limiting state
        self._last_request_time: float = 0.0
        
        # In-memory cache with TTL
        self._cache: Dict[str, Dict[str, Any]] = {}
        
        # Initialize httpx client with conservative settings
        self._client = httpx.Client(
            timeout=self.timeout,
            max_redirects=self.max_redirects,
            follow_redirects=True,
            verify=self.verify_ssl,
            headers={
                'User-Agent': self.USER_AGENT,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'close',  # Use separate connections for each request
                'Upgrade-Insecure-Requests': '1',
            },
            limits=httpx.Limits(
                max_keepalive_connections=0,  # Disable keep-alive for safety
                max_connections=1,  # Conservative single connection
            ),
            event_hooks={
                'response': [self._response_hook]
            }
        )
        
        logger.info(
            f"Initialized BareNetHttpClient with rate_limit={rate_limit}/sec, "
            f"cache_ttl={cache_ttl}s, timeout={timeout}s"
        )
    
    def _response_hook(self, response: httpx.Response) -> None:
        """
        Hook to log response details for debugging.
        
        This is called after each response and can be used for logging,
        metrics collection, or additional validation.
        
        Args:
            response: The HTTP response object
        """
        logger.debug(
            f"HTTP {response.request.method} {response.url} -> "
            f"{response.status_code} ({len(response.content)} bytes)"
        )
    
    def _enforce_rate_limit(self) -> None:
        """
        Enforce rate limiting by sleeping if necessary.
        
        This implements a simple token bucket-like rate limiter
        to ensure we don't exceed the configured requests per second.
        
        Raises:
            RateLimitExceededError: If rate limiting is too aggressive
        """
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        # Calculate required delay
        min_interval = 1.0 / self.rate_limit
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            if sleep_time > 5.0:  # Safety check
                raise RateLimitExceededError(
                    f"Rate limit would require sleeping {sleep_time:.1f}s, "
                    f"which is unusually long. Check rate_limit configuration."
                )
            
            logger.debug(f"Rate limiting: sleeping {sleep_time:.3f}s")
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    def _normalize_url(self, url: str) -> str:
        """
        Normalize and validate a URL for safe processing.
        
        Performs:
        - Scheme normalization (adds https:// if no scheme)
        - Basic URL validation
        - Path normalization
        
        Args:
            url: The URL to normalize
            
        Returns:
            Normalized URL string
            
        Raises:
            InvalidURLError: If the URL is malformed or invalid
        """
        if not url or not isinstance(url, str):
            raise InvalidURLError(f"URL must be non-empty string, got: {type(url)}")
        
        url = url.strip()
        
        # Basic validation - reject obviously bad URLs
        if len(url) > 2048:  # RFC 7230 recommends 8000, but we're conservative
            raise InvalidURLError(f"URL exceeds maximum length (2048 chars)")
        
        if ' ' in url or '\n' in url or '\r' in url:
            raise InvalidURLError("URL contains whitespace or control characters")
        
        # Add default scheme if missing
        if not url.startswith(('http://', 'https://')):
            logger.debug(f"URL '{url}' missing scheme, defaulting to https")
            url = f'https://{url}'
        
        try:
            parsed = urlparse(url)
            
            # Validate parsed components
            if not parsed.netloc:
                raise InvalidURLError(f"URL has no network location: {url}")
            
            # Reject non-HTTP(S) schemes for safety
            if parsed.scheme not in ('http', 'https'):
                raise InvalidURLError(
                    f"Unsupported scheme '{parsed.scheme}'. Only http/https allowed."
                )
            
            # Normalize the URL
            normalized = ParseResult(
                scheme=parsed.scheme,
                netloc=parsed.netloc.lower(),  # Case-insensitive
                path=parsed.path or '/',  # Default to root
                params=parsed.params,
                query=parsed.query,
                fragment=''  # Strip fragments as they're client-side
            )
            
            normalized_url = urlunparse(normalized)
            logger.debug(f"Normalized '{url}' -> '{normalized_url}'")
            
            return normalized_url
            
        except Exception as e:
            raise InvalidURLError(f"Failed to parse URL '{url}': {e}")
    
    def _generate_cache_key(self, url: str) -> str:
        """
        Generate a deterministic cache key for a URL.
        
        Uses SHA-256 to create a fixed-length key that's resistant
        to collision attacks.
        
        Args:
            url: The URL to generate a key for
            
        Returns:
            SHA-256 hash of the normalized URL
        """
        # Use normalized URL for consistency
        normalized = self._normalize_url(url)
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached response if valid.
        
        Args:
            cache_key: The cache key to look up
            
        Returns:
            Cached response dict or None if not found/expired
        """
        if cache_key not in self._cache:
            return None
        
        cached = self._cache[cache_key]
        
        # Check TTL
        cache_time = cached['timestamp']
        current_time = time.time()
        
        if current_time - cache_time > self.cache_ttl:
            logger.debug(f"Cache entry expired for key {cache_key[:16]}...")
            del self._cache[cache_key]
            return None
        
        logger.debug(f"Cache hit for key {cache_key[:16]}...")
        return cached
    
    def _save_to_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """
        Save a response to the cache.
        
        Args:
            cache_key: The cache key
            data: The response data to cache
        """
        # Add timestamp for TTL calculation
        data['timestamp'] = time.time()
        self._cache[cache_key] = data
        
        logger.debug(f"Cached response for key {cache_key[:16]}...")
        
        # Optional: Implement cache size limiting here if needed
        # Currently relying on TTL for cache management
    
    def _fetch_url_metadata(
        self,
        url: str
    ) -> Tuple[str, str, int, Dict[str, str]]:
        """
        Fetch metadata for a URL using HEAD-first approach.
        
        Strategy:
        1. Try HEAD request (lightweight, no body)
        2. If HEAD fails (405, 501, etc.), try GET with size limit
        3. Extract final URL after redirects
        
        Args:
            url: The URL to fetch
            
        Returns:
            Tuple of (final_url, final_scheme, status_code, headers)
            
        Raises:
            HttpClientError: For network or protocol errors
        """
        final_url = url
        final_scheme = urlparse(url).scheme
        status_code = 0
        headers: Dict[str, str] = {}
        
        try:
            # Step 1: Try HEAD request
            logger.debug(f"Attempting HEAD request to {url}")
            response = self._client.head(url, allow_redirects=True)
            
            final_url = str(response.url)
            final_scheme = urlparse(final_url).scheme
            status_code = response.status_code
            headers = dict(response.headers)
            
            # If HEAD succeeded, we're done
            logger.debug(f"HEAD successful: {status_code} for {final_url}")
            
        except httpx.HTTPStatusError as e:
            # HEAD method not allowed or other 4xx/5xx
            if e.response.status_code in (405, 501):  # Method Not Allowed / Not Implemented
                logger.debug(f"HEAD not supported, falling back to GET for {url}")
                
                # Step 2: Fall back to GET with size limit
                try:
                    with self._client.stream('GET', url) as response:
                        final_url = str(response.url)
                        final_scheme = urlparse(final_url).scheme
                        status_code = response.status_code
                        headers = dict(response.headers)
                        
                        # Read response body up to size limit
                        total_bytes = 0
                        for chunk in response.iter_bytes():
                            total_bytes += len(chunk)
                            if total_bytes > self.max_get_size:
                                logger.warning(
                                    f"GET response exceeded size limit ({self.max_get_size} bytes) "
                                    f"for {url}, truncating"
                                )
                                break
                        
                        logger.debug(f"GET successful: {status_code} for {final_url}")
                        
                except httpx.RequestError as get_err:
                    raise HttpClientError(f"GET request failed for {url}: {get_err}")
            else:
                # Other HTTP error from HEAD request
                raise HttpClientError(f"HEAD request failed for {url}: {e}")
                
        except httpx.RequestError as e:
            # Network errors, timeouts, etc.
            raise HttpClientError(f"Request failed for {url}: {e}")
        
        # Normalize headers to lowercase keys for consistency
        normalized_headers = {k.lower(): v for k, v in headers.items()}
        
        return final_url, final_scheme, status_code, normalized_headers
    
    def validate_url(self, url: str) -> Dict[str, Any]:
        """
        Validate a URL and collect metadata for research purposes.
        
        This is the primary public method. It performs:
        1. URL normalization and validation
        2. Cache lookup
        3. Rate limiting
        4. HEAD-first metadata collection
        5. Result caching
        
        Args:
            url: The URL to validate
            
        Returns:
            Dictionary containing:
                - original_url: The input URL
                - final_url: URL after redirects
                - final_scheme: Scheme of final URL (http/https)
                - status_code: HTTP status code
                - headers: HTTP response headers (lowercase keys)
                - from_cache: Boolean indicating if result was cached
                
        Raises:
            InvalidURLError: If URL is malformed
            HttpClientError: For network or protocol errors
            SecurityPolicyViolationError: For policy violations
            RateLimitExceededError: If rate limit is exceeded
        """
        logger.info(f"Validating URL: {url}")
        
        # Step 1: Normalize URL
        try:
            normalized_url = self._normalize_url(url)
        except InvalidURLError:
            raise
        except Exception as e:
            raise InvalidURLError(f"URL validation failed: {e}")
        
        # Step 2: Check cache
        cache_key = self._generate_cache_key(normalized_url)
        cached = self._get_from_cache(cache_key)
        
        if cached:
            result = cached.copy()
            result['from_cache'] = True
            result['original_url'] = url  # Preserve original input
            logger.info(f"Returning cached result for {normalized_url}")
            return result
        
        # Step 3: Enforce rate limit
        self._enforce_rate_limit()
        
        # Step 4: Fetch metadata
        try:
            final_url, final_scheme, status_code, headers = self._fetch_url_metadata(
                normalized_url
            )
        except Exception as e:
            logger.error(f"Failed to fetch metadata for {normalized_url}: {e}")
            raise
        
        # Step 5: Prepare result
        result = {
            'original_url': url,
            'final_url': final_url,
            'final_scheme': final_scheme,
            'status_code': status_code,
            'headers': headers,
            'from_cache': False,
            'timestamp': time.time()
        }
        
        # Step 6: Cache the result
        self._save_to_cache(cache_key, result.copy())
        
        logger.info(
            f"Validation complete: {normalized_url} -> {final_url} "
            f"({status_code}, scheme: {final_scheme})"
        )
        
        return result
    
    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        cache_size = len(self._cache)
        self._cache.clear()
        logger.info(f"Cleared cache ({cache_size} entries)")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for monitoring.
        
        Returns:
            Dictionary with cache statistics
        """
        current_time = time.time()
        expired_count = 0
        
        # Count expired entries
        for key, entry in list(self._cache.items()):
            if current_time - entry['timestamp'] > self.cache_ttl:
                expired_count += 1
        
        return {
            'total_entries': len(self._cache),
            'expired_entries': expired_count,
            'cache_ttl': self.cache_ttl,
            'rate_limit': self.rate_limit
        }
    
    def close(self) -> None:
        """Close the HTTP client and release resources."""
        if hasattr(self, '_client'):
            self._client.close()
            logger.info("HTTP client closed")
    
    def __enter__(self) -> 'BareNetHttpClient':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensure cleanup."""
        self.close()
    
    def __del__(self) -> None:
        """Destructor - ensure cleanup."""
        self.close()


# Convenience function for single URL validation
def validate_single_url(url: str, **kwargs) -> Dict[str, Any]:
    """
    Convenience function for validating a single URL.
    
    Creates a temporary client, validates the URL, and cleans up.
    
    Args:
        url: The URL to validate
        **kwargs: Additional arguments passed to BareNetHttpClient
        
    Returns:
        Validation result dictionary
    """
    with BareNetHttpClient(**kwargs) as client:
        return client.validate_url(url)
