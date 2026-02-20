"""
NVD Loader — Download CVE data from National Vulnerability Database.

Handles NVD API communication with proper rate limiting, retries,
and caching to respect API limits and enable incremental downloads.

Architectural notes:
    - Rate limiting: 5 requests per 30 seconds (unauthenticated)
    - Retries: 3 attempts with exponential backoff
    - Caching: Downloaded data saved to data/raw/nvd/ to avoid re-fetching
    - Pagination: Handles NVD's 2000 CVE per request limit automatically
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from vulnremedy.utils.config import settings
from vulnremedy.utils.logging import logger


class NVDLoader:
    """
    Downloads CVE data from NVD API with rate limiting and caching.
    
    Usage:
        loader = NVDLoader()
        cves = loader.fetch_cves(days_back=90)  # Last 90 days
    """
    
    #BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    RATE_LIMIT_REQUESTS = 5
    RATE_LIMIT_WINDOW_SECONDS = 30
    RESULTS_PER_PAGE = 2000  # NVD max
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize NVD loader.
        
        Args:
            cache_dir: Where to cache downloaded data. Defaults to data/raw/nvd/
        """
        self.base_url = settings.nvd_api_base_url
        self.cache_dir = cache_dir or Path("data/raw/nvd")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting state
        self.request_timestamps: list[float] = []
        
        # HTTP client with timeout
        self.client = httpx.Client(timeout=60.0)
        
        logger.info(
            "NVD Loader initialized",
            cache_dir=str(self.cache_dir),
            rate_limit=f"{self.RATE_LIMIT_REQUESTS} requests per {self.RATE_LIMIT_WINDOW_SECONDS}s"
        )
    
    def fetch_cves(
        self,
        days_back: int = 90,
        max_results: Optional[int] = None
    ) -> list[dict[str, Any]]:
        """
        Fetch CVEs from NVD published in the last N days.
        
        Args:
            days_back: How many days back to fetch CVEs from
            max_results: Maximum number of CVEs to return (None = all)
        
        Returns:
            List of CVE dictionaries in NVD API format
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        
        logger.info(
            "Starting NVD fetch",
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            max_results=max_results or "unlimited"
        )
        
        all_cves = []
        start_index = 0
        
        while True:
            # Check cache first
            cache_file = self._get_cache_filename(start_date, end_date, start_index)
            
            if cache_file.exists():
                logger.info(f"Loading from cache: {cache_file.name}")
                with open(cache_file, "r") as f:
                    response_data = json.load(f)
            else:
                # Fetch from API
                response_data = self._fetch_page(
                    start_date=start_date,
                    end_date=end_date,
                    start_index=start_index
                )
                
                # Cache the response
                self._save_to_cache(cache_file, response_data)
            
            # Extract CVEs from response
            vulnerabilities = response_data.get("vulnerabilities", [])
            if not vulnerabilities:
                logger.info("No more CVEs to fetch")
                break
            
            all_cves.extend(vulnerabilities)
            
            logger.info(
                f"Fetched {len(vulnerabilities)} CVEs (total: {len(all_cves)})"
            )
            
            # Check if we hit max_results
            if max_results and len(all_cves) >= max_results:
                all_cves = all_cves[:max_results]
                logger.info(f"Reached max_results limit: {max_results}")
                break
            
            # Check if there are more pages
            total_results = response_data.get("totalResults", 0)
            if start_index + self.RESULTS_PER_PAGE >= total_results:
                logger.info("Fetched all available CVEs")
                break
            
            start_index += self.RESULTS_PER_PAGE
        
        logger.info(f"NVD fetch complete. Total CVEs: {len(all_cves)}")
        return all_cves
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(httpx.HTTPError),
    )
    def _fetch_page(
        self,
        start_date: datetime,
        end_date: datetime,
        start_index: int
    ) -> dict[str, Any]:
        """
        Fetch a single page of CVEs from NVD API with retries.
        
        This method is decorated with @retry so it automatically
        retries on HTTP errors with exponential backoff.
        """
        self._respect_rate_limit()
        
        params = {
            "pubStartDate": start_date.strftime("%Y-%m-%dT00:00:00.000"),
            "pubEndDate": end_date.strftime("%Y-%m-%dT23:59:59.999"),
            "startIndex": start_index,
            "resultsPerPage": self.RESULTS_PER_PAGE,
        }
        
        # Add API key if configured
        if settings.nvd_api_key:
            params["apiKey"] = settings.nvd_api_key
        
        logger.info(
            "Making NVD API request",
            start_index=start_index,
            results_per_page=self.RESULTS_PER_PAGE
        )
        
        response = self.client.get(self.base_url, params=params)
        response.raise_for_status()
        
        # Track request for rate limiting
        self.request_timestamps.append(time.time())
        
        return response.json()
    
    def _respect_rate_limit(self) -> None:
        """
        Sleep if necessary to respect NVD rate limits.
        
        Tracks last N requests and sleeps if we're about to exceed
        the rate limit window.
        """
        now = time.time()
        
        # Remove timestamps outside the rate limit window
        cutoff = now - self.RATE_LIMIT_WINDOW_SECONDS
        self.request_timestamps = [
            ts for ts in self.request_timestamps if ts > cutoff
        ]
        
        # If we're at the limit, sleep until oldest request expires
        if len(self.request_timestamps) >= self.RATE_LIMIT_REQUESTS:
            oldest = self.request_timestamps[0]
            sleep_time = self.RATE_LIMIT_WINDOW_SECONDS - (now - oldest) + 1
            
            if sleep_time > 0:
                logger.info(
                    f"Rate limit reached. Sleeping for {sleep_time:.1f}s"
                )
                time.sleep(sleep_time)
    
    def _get_cache_filename(
        self,
        start_date: datetime,
        end_date: datetime,
        start_index: int
    ) -> Path:
        """
        Generate cache filename based on query parameters.
        
        This ensures we don't re-download the same data.
        """
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        return self.cache_dir / f"nvd_{start_str}_{end_str}_{start_index}.json"
    
    def _save_to_cache(self, cache_file: Path, data: dict[str, Any]) -> None:
        """Save API response to cache file."""
        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved to cache: {cache_file.name}")
    
    def close(self) -> None:
        """Close HTTP client."""
        self.client.close()
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, *args):
        """Context manager support."""
        self.close()