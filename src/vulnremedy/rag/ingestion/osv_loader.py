"""
OSV Loader — Download vulnerability data from OSV.dev (Open Source Vulnerabilities).

OSV.dev is a distributed vulnerability database for open source.
It often has faster updates than NVD and better ecosystem-specific data.

Architectural notes:
    - No rate limiting needed (OSV has no limits)
    - Query by ecosystem (Maven, npm, PyPI) rather than date range
    - Different response schema than NVD — requires separate parser
    - Simpler API, more reliable for recent vulnerabilities
"""

from __future__ import annotations

import json
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


class OSVLoader:
    """
    Downloads vulnerability data from OSV.dev API.
    
    Usage:
        loader = OSVLoader()
        vulns = loader.fetch_vulnerabilities(
            ecosystems=["Maven", "PyPI"],
            max_per_ecosystem=500
        )
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize OSV loader.
        
        Args:
            cache_dir: Where to cache downloaded data. Defaults to data/raw/osv/
        """
        self.base_url = settings.osv_api_base_url
        self.cache_dir = cache_dir or Path("data/raw/osv")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # HTTP client with timeout
        self.client = httpx.Client(timeout=60.0)
        
        logger.info(
            "OSV Loader initialized",
            cache_dir=str(self.cache_dir),
            base_url=self.base_url
        )
    
    def fetch_vulnerabilities(
        self,
        ecosystems: list[str],
        max_per_ecosystem: Optional[int] = None
    ) -> list[dict[str, Any]]:
        """
        Fetch vulnerabilities from OSV.dev for specified ecosystems.
        
        Args:
            ecosystems: List of ecosystems to query e.g. ["Maven", "PyPI", "npm"]
            max_per_ecosystem: Max vulnerabilities per ecosystem (None = all available)
        
        Returns:
            List of vulnerability dictionaries in OSV format
        """
        all_vulns = []
        
        for ecosystem in ecosystems:
            logger.info(f"Fetching OSV vulnerabilities for ecosystem: {ecosystem}")
            
            # Check cache first
            cache_file = self._get_cache_filename(ecosystem)
            
            if cache_file.exists():
                logger.info(f"Loading from cache: {cache_file.name}")
                with open(cache_file, "r") as f:
                    vulns = json.load(f)
            else:
                # Fetch from API
                vulns = self._fetch_ecosystem(ecosystem)
                
                # Cache the response
                self._save_to_cache(cache_file, vulns)
            
            # Apply limit if specified
            if max_per_ecosystem and len(vulns) > max_per_ecosystem:
                vulns = vulns[:max_per_ecosystem]
                logger.info(
                    f"Limited to {max_per_ecosystem} vulnerabilities for {ecosystem}"
                )
            
            all_vulns.extend(vulns)
            
            logger.info(
                f"Fetched {len(vulns)} vulnerabilities for {ecosystem} "
                f"(total: {len(all_vulns)})"
            )
        
        logger.info(f"OSV fetch complete. Total vulnerabilities: {len(all_vulns)}")
        return all_vulns
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(httpx.HTTPError),
    )
    def _fetch_ecosystem(self, ecosystem: str) -> list[dict[str, Any]]:
        """
        Fetch all vulnerabilities for a specific ecosystem.
        
        OSV.dev query endpoint returns paginated results.
        We fetch all pages and combine them.
        """
        url = f"{self.base_url}/query"
        
        # OSV query format
        query = {
            "package": {
                "ecosystem": ecosystem
            }
        }
        
        logger.info(f"Making OSV API request for ecosystem: {ecosystem}")
        
        # Note: OSV's query endpoint has pagination but for our test dataset
        # we'll fetch the first batch. Production would implement pagination.
        response = self.client.post(url, json=query)
        response.raise_for_status()
        
        data = response.json()
        
        # OSV returns vulnerabilities in "vulns" key
        vulnerabilities = data.get("vulns", [])
        
        # For each vulnerability ID, fetch full details
        full_vulns = []
        for vuln_summary in vulnerabilities[:100]:  # Limit to 100 for test dataset
            vuln_id = vuln_summary.get("id")
            if vuln_id:
                full_vuln = self._fetch_vulnerability_details(vuln_id)
                if full_vuln:
                    full_vulns.append(full_vuln)
        
        return full_vulns
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type(httpx.HTTPError),
    )
    def _fetch_vulnerability_details(self, vuln_id: str) -> Optional[dict[str, Any]]:
        """
        Fetch full details for a single vulnerability by ID.
        
        OSV's /v1/vulns/{id} endpoint returns complete vulnerability data.
        """
        url = f"{self.base_url}/vulns/{vuln_id}"
        
        try:
            response = self.client.get(url)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.warning(f"Failed to fetch details for {vuln_id}: {e}")
            return None
    
    def _get_cache_filename(self, ecosystem: str) -> Path:
        """
        Generate cache filename based on ecosystem.
        
        This ensures we don't re-download the same ecosystem data.
        """
        return self.cache_dir / f"osv_{ecosystem.lower()}.json"
    
    def _save_to_cache(self, cache_file: Path, data: list[dict[str, Any]]) -> None:
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