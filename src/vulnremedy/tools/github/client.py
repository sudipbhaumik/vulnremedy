"""
GitHub API Client — Handles communication with GitHub REST API.

Provides methods for fetching repository files and metadata.

Architectural notes:
    - Handles authentication (token-based)
    - Implements rate limiting and retries
    - Separates API communication from tool logic
    - Can be swapped for GitLab client without changing tool interface
"""

from __future__ import annotations

import base64
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


class GitHubClient:
    """
    GitHub API client for repository operations.
    
    Usage:
        client = GitHubClient()
        content = client.fetch_file("owner/repo", "pom.xml")
    """ 
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize GitHub client.
        
        Args:
            token: GitHub personal access token (defaults to config)
        """
        self.token = token or settings.github_token
        
        # Build headers
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "VulnRemedy-Scanner"
        }
        
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        else:
            logger.warning(
                "No GitHub token provided. API rate limits will be restrictive "
                "(60 requests/hour vs 5000 requests/hour with token)"
            )
        
        # HTTP client with timeout
        self.client = httpx.Client(
            base_url=settings.github_api_base_url,
            headers=headers,
            timeout=30.0
        )
        
        logger.info("GitHub client initialized", authenticated=bool(self.token))
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(httpx.HTTPError),
    )
    def fetch_file(
        self,
        repo: str,
        file_path: str,
        branch: str = "main"
    ) -> str:
        """
        Fetch file content from a GitHub repository.
        
        Args:
            repo: Repository in format "owner/repo"
            file_path: Path to file (e.g., "pom.xml", "package.json")
            branch: Branch name (default: "main")
        
        Returns:
            File content as string
        
        Raises:
            httpx.HTTPStatusError: If file not found or API error
        """
        logger.info(
            f"Fetching file from GitHub",
            repo=repo,
            file_path=file_path,
            branch=branch
        )
        
        # GitHub Contents API endpoint
        url = f"/repos/{repo}/contents/{file_path}"
        
        params = {"ref": branch}
        
        response = self.client.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # GitHub returns file content base64-encoded
        if "content" not in data:
            raise ValueError(
                f"File {file_path} exists but has no content. "
                f"It might be a directory or too large."
            )
        
        content_b64 = data["content"]
        
        # Decode from base64
        content = base64.b64decode(content_b64).decode("utf-8")
        
        logger.debug(
            f"Successfully fetched file",
            repo=repo,
            file_path=file_path,
            size=len(content)
        )
        
        return content
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(httpx.HTTPError),
    )
    def list_files(
        self,
        repo: str,
        path: str = "",
        branch: str = "main"
    ) -> list[dict[str, Any]]:
        """
        List files in a directory.
        
        Args:
            repo: Repository in format "owner/repo"
            path: Directory path (empty string = root)
            branch: Branch name
        
        Returns:
            List of file/directory metadata
        """
        logger.info(
            f"Listing files from GitHub",
            repo=repo,
            path=path or "/",
            branch=branch
        )
        
        url = f"/repos/{repo}/contents/{path}"
        params = {"ref": branch}
        
        response = self.client.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Filter to only files (not directories)
        if isinstance(data, list):
            files = [item for item in data if item["type"] == "file"]
            logger.debug(f"Found {len(files)} files in {path or '/'}")
            return files
        else:
            # Single file response
            return [data] if data["type"] == "file" else []
    
    def get_rate_limit(self) -> dict[str, Any]:
        """
        Get current rate limit status.
        
        Returns:
            Rate limit information
        """
        response = self.client.get("/rate_limit")
        response.raise_for_status()
        
        data = response.json()
        core = data["resources"]["core"]
        
        logger.info(
            f"GitHub rate limit",
            remaining=core["remaining"],
            limit=core["limit"],
            reset_at=core["reset"]
        )
        
        return core
    
    def close(self) -> None:
        """Close HTTP client."""
        self.client.close()
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, *args):
        """Context manager support."""
        self.close()