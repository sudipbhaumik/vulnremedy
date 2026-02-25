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
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(httpx.HTTPError),
    )
    def get_branch_sha(self, repo: str, branch: str) -> str:
        """
        Get the current HEAD SHA of a branch.

        Args:
            repo: Repository in format "owner/repo"
            branch: Branch name

        Returns:
            SHA string of the branch HEAD commit
        """
        url = f"/repos/{repo}/git/ref/heads/{branch}"
        response = self.client.get(url)
        response.raise_for_status()
        return response.json()["object"]["sha"]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(httpx.HTTPError),
    )
    def create_branch(self, repo: str, branch_name: str, source_sha: str) -> None:
        """
        Create a new branch from a given commit SHA.

        Args:
            repo: Repository in format "owner/repo"
            branch_name: New branch name (e.g. "fix/cve-2021-44228-20240224")
            source_sha: SHA to branch from (typically from get_branch_sha)
        """
        url = f"/repos/{repo}/git/refs"
        payload = {"ref": f"refs/heads/{branch_name}", "sha": source_sha}
        response = self.client.post(url, json=payload)
        response.raise_for_status()
        logger.debug("Branch created", repo=repo, branch=branch_name, sha=source_sha[:8])

    def get_file_sha(
        self, repo: str, file_path: str, branch: str
    ) -> Optional[str]:
        """
        Get the blob SHA of an existing file.

        Required when updating a file via the Contents API — GitHub
        rejects updates that omit the existing file SHA.

        Args:
            repo: Repository in format "owner/repo"
            file_path: Path to file (e.g. "pom.xml")
            branch: Branch name

        Returns:
            SHA string if the file exists, None otherwise
        """
        url = f"/repos/{repo}/contents/{file_path}"
        params = {"ref": branch}
        response = self.client.get(url, params=params)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json().get("sha")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(httpx.HTTPError),
    )
    def create_or_update_file(
        self,
        repo: str,
        file_path: str,
        message: str,
        content: str,
        branch: str,
        existing_sha: Optional[str] = None,
    ) -> str:
        """
        Create or update a file via the GitHub Contents API.

        Args:
            repo: Repository in format "owner/repo"
            file_path: Path to the file
            message: Commit message
            content: New file content (plain text — will be base64-encoded internally)
            branch: Branch to commit to
            existing_sha: Current file SHA (required for updates, omit for new files)

        Returns:
            New blob SHA of the committed file
        """
        url = f"/repos/{repo}/contents/{file_path}"
        payload: dict[str, Any] = {
            "message": message,
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            "branch": branch,
        }
        if existing_sha:
            payload["sha"] = existing_sha
        response = self.client.put(url, json=payload)
        response.raise_for_status()
        new_sha: str = response.json()["content"]["sha"]
        logger.debug(
            "File committed",
            repo=repo,
            file_path=file_path,
            branch=branch,
            sha=new_sha[:8],
        )
        return new_sha

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(httpx.HTTPError),
    )
    def create_pull_request(
        self,
        repo: str,
        title: str,
        body: str,
        head: str,
        base: str = "main",
    ) -> dict[str, Any]:
        """
        Open a pull request.

        Args:
            repo: Repository in format "owner/repo"
            title: PR title
            body: PR body (Markdown)
            head: Source branch name
            base: Target branch (default: "main")

        Returns:
            Dict with keys: number (int), url (str)
        """
        url = f"/repos/{repo}/pulls"
        payload = {"title": title, "body": body, "head": head, "base": base}
        response = self.client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        pr_number: int = data["number"]
        pr_url: str = data["html_url"]
        logger.info("Pull request created", repo=repo, number=pr_number, url=pr_url)
        return {"number": pr_number, "url": pr_url}

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