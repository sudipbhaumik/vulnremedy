"""
GitHub MCP Tool — Agent-facing interface for GitHub operations.

Wraps GitHubClient to provide a clean tool interface following
the Model Context Protocol (MCP) pattern.

Architectural notes:
    - Agents call this tool, not the client directly
    - Tool handles input validation and error formatting
    - Tool provides synchronous interface (client handles async internally)
    - Tool can be easily mocked for testing
"""

from __future__ import annotations

from typing import Optional

from vulnremedy.tools.github.client import GitHubClient
from vulnremedy.utils.logging import logger


class GitHubTool:
    """
    MCP tool for GitHub repository operations.
    
    Provides agent-friendly interface for fetching files and
    listing repository contents.
    
    Usage:
        tool = GitHubTool()
        content = tool.fetch_file(
            repo="apache/log4j",
            file_path="pom.xml"
        )
    """
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize GitHub tool.
        
        Args:
            token: GitHub personal access token (optional)
        """
        self.client = GitHubClient(token=token)
        logger.info("GitHub tool initialized")
    
    def fetch_file(
        self,
        repo: str,
        file_path: str,
        branch: str = "main"
    ) -> dict[str, str | bool]:
        """
        Fetch a file from a GitHub repository.
        
        Args:
            repo: Repository in format "owner/repo" (e.g., "apache/log4j")
            file_path: Path to file (e.g., "pom.xml", "src/main/java/App.java")
            branch: Branch name (default: "main")
        
        Returns:
            Dictionary with:
                - success: bool
                - content: str (if success=True)
                - error: str (if success=False)
        """
        # Validate repo format
        if "/" not in repo:
            return {
                "success": False,
                "error": f"Invalid repo format: '{repo}'. Expected 'owner/repo'"
            }
        
        try:
            content = self.client.fetch_file(
                repo=repo,
                file_path=file_path,
                branch=branch
            )
            
            return {
                "success": True,
                "content": content
            }
            
        except Exception as e:
            logger.error(
                f"Failed to fetch file from GitHub",
                repo=repo,
                file_path=file_path,
                error=str(e)
            )
            
            return {
                "success": False,
                "error": f"Failed to fetch {file_path}: {str(e)}"
            }
    
    def list_files(
        self,
        repo: str,
        path: str = "",
        branch: str = "main"
    ) -> dict[str, list | str | bool]:
        """
        List files in a repository directory.
        
        Args:
            repo: Repository in format "owner/repo"
            path: Directory path (empty string = root)
            branch: Branch name (default: "main")
        
        Returns:
            Dictionary with:
                - success: bool
                - files: list[dict] (if success=True)
                - error: str (if success=False)
        """
        # Validate repo format
        if "/" not in repo:
            return {
                "success": False,
                "error": f"Invalid repo format: '{repo}'. Expected 'owner/repo'"
            }
        
        try:
            files = self.client.list_files(
                repo=repo,
                path=path,
                branch=branch
            )
            
            return {
                "success": True,
                "files": files
            }
            
        except Exception as e:
            logger.error(
                f"Failed to list files from GitHub",
                repo=repo,
                path=path,
                error=str(e)
            )
            
            return {
                "success": False,
                "error": f"Failed to list files in {path or '/'}: {str(e)}"
            }
    
    def find_manifest_files(
        self,
        repo: str,
        branch: str = "main"
    ) -> dict[str, list | str | bool]:
        """
        Find common dependency manifest files in repository.
        
        Searches for:
        - pom.xml (Maven)
        - build.gradle (Gradle)
        - package.json (npm)
        - requirements.txt (pip)
        - Cargo.toml (Rust)
        - go.mod (Go)
        
        Args:
            repo: Repository in format "owner/repo"
            branch: Branch name (default: "main")
        
        Returns:
            Dictionary with:
                - success: bool
                - manifests: list[dict] (if success=True)
                - error: str (if success=False)
        """
        manifest_filenames = [
            "pom.xml",
            "build.gradle",  
            "package.json",
            "build.gradle.kts",
            "requirements.txt",
            "Cargo.toml",
            "go.mod"
        ]
        
        found_manifests = []
        
        try:
            # List files in root directory
            result = self.list_files(repo=repo, path="", branch=branch)
            
            if not result["success"]:
                return result
            
            files = result["files"]
            
            # Filter for manifest files
            for file in files:
                if file["name"] in manifest_filenames:
                    found_manifests.append({
                        "name": file["name"],
                        "path": file["path"],
                        "type": self._get_manifest_type(file["name"])
                    })
            
            return {
                "success": True,
                "manifests": found_manifests
            }
            
        except Exception as e:
            logger.error(
                f"Failed to find manifest files",
                repo=repo,
                error=str(e)
            )
            
            return {
                "success": False,
                "error": f"Failed to find manifests: {str(e)}"
            }
    
    def create_branch(
        self,
        repo: str,
        branch_name: str,
        from_branch: str = "main",
    ) -> dict[str, str | bool]:
        """
        Create a new branch off an existing branch.

        Args:
            repo: Repository in format "owner/repo"
            branch_name: Name for the new branch (e.g. "fix/cve-2021-44228-20240224")
            from_branch: Source branch (default: "main")

        Returns:
            Dictionary with:
                - success: bool
                - branch: str (new branch name, if success=True)
                - sha: str (source SHA used, if success=True)
                - error: str (if success=False)
        """
        if "/" not in repo:
            return {"success": False, "error": f"Invalid repo format: '{repo}'"}

        try:
            sha = self.client.get_branch_sha(repo, from_branch)
            self.client.create_branch(repo, branch_name, sha)
            logger.info(
                "GitHub Tool: branch created",
                repo=repo,
                branch=branch_name,
                from_branch=from_branch,
            )
            return {"success": True, "branch": branch_name, "sha": sha}
        except Exception as e:
            logger.error(
                "Failed to create branch",
                repo=repo,
                branch_name=branch_name,
                from_branch=from_branch,
                error=str(e),
            )
            return {"success": False, "error": f"Failed to create branch '{branch_name}': {e}"}

    def update_file(
        self,
        repo: str,
        file_path: str,
        content: str,
        commit_message: str,
        branch: str,
    ) -> dict[str, str | bool]:
        """
        Create or update a file on a branch.

        Automatically fetches the existing file SHA so callers do not
        need to track it separately.

        Args:
            repo: Repository in format "owner/repo"
            file_path: Path to the file relative to repo root
            content: New full file content (plain text)
            commit_message: Git commit message
            branch: Branch to commit to

        Returns:
            Dictionary with:
                - success: bool
                - sha: str (new blob SHA, if success=True)
                - error: str (if success=False)
        """
        if "/" not in repo:
            return {"success": False, "error": f"Invalid repo format: '{repo}'"}

        try:
            existing_sha = self.client.get_file_sha(repo, file_path, branch)
            new_sha = self.client.create_or_update_file(
                repo=repo,
                file_path=file_path,
                message=commit_message,
                content=content,
                branch=branch,
                existing_sha=existing_sha,
            )
            logger.info(
                "GitHub Tool: file updated",
                repo=repo,
                file_path=file_path,
                branch=branch,
            )
            return {"success": True, "sha": new_sha}
        except Exception as e:
            logger.error(
                "Failed to update file",
                repo=repo,
                file_path=file_path,
                branch=branch,
                error=str(e),
            )
            return {"success": False, "error": f"Failed to update '{file_path}': {e}"}

    def create_pull_request(
        self,
        repo: str,
        title: str,
        body: str,
        head: str,
        base: str = "main",
    ) -> dict[str, str | int | bool]:
        """
        Open a pull request.

        Args:
            repo: Repository in format "owner/repo"
            title: PR title
            body: PR body (Markdown)
            head: Source branch name
            base: Target branch (default: "main")

        Returns:
            Dictionary with:
                - success: bool
                - number: int (PR number, if success=True)
                - url: str (PR HTML URL, if success=True)
                - error: str (if success=False)
        """
        if "/" not in repo:
            return {"success": False, "error": f"Invalid repo format: '{repo}'"}

        try:
            pr_data = self.client.create_pull_request(
                repo=repo, title=title, body=body, head=head, base=base
            )
            return {"success": True, **pr_data}
        except Exception as e:
            logger.error(
                "Failed to create pull request",
                repo=repo,
                head=head,
                base=base,
                error=str(e),
            )
            return {"success": False, "error": f"Failed to create PR: {e}"}

    def _get_manifest_type(self, filename: str) -> str:
        """Map manifest filename to ecosystem type."""
        manifest_types = {
            "pom.xml": "maven",
            "build.gradle": "gradle",
            "build.gradle.kts": "gradle",
            "package.json": "npm",
            "requirements.txt": "pip",
            "Cargo.toml": "rust",
            "go.mod": "go"
        }
        return manifest_types.get(filename, "unknown")
    
    def close(self) -> None:
        """Close the underlying GitHub client."""
        self.client.close()
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, *args):
        """Context manager support."""
        self.close()