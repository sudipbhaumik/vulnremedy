"""
Scanner Agent — Discovers and parses dependency manifests.

Orchestrates:
1. Manifest discovery via GitHubTool
2. Content fetching per manifest
3. Parser selection based on ecosystem type
4. Dependency collection and deduplication
"""

from __future__ import annotations

from typing import Optional

from vulnremedy.agents.scanner.parsers.gradle_parser import GradleParser
from vulnremedy.agents.scanner.parsers.maven_parser import MavenParser
from vulnremedy.agents.scanner.parsers.npm_parser import NpmParser
from vulnremedy.agents.scanner.parsers.pip_parser import PipParser
from vulnremedy.models.dependency import Dependency
from vulnremedy.tools.github.tool import GitHubTool
from vulnremedy.utils.logging import logger


class ScannerAgent:
    """
    Scans a GitHub repository for dependency manifests and extracts
    all declared dependencies using ecosystem-specific parsers.
    """

    def __init__(self, github_tool: Optional[GitHubTool] = None) -> None:
        self.github_tool = github_tool or GitHubTool()

        # Parser registry — keyed by manifest type returned from GitHubTool
        self.parsers = {
            "maven": MavenParser(),
            "gradle": GradleParser(),
            "npm": NpmParser(),
            "pip": PipParser(),
        }

    def scan(self, repo: str, branch: str = "main") -> dict:
        """
        Scan a repository for all dependencies declared in manifest files.

        Args:
            repo:   Repository in format "owner/repo" (e.g., "apache/log4j")
            branch: Branch name (default: "main")

        Returns:
            Dictionary with:
                - success:           bool
                - dependencies:      list[Dependency]
                - manifests_scanned: list[str]
                - errors:            list[str]
        """
        logger.info("Starting scan", repo=repo, branch=branch)

        all_dependencies: list[Dependency] = []
        manifests_scanned: list[str] = []
        errors: list[str] = []

        # Step 1: Discover manifest files in the repository root
        result = self.github_tool.find_manifest_files(repo, branch)
        if not result["success"]:
            return {
                "success": False,
                "dependencies": [],
                "manifests_scanned": [],
                "errors": [result["error"]],
            }

        manifests = result["manifests"]
        logger.info(
            f"Found {len(manifests)} manifests",
            manifests=[m["name"] for m in manifests],
        )

        # Step 2: Process each manifest
        for manifest in manifests:
            manifest_name: str = manifest["name"]
            manifest_type: str = manifest["type"]

            # Skip manifests with no registered parser
            if manifest_type not in self.parsers:
                logger.warning(
                    "No parser available for manifest type",
                    manifest=manifest_name,
                    type=manifest_type,
                )
                continue

            parser = self.parsers[manifest_type]

            try:
                # Fetch raw content from GitHub
                content_result = self.github_tool.fetch_file(
                    repo, manifest["path"], branch
                )
                if not content_result["success"]:
                    errors.append(
                        f"Failed to fetch {manifest_name}: {content_result['error']}"
                    )
                    continue

                content: str = content_result["content"]

                # Parse into Dependency objects
                dependencies = parser.parse(content)
                all_dependencies.extend(dependencies)
                manifests_scanned.append(manifest_name)

                logger.info(
                    "Processed manifest",
                    manifest=manifest_name,
                    dependencies_found=len(dependencies),
                )

            except Exception as exc:
                logger.error(
                    "Error processing manifest",
                    manifest=manifest_name,
                    error=str(exc),
                )
                errors.append(f"Error in {manifest_name}: {str(exc)}")
                continue  # Don't crash the pipeline — process remaining manifests

        # Step 3: Deduplicate by fully_qualified_name + version
        seen: set[str] = set()
        unique_deps: list[Dependency] = []

        for dep in all_dependencies:
            key = f"{dep.fully_qualified_name}@{dep.version}"
            if key not in seen:
                seen.add(key)
                unique_deps.append(dep)

        logger.info(
            "Deduplication complete",
            total=len(all_dependencies),
            unique=len(unique_deps),
        )

        # Step 4: Return structured output
        return {
            "success": True,
            "dependencies": unique_deps,
            "manifests_scanned": manifests_scanned,
            "errors": errors,
        }
