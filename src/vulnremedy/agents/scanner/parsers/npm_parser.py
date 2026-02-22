"""
npm Parser — Extracts dependencies from package.json files.

package.json is JSON, so parsing is straightforward. The complexity here is
in correctly mapping dependency sections to scopes and handling npm's flexible
version string formats (exact, range, tag, URL, git) without trying to resolve
them — version strings are stored as declared.

Handles:
  - dependencies         → runtime deps, scope=None
  - devDependencies      → dev-only deps, scope="dev"
  - peerDependencies     → peer requirements, scope="peer"
  - optionalDependencies → optional deps, scope="optional"

Version string formats kept as-is (not resolved):
  - Exact:   "4.17.21"
  - Range:   "^4.17.0", "~1.2.5", ">=16.8.0", "1.x"
  - Tag:     "latest", "next", "beta"
  - Wildcard: "*"
  - URL/git: "github:user/repo", "file:../local-pkg" (kept, flagged in log)

Does NOT parse:
  - bundledDependencies / bundleDependencies (array of names, no versions)
  - workspaces (monorepo sub-packages — out of scope for Week 3)

Architectural note:
    No group_id on npm packages — npm coordinates are a flat name only.
    The CVE Analyst matches on package_name against NVD/OSV records.
"""

from __future__ import annotations

import json
from typing import Optional

from vulnremedy.agents.scanner.parsers.base import DependencyParser
from vulnremedy.models.cve import Ecosystem
from vulnremedy.models.dependency import Dependency
from vulnremedy.utils.logging import logger

# Sections to parse and their corresponding scope values.
# Order matters: dependencies first so runtime packages are seen before
# devDependencies in case the same package appears in both sections.
_SECTIONS: list[tuple[str, Optional[str]]] = [
    ("dependencies", None),
    ("devDependencies", "dev"),
    ("peerDependencies", "peer"),
    ("optionalDependencies", "optional"),
]

# Version prefixes that indicate a non-registry source (URL, git, local path).
# We still keep these but log them so the CVE Analyst knows version matching
# against NVD/OSV records will not be reliable for these entries.
_NON_REGISTRY_PREFIXES = (
    "file:",
    "link:",
    "github:",
    "gitlab:",
    "bitbucket:",
    "git+",
    "git://",
    "http://",
    "https://",
)


class NpmParser(DependencyParser):
    """
    Parses npm package.json files and extracts declared dependencies.

    Usage:
        parser = NpmParser()
        deps = parser.parse(open("package.json").read())
        for dep in deps:
            print(dep.package_name, dep.version, dep.scope)
    """

    @property
    def ecosystem(self) -> str:
        return "npm"

    def parse(
        self, content: str, manifest_file: str = "package.json"
    ) -> list[Dependency]:
        """
        Parse package.json content and return Dependency objects.

        Args:
            content:       Raw package.json content as a string.
            manifest_file: Source filename stored on each Dependency.

        Returns:
            List of Dependency objects. Empty list on any failure.
        """
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            self._log_parse_error(exc, context=manifest_file)
            return []

        if not isinstance(data, dict):
            logger.warning(
                "Skipping package.json — root is not a JSON object",
                manifest_file=manifest_file,
            )
            return []

        dependencies: list[Dependency] = []
        # Track (name, scope) pairs to suppress exact duplicates across sections.
        # Same package in both dependencies and devDependencies is kept because
        # different scopes carry different meaning for the CVE Analyst.
        seen: set[tuple[str, Optional[str]]] = set()

        for section, scope in _SECTIONS:
            section_data = data.get(section)
            if not section_data:
                continue

            if not isinstance(section_data, dict):
                logger.warning(
                    "Skipping non-dict dependency section",
                    section=section,
                    manifest_file=manifest_file,
                )
                continue

            for package_name, version_raw in section_data.items():
                dep = self._parse_entry(
                    package_name=package_name,
                    version_raw=version_raw,
                    scope=scope,
                    manifest_file=manifest_file,
                )
                if dep is None:
                    continue

                key = (dep.package_name, dep.scope)
                if key not in seen:
                    seen.add(key)
                    dependencies.append(dep)

        self._log_parse_success(len(dependencies))
        return dependencies

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_entry(
        self,
        package_name: str,
        version_raw: object,
        scope: Optional[str],
        manifest_file: str,
    ) -> Optional[Dependency]:
        """
        Validate and build a single Dependency from a package.json entry.

        Returns None (with a warning) for entries that cannot be represented
        as a Dependency — e.g. the version field is not a string.
        """
        if not isinstance(version_raw, str):
            logger.warning(
                "Skipping npm dependency with non-string version",
                package_name=package_name,
                version_type=type(version_raw).__name__,
                manifest_file=manifest_file,
            )
            return None

        version = version_raw.strip()

        if not version:
            logger.warning(
                "Skipping npm dependency with empty version",
                package_name=package_name,
                manifest_file=manifest_file,
            )
            return None

        if any(version.startswith(prefix) for prefix in _NON_REGISTRY_PREFIXES):
            logger.info(
                "npm dependency has non-registry version (URL/git/local) — "
                "CVE version matching will be unreliable",
                package_name=package_name,
                version=version,
                manifest_file=manifest_file,
            )

        return Dependency(
            package_name=package_name,
            version=version,
            ecosystem=Ecosystem.NPM,
            group_id=None,
            manifest_file=manifest_file,
            scope=scope,
        )
