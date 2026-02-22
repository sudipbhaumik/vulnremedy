"""
pip Parser — Extracts dependencies from requirements.txt files.

requirements.txt is plain text with one package specification per line.
Unlike XML/JSON parsers, this must handle a broad set of pip-specific syntax
before hitting the actual package data.

PEP 508 package specification format:
    name [extras] [version_spec] [; env_marker]

Handles:
  - Pinned:        package==1.0.0
  - Range:         package>=1.0,<2.0   (kept as full constraint string)
  - Compatible:    package~=1.4.2
  - Unpinned:      package             (version stored as "*")
  - Extras:        package[extra,extra2]==1.0.0  (extras stripped from name)
  - Env markers:   package==1.0 ; python_version >= "3.8"  (marker stripped)
  - Inline comments: package==1.0  # some note  (stripped)
  - Line continuations: backslash-newline joined before parsing
  - Hash options:  package==1.0 --hash=sha256:abc  (hash stripped)

Skips (with debug log):
  - Blank lines and pure comment lines  (# ...)
  - pip options and flags               (-r, -c, --index-url, -e, ...)
  - URL requirements                    (http://, https://, git+...)

Version representation:
  - Exact pin  (==1.0.0):  stored as "1.0.0"  (operator stripped)
  - Any other: (>=1.0):    stored as-is        (e.g. ">=1.0,<2.0")
  - Unpinned:              stored as "*"

Architectural note:
    pip has no scope concept in a single requirements.txt — scope is None.
    Projects that use separate files (requirements-dev.txt) can pass the
    filename via manifest_file and the Scanner Agent can set scope from context.
"""

from __future__ import annotations

import re
from typing import Optional

from vulnremedy.agents.scanner.parsers.base import DependencyParser
from vulnremedy.models.cve import Ecosystem
from vulnremedy.models.dependency import Dependency
from vulnremedy.utils.logging import logger

# ---------------------------------------------------------------------------
# PEP 508 package name regex
# Letters, digits, hyphens, underscores, dots. Must start and end with
# alphanumeric. Normalise hyphens/underscores/dots to a canonical form is
# intentionally NOT done here — CVE Analyst handles name normalisation.
# ---------------------------------------------------------------------------
_NAME_RE = r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"

# Extras: [extra1,extra2]
_EXTRAS_RE = r"(?:\[[^\]]*\])?"

# Version specifier operators
_OP_RE = r"(?:===|~=|==|!=|>=|<=|>|<)"

# Full package line pattern (VERBOSE for readability).
# Groups: name, extras (optional), spec (optional), marker (optional)
_PKG_LINE = re.compile(
    rf"""
    ^
    (?P<name>{_NAME_RE})          # package name
    (?P<extras>{_EXTRAS_RE})      # optional extras e.g. [security]
    \s*
    (?P<spec>                     # optional version specifier(s)
        (?:{_OP_RE}\s*[^\s;,#\\]+)   # first constraint
        (?:\s*,\s*{_OP_RE}\s*[^\s;,#\\]+)*  # additional constraints
    )?
    \s*
    (?:;[^#\\]*)?                 # optional environment marker
    (?:\s+--\S+)*                 # optional pip inline options e.g. --hash=sha256:...
    (?:\s*\#.*)?                  # optional inline comment
    $
    """,
    re.VERBOSE,
)

# Lines that must be skipped entirely.
_SKIP_PREFIXES = (
    "#",           # comment line
    "-",           # pip option or flag (-r, -c, -e, --index-url, ...)
    "http://",     # URL requirement
    "https://",    # URL requirement
    "git+",        # git requirement
)


class PipParser(DependencyParser):
    """
    Parses pip requirements.txt files and extracts declared packages.

    Usage:
        parser = PipParser()
        deps = parser.parse(open("requirements.txt").read())
        for dep in deps:
            print(dep.package_name, dep.version)
    """

    @property
    def ecosystem(self) -> str:
        return "pip"

    def parse(
        self, content: str, manifest_file: str = "requirements.txt"
    ) -> list[Dependency]:
        """
        Parse requirements.txt content and return Dependency objects.

        Args:
            content:       Raw file content as a string.
            manifest_file: Source filename stored on each Dependency.

        Returns:
            List of Dependency objects. Empty list on any failure.
        """
        try:
            lines = self._preprocess(content)
            dependencies: list[Dependency] = []
            # Normalise names to lowercase for deduplication — pip is
            # case-insensitive (PEP 503).
            seen: set[str] = set()

            for line in lines:
                dep = self._parse_line(line, manifest_file)
                if dep is None:
                    continue
                normalised = dep.package_name.lower().replace("-", "_")
                if normalised not in seen:
                    seen.add(normalised)
                    dependencies.append(dep)

            self._log_parse_success(len(dependencies))
            return dependencies

        except Exception as exc:
            self._log_parse_error(exc, context=manifest_file)
            return []

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, content: str) -> list[str]:
        """
        Prepare raw file content for line-by-line parsing.

        Steps:
          1. Join backslash-newline continuations into a single line.
          2. Split into lines.
          3. Strip whitespace.
          4. Discard blank lines and lines matching _SKIP_PREFIXES.
        """
        # Join line continuations: "pkg==1.0 \\\n    --hash=..." → single line
        content = re.sub(r"\\\n\s*", " ", content)

        meaningful: list[str] = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            if any(line.startswith(prefix) for prefix in _SKIP_PREFIXES):
                logger.debug(
                    "Skipping pip non-package line",
                    line=line[:60],
                )
                continue
            meaningful.append(line)

        return meaningful

    # ------------------------------------------------------------------
    # Line parsing
    # ------------------------------------------------------------------

    def _parse_line(self, line: str, manifest_file: str) -> Optional[Dependency]:
        """
        Parse a single preprocessed requirements line into a Dependency.

        Returns None for lines that don't match the PEP 508 name pattern
        (should be rare after preprocessing).
        """
        match = _PKG_LINE.match(line)
        if not match:
            logger.warning(
                "Could not parse pip requirements line",
                line=line,
                manifest_file=manifest_file,
            )
            return None

        package_name = match.group("name")
        extras = match.group("extras") or ""
        spec_raw = (match.group("spec") or "").strip()

        if extras:
            logger.debug(
                "Stripping extras from pip package name",
                package_name=package_name,
                extras=extras,
            )

        version = self._extract_version(spec_raw)

        return Dependency(
            package_name=package_name,
            version=version,
            ecosystem=Ecosystem.PYPI,
            group_id=None,
            manifest_file=manifest_file,
            scope=None,  # requirements.txt has no scope concept
        )

    # ------------------------------------------------------------------
    # Version helpers
    # ------------------------------------------------------------------

    def _extract_version(self, spec: str) -> str:
        """
        Derive a clean version string from a PEP 440 version specifier.

        Rules:
          - Empty specifier → "*" (unpinned)
          - Single exact pin (==x.y.z, no comma) → "x.y.z" (operator stripped)
          - Any other constraint → kept as-is (e.g. ">=1.0,<2.0")

        Rationale for keeping range constraints as-is: the CVE Analyst needs
        to know the constraint to determine if a fixed version satisfies it.
        Stripping to a single bound would lose information.
        """
        if not spec:
            return "*"

        # Single exact pin — strip the == operator.
        exact = re.fullmatch(r"==\s*(?P<ver>[^\s,]+)", spec)
        if exact:
            return exact.group("ver")

        return spec
