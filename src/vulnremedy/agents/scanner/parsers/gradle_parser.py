"""
Gradle Parser — Extracts dependencies from build.gradle and build.gradle.kts files.

Gradle build files are executable code (Groovy DSL or Kotlin DSL), not structured
data like XML or JSON. Regex is the only practical approach for static parsing
without executing the build.

Handles:
  Groovy DSL (build.gradle):
    - String notation:   implementation 'group:artifact:version'
    - String notation:   implementation "group:artifact:version"
    - Parens notation:   implementation('group:artifact:version')
    - Map notation:      implementation group: 'g', name: 'a', version: 'v'
    - Platform import:   implementation platform('group:artifact:version')

  Kotlin DSL (build.gradle.kts):
    - Standard notation: implementation("group:artifact:version")
    - Platform import:   implementation(platform("group:artifact:version"))
    - Kotlin interpolation: "group:artifact:$varName" or "group:artifact:${varName}"

  Both DSLs:
    - Variable interpolation resolved from ext{}/val/def declarations
    - All common configurations: implementation, api, compileOnly, runtimeOnly,
      testImplementation, testCompileOnly, testRuntimeOnly, annotationProcessor, kapt
    - BOM-managed dependencies (no version) stored with version="managed"
    - Duplicate suppression (same group:artifact seen via multiple patterns)
    - Comment stripping before regex matching (block and line comments)

Limitations (known, documented):
    - Dynamic versions ('latest.release', '+') kept as-is — not resolved
    - Multi-line dependency declarations not reliably matched
    - Deeply nested ext{} blocks or computed version strings not resolved
    - Does not execute Groovy/Kotlin — no transitive dependency resolution

Architectural note:
    Returns Dependency objects only. CVE matching happens later in CVE Analyst.
"""

from __future__ import annotations

import re
from typing import Optional

from vulnremedy.agents.scanner.parsers.base import DependencyParser
from vulnremedy.models.cve import Ecosystem
from vulnremedy.models.dependency import Dependency
from vulnremedy.utils.logging import logger

# ---------------------------------------------------------------------------
# Configuration names (Gradle dependency scopes)
# ---------------------------------------------------------------------------

_CONFIGURATIONS = (
    "implementation",
    "api",
    "compileOnly",
    "runtimeOnly",
    "testImplementation",
    "testCompileOnly",
    "testRuntimeOnly",
    "annotationProcessor",
    "kapt",
    "classpath",
)

_CONF = "|".join(_CONFIGURATIONS)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Shared flag: MULTILINE so ^ and $ match per-line; VERBOSE for readability.
_FLAGS = re.MULTILINE

# Quote character class — single or double quote.
_Q = r"""['"]"""

# Coordinate group — anything except quotes (captures group:artifact:version).
_COORDS = r"""(?P<coords>[^'"]+)"""

# Trailing comment — optional // comment at end of line.
_TRAIL = r"""(?:\s*//.*)?$"""

# ---- Groovy: space-separated string notation --------------------------------
# implementation 'group:artifact:version'
# implementation "group:artifact:version"
_GROOVY_STRING = re.compile(
    rf"""^\s*(?P<config>{_CONF})\s+{_Q}{_COORDS}{_Q}\s*{_TRAIL}""",
    _FLAGS,
)

# ---- Groovy/Kotlin: parenthesised string notation ---------------------------
# implementation('group:artifact:version')      ← Groovy with parens
# implementation("group:artifact:version")      ← Kotlin standard
_PAREN_STRING = re.compile(
    rf"""^\s*(?P<config>{_CONF})\s*\(\s*{_Q}{_COORDS}{_Q}\s*\)\s*{_TRAIL}""",
    _FLAGS,
)

# ---- Groovy: platform import (space-separated) ------------------------------
# implementation platform('group:artifact:version')
# implementation platform("group:artifact:version")
_GROOVY_PLATFORM = re.compile(
    rf"""^\s*(?P<config>{_CONF})\s+platform\({_Q}{_COORDS}{_Q}\)\s*{_TRAIL}""",
    _FLAGS,
)

# ---- Kotlin: platform import (nested parens) --------------------------------
# implementation(platform("group:artifact:version"))
_KOTLIN_PLATFORM = re.compile(
    rf"""^\s*(?P<config>{_CONF})\s*\(\s*platform\(\s*{_Q}{_COORDS}{_Q}\s*\)\s*\)\s*{_TRAIL}""",
    _FLAGS,
)

# ---- Groovy: map notation ---------------------------------------------------
# implementation group: 'org.example', name: 'artifact', version: '1.0'
# version is optional (BOM-managed)
_GROOVY_MAP = re.compile(
    rf"""
    ^\s*(?P<config>{_CONF})\s+
    group\s*:\s*{_Q}(?P<group>[^'"]+){_Q}\s*,\s*
    name\s*:\s*{_Q}(?P<name>[^'"]+){_Q}
    (?:\s*,\s*version\s*:\s*{_Q}(?P<version>[^'"]+){_Q})?
    \s*{_TRAIL}
    """,
    _FLAGS | re.VERBOSE,
)

# ---- Variable extraction (Groovy ext / Kotlin val / Groovy def) -------------
# Groovy:  def varName = "value"  |  ext.varName = "value"  |  varName = "value" (inside ext{})
# Kotlin:  val varName = "value"
_VAR_DECL = re.compile(
    rf"""(?:def|val|ext\.)\s*(\w+)\s*=\s*{_Q}([^'"]+){_Q}""",
    re.MULTILINE,
)

# Inside ext { varName = "value" } — bare assignment inside the block.
# We capture any  word = "value"  that wasn't already caught above.
_EXT_BLOCK_VAR = re.compile(
    rf"""^\s*(\w+)\s*=\s*{_Q}([^'"]+){_Q}\s*{_TRAIL}""",
    _FLAGS,
)

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class GradleParser(DependencyParser):
    """
    Parses Gradle build files (Groovy DSL and Kotlin DSL) using regex.

    Usage:
        parser = GradleParser()
        deps = parser.parse(open("build.gradle").read())
        # or
        deps = parser.parse(open("build.gradle.kts").read(), manifest_file="build.gradle.kts")
    """

    @property
    def ecosystem(self) -> str:
        return "gradle"

    def parse(
        self, content: str, manifest_file: str = "build.gradle"
    ) -> list[Dependency]:
        """
        Parse Gradle build file content and return Dependency objects.

        Args:
            content:       Raw file content as a string.
            manifest_file: Source filename stored on each Dependency.

        Returns:
            List of Dependency objects. Empty list on any failure.
        """
        try:
            clean = self._strip_comments(content)
            variables = self._extract_variables(clean)
            return self._extract_dependencies(clean, variables, manifest_file)
        except Exception as exc:
            self._log_parse_error(exc, context=manifest_file)
            return []

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _strip_comments(self, content: str) -> str:
        """
        Remove block comments (/* ... */) from content.

        Line comments (// ...) are handled per-pattern via the _TRAIL suffix
        so they are NOT stripped here — stripping them could corrupt
        string literals that contain double slashes (rare but possible).
        """
        return re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)

    def _extract_variables(self, content: str) -> dict[str, str]:
        """
        Build a variable name → value map from common Gradle variable patterns.

        Covers:
          - Groovy: def version = "1.0", ext.version = "1.0"
          - Kotlin: val version = "1.0"
          - Bare assignments inside ext { } blocks: version = "1.0"

        Returns a dict used for ${varName} / $varName interpolation.
        """
        variables: dict[str, str] = {}

        for match in _VAR_DECL.finditer(content):
            variables[match.group(1)] = match.group(2)

        # Capture bare assignments that sit inside ext { } blocks.
        # Heuristic: any line of the form  word = "value"  that was not already
        # captured by _VAR_DECL (which requires def/val/ext. prefix).
        for match in _EXT_BLOCK_VAR.finditer(content):
            name = match.group(1)
            if name not in variables:
                variables[name] = match.group(2)

        return variables

    # ------------------------------------------------------------------
    # Dependency extraction
    # ------------------------------------------------------------------

    def _extract_dependencies(
        self,
        content: str,
        variables: dict[str, str],
        manifest_file: str,
    ) -> list[Dependency]:
        """
        Apply all regex patterns against the cleaned content and collect
        unique Dependency objects.

        A seen set on (group_id, package_name) prevents duplicates when the
        same dependency is matched by both _GROOVY_STRING and _PAREN_STRING.
        """
        dependencies: list[Dependency] = []
        seen: set[tuple[str, str]] = set()

        def add(dep: Optional[Dependency]) -> None:
            if dep is None:
                return
            key = (dep.group_id or "", dep.package_name)
            if key not in seen:
                seen.add(key)
                dependencies.append(dep)

        # Platform patterns must be tried BEFORE the generic string/paren
        # patterns because platform('g:a:v') would also partially match the
        # string pattern (it would capture 'g:a:v)' with a trailing paren).
        for match in _GROOVY_PLATFORM.finditer(content):
            add(self._from_coords(
                match.group("coords"), match.group("config"),
                variables, manifest_file, is_platform=True,
            ))

        for match in _KOTLIN_PLATFORM.finditer(content):
            add(self._from_coords(
                match.group("coords"), match.group("config"),
                variables, manifest_file, is_platform=True,
            ))

        for match in _GROOVY_STRING.finditer(content):
            add(self._from_coords(
                match.group("coords"), match.group("config"),
                variables, manifest_file,
            ))

        for match in _PAREN_STRING.finditer(content):
            add(self._from_coords(
                match.group("coords"), match.group("config"),
                variables, manifest_file,
            ))

        for match in _GROOVY_MAP.finditer(content):
            add(self._from_map(
                group=match.group("group"),
                name=match.group("name"),
                version=match.group("version"),
                config=match.group("config"),
                manifest_file=manifest_file,
            ))

        self._log_parse_success(len(dependencies))
        return dependencies

    # ------------------------------------------------------------------
    # Coordinate parsing helpers
    # ------------------------------------------------------------------

    def _resolve_interpolation(
        self, raw: str, variables: dict[str, str]
    ) -> str:
        """
        Replace ${varName} and $varName placeholders with known values.

        Both Groovy `${version}` and Kotlin `$version` / `${version}` forms
        are handled. Unknown placeholders are left as-is for traceability.
        """
        def _replace(m: re.Match) -> str:
            name = m.group(1) or m.group(2)  # group 1 = braced, group 2 = bare
            resolved = variables.get(name)
            if resolved:
                logger.debug(
                    "Resolved Gradle variable",
                    placeholder=m.group(0),
                    resolved=resolved,
                )
                return resolved
            logger.warning(
                "Unresolvable Gradle variable placeholder",
                placeholder=m.group(0),
            )
            return m.group(0)

        # Match ${varName} first, then bare $varName (word chars only).
        return re.sub(r"\$\{(\w+)\}|\$(\w+)", _replace, raw)

    def _from_coords(
        self,
        coords: str,
        config: str,
        variables: dict[str, str],
        manifest_file: str,
        is_platform: bool = False,
    ) -> Optional[Dependency]:
        """
        Parse a 'group:artifact[:version]' coordinate string into a Dependency.

        Returns None if the coordinate has fewer than two parts (malformed).
        """
        coords = self._resolve_interpolation(coords.strip(), variables)
        parts = [p.strip() for p in coords.split(":")]

        if len(parts) < 2 or not parts[0] or not parts[1]:
            logger.warning(
                "Skipping malformed Gradle coordinates",
                coords=coords,
                manifest_file=manifest_file,
            )
            return None

        group_id = parts[0]
        artifact_id = parts[1]
        version = parts[2] if len(parts) > 2 else "managed"

        scope = f"{config}(platform)" if is_platform else config

        return Dependency(
            package_name=artifact_id,
            version=version,
            ecosystem=Ecosystem.GRADLE,
            group_id=group_id,
            manifest_file=manifest_file,
            scope=scope,
        )

    def _from_map(
        self,
        group: str,
        name: str,
        version: Optional[str],
        config: str,
        manifest_file: str,
    ) -> Optional[Dependency]:
        """
        Build a Dependency from Groovy map notation fields.
        """
        if not group or not name:
            logger.warning(
                "Skipping Gradle map dependency missing group or name",
                manifest_file=manifest_file,
            )
            return None

        return Dependency(
            package_name=name,
            version=version or "managed",
            ecosystem=Ecosystem.GRADLE,
            group_id=group,
            manifest_file=manifest_file,
            scope=config,
        )
