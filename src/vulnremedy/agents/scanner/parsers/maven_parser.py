"""
Maven Parser — Extracts dependencies from pom.xml files.

Handles:
  - Standard <dependencies> block
  - Maven XML namespaces (xmlns="http://maven.apache.org/POM/4.0.0")
  - All four scopes: compile (default), test, provided, runtime
  - Property interpolation for versions declared as ${property.name}
  - Malformed XML (returns empty list, logs error — never crashes)

Architectural note:
    Returns Dependency objects only. No CVE matching here.
    CVE Analyst agent handles matching later via RAG retrieval.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Optional

from vulnremedy.agents.scanner.parsers.base import DependencyParser
from vulnremedy.models.cve import Ecosystem
from vulnremedy.models.dependency import Dependency
from vulnremedy.utils.logging import logger

# Maven POM XML namespace — present when xmlns is declared in the root element
_MAVEN_NS = "http://maven.apache.org/POM/4.0.0"


def _ns(tag: str, namespace: Optional[str]) -> str:
    """
    Build a namespace-qualified tag string for ElementTree queries.

    ElementTree represents namespaced tags as '{namespace}localname'.
    If the POM has no namespace declaration we use the bare tag name.
    """
    if namespace:
        return f"{{{namespace}}}{tag}"
    return tag


class MavenParser(DependencyParser):
    """
    Parses Maven pom.xml files and extracts declared dependencies.

    Usage:
        parser = MavenParser()
        deps = parser.parse(open("pom.xml").read())
        for dep in deps:
            print(dep.fully_qualified_name, dep.version)
    """

    @property
    def ecosystem(self) -> str:
        return "maven"

    def parse(self, content: str, manifest_file: str = "pom.xml") -> list[Dependency]:
        """
        Parse pom.xml content and return a list of Dependency objects.

        Args:
            content:       Raw pom.xml file content as a string.
            manifest_file: Source filename stored on each Dependency for traceability.

        Returns:
            List of Dependency objects. Empty list on any parse failure.
        """
        try:
            root = ET.fromstring(content)
        except ET.ParseError as exc:
            self._log_parse_error(exc, context=manifest_file)
            return []

        # Detect whether the POM uses the standard Maven namespace.
        # ElementTree stores the namespace in the tag itself: '{ns}project'
        namespace: Optional[str] = None
        if root.tag.startswith("{"):
            namespace = root.tag[1: root.tag.index("}")]

        # Collect <properties> so we can resolve ${property.name} placeholders.
        properties = self._extract_properties(root, namespace)

        dependencies: list[Dependency] = []

        deps_element = root.find(_ns("dependencies", namespace))
        if deps_element is None:
            # Valid POM with no dependencies (e.g. parent POM).
            self._log_parse_success(0)
            return []

        for dep_el in deps_element.findall(_ns("dependency", namespace)):
            dep = self._parse_dependency_element(
                dep_el, namespace, properties, manifest_file
            )
            if dep is not None:
                dependencies.append(dep)

        self._log_parse_success(len(dependencies))
        return dependencies

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_properties(
        self, root: ET.Element, namespace: Optional[str]
    ) -> dict[str, str]:
        """
        Extract all <properties> entries from the POM root.

        These are used to resolve version placeholders like ${junit.version}.
        Returns a plain dict of property-name → value.
        """
        properties: dict[str, str] = {}
        props_el = root.find(_ns("properties", namespace))
        if props_el is None:
            return properties

        for prop in props_el:
            # Strip namespace prefix from the tag to get the bare property name.
            tag = prop.tag
            if tag.startswith("{"):
                tag = tag[tag.index("}") + 1:]
            if prop.text:
                properties[tag] = prop.text.strip()

        return properties

    def _resolve_version(self, raw: str, properties: dict[str, str]) -> str:
        """
        Substitute ${property.name} placeholders with values from <properties>.

        If the placeholder is not found in the properties map the raw string
        is returned unchanged — the calling code will receive e.g.
        "${junit.version}" which is still useful for traceability.
        """
        match = re.fullmatch(r"\$\{([^}]+)\}", raw.strip())
        if match:
            prop_name = match.group(1)
            resolved = properties.get(prop_name)
            if resolved:
                logger.debug(
                    "Resolved Maven property",
                    placeholder=raw,
                    resolved=resolved,
                )
                return resolved
            logger.warning(
                "Unresolvable Maven property placeholder",
                placeholder=raw,
            )
        return raw.strip()

    def _parse_dependency_element(
        self,
        dep_el: ET.Element,
        namespace: Optional[str],
        properties: dict[str, str],
        manifest_file: str,
    ) -> Optional[Dependency]:
        """
        Parse a single <dependency> element into a Dependency model.

        Returns None (and logs a warning) if required fields are missing.
        """
        def text(tag: str) -> Optional[str]:
            el = dep_el.find(_ns(tag, namespace))
            return el.text.strip() if el is not None and el.text else None

        group_id = text("groupId")
        artifact_id = text("artifactId")
        version_raw = text("version")
        scope = text("scope") or "compile"  # Maven default scope is compile

        if not group_id or not artifact_id:
            logger.warning(
                "Skipping Maven dependency missing groupId or artifactId",
                manifest_file=manifest_file,
            )
            return None

        # Version is optional in POMs that use dependency management / BOM imports.
        # We keep the dependency but mark version as "managed" so the CVE Analyst
        # knows it requires resolution from the BOM before matching.
        if version_raw:
            version = self._resolve_version(version_raw, properties)
        else:
            version = "managed"

        return Dependency(
            package_name=artifact_id,
            version=version,
            ecosystem=Ecosystem.MAVEN,
            group_id=group_id,
            manifest_file=manifest_file,
            scope=scope,
        )
