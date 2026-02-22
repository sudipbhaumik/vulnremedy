"""
Dependency Model — Represents a dependency discovered during scanning.

This is simpler than AffectedDependency because at parse time we don't yet know:
- If it's vulnerable
- What the fixed version is
- If it's direct or transitive (requires dependency resolution)

Architectural note:
    Parsers return Dependency objects.
    CVE Analyst converts Dependency → AffectedDependency when vulnerability is found.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field

from vulnremedy.models.cve import Ecosystem

# Import only for type checking to avoid circular imports
if TYPE_CHECKING:
    from vulnremedy.models.finding import AffectedDependency

class Dependency(BaseModel):
    """
    A dependency extracted from a manifest file during scanning.
    
    This is the parser output. Later stages enrich this into AffectedDependency
    when vulnerabilities are matched.
    """
    
    package_name: str = Field(
        description="Package name (e.g., 'log4j-core', 'lodash')"
    )
    
    version: str = Field(
        description="Version string as declared in manifest (e.g., '2.14.1', '^4.17.0')"
    )
    
    ecosystem: Ecosystem = Field(
        description="Package ecosystem (maven, npm, pip, etc.)"
    )
    
    group_id: Optional[str] = Field(
        default=None,
        description="Maven/Gradle group ID (e.g., 'org.apache.logging.log4j')"
    )
    
    manifest_file: str = Field(
        description="Source manifest file (e.g., 'pom.xml', 'package.json')"
    )
    
    scope: Optional[str] = Field(
        default=None,
        description="Dependency scope (e.g., 'compile', 'test', 'dev')"
    )
    
    @property
    def fully_qualified_name(self) -> str:
        """
        Returns fully qualified name.
        
        Maven/Gradle: 'group_id:package_name'
        Others: 'package_name'
        """
        if self.group_id:
            return f"{self.group_id}:{self.package_name}"
        return self.package_name
    
    def to_affected_dependency(
        self,
        fixed_version: Optional[str] = None,
        manifest_line: Optional[int] = None
    ) -> "AffectedDependency":
        """
        Convert to AffectedDependency when vulnerability is confirmed.
        
        This happens in the CVE Analyst agent.
        """
        from vulnremedy.models.finding import AffectedDependency, DependencyType
        
        return AffectedDependency(
            package_name=self.package_name,
            group_id=self.group_id,
            current_version=self.version,
            fixed_version=fixed_version,
            ecosystem=self.ecosystem,
            dependency_type=DependencyType.UNKNOWN,  # Will be resolved later
            manifest_file=self.manifest_file,
            manifest_line=manifest_line
        )