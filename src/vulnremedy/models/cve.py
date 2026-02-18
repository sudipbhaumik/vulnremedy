"""
CVE Domain Models — Core vulnerability data contracts.

These models represent the canonical shape of CVE data inside VulnRemedy,
regardless of which external source it came from (NVD, OSV, GitHub Advisory).

Architectural note:
    External API responses are NEVER passed raw through the system.
    They are always parsed into these models at the ingestion boundary.
    This decouples the rest of the system from external API schema changes.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

class Ecosystem(str, Enum):
    """
    Package ecosystems VulnRemedy supports.
    
    Using an Enum prevents typos like 'Maven' vs 'maven' causing missed matches.
    String enum allows direct comparison: ecosystem == "maven" works naturally.
    """
    MAVEN = "maven"
    NPM = "npm"
    PYPI = "pypi"
    GO = "go"
    RUST = "rust"
    NUGET = "nuget"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    """
    CVSS severity bands.
    
    Derived from CVSS score — not free text from the source API.
    We normalise all sources to this enum so agents can make 
    severity-based routing decisions reliably.
    """
    CRITICAL = "critical"   # CVSS 9.0 - 10.0
    HIGH = "high"           # CVSS 7.0 - 8.9
    MEDIUM = "medium"       # CVSS 4.0 - 6.9
    LOW = "low"             # CVSS 0.1 - 3.9
    NONE = "none"           # CVSS 0.0
    UNKNOWN = "unknown"     # Score not available

class CVSSScore(BaseModel):
    """
    CVSS scoring data.
    
    Separated from CVERecord because CVSS v2, v3, and v4 coexist
    in the wild. We store the most recent available version.
    """
    version: str = Field(
        description="CVSS version e.g. '3.1', '4.0'"
    )
    score: float = Field(
        ge=0.0,
        le=10.0,
        description="Numeric score between 0.0 and 10.0"
    )
    vector: Optional[str] = Field(
        default=None,
        description="CVSS vector string e.g. CVSS:3.1/AV:N/AC:L/..."
    )
    severity: Severity = Field(
        description="Severity band derived from score"
    )

    @field_validator("severity", mode="before")
    @classmethod
    def derive_severity_from_score(cls, v: str, info) -> str:
        """
        If severity is not explicitly provided,
        derive it from the score automatically.
        
        This handles sources that provide score but not severity band.
        """
        if v and v != Severity.UNKNOWN:
            return v
        score = info.data.get("score", 0.0)
        if score >= 9.0:
            return Severity.CRITICAL
        elif score >= 7.0:
            return Severity.HIGH
        elif score >= 4.0:
            return Severity.MEDIUM
        elif score > 0.0:
            return Severity.LOW
        return Severity.NONE

class AffectedPackage(BaseModel):
    """
    A specific package version range affected by a CVE.
    
    One CVE can affect multiple packages across multiple ecosystems.
    For example, a vulnerability in a shared C library might affect
    both the Python wrapper and the Node.js wrapper as separate packages.
    """
    ecosystem: Ecosystem
    package_name: str = Field(
        description="Canonical package name e.g. 'log4j-core'"
    )
    group_id: Optional[str] = Field(
        default=None,
        description="Maven group ID e.g. 'org.apache.logging.log4j'. Only relevant for Maven ecosystem."
    )
    affected_versions: list[str] = Field(
        default_factory=list,
        description="Version ranges affected e.g. ['>= 2.0', '< 2.17.1']"
    )
    fixed_version: Optional[str] = Field(
        default=None,
        description="Earliest version with the fix applied"
    )

    @property
    def fully_qualified_name(self) -> str:
        """
        Returns Maven-style fully qualified name where applicable.
        
        e.g. 'org.apache.logging.log4j:log4j-core'
        
        For other ecosystems returns package_name directly.
        This is used when matching against dependency manifests.
        """
        if self.group_id:
            return f"{self.group_id}:{self.package_name}"
        return self.package_name

class CVERecord(BaseModel):
    """
    Canonical CVE record inside VulnRemedy.

    Architectural note:
        This is NOT a direct mapping of NVD or OSV API response.
        It is a normalised internal representation that both sources
        map INTO. This protects the rest of the system from
        external schema changes — only the ingestion layer changes
        if NVD updates their API.
    """
    cve_id: str = Field(
        description="CVE identifier e.g. 'CVE-2021-44228'",
        pattern=r"^CVE-\d{4}-\d{4,}$"
    )
    description: str = Field(
        description="Human-readable vulnerability description"
    )
    cvss: Optional[CVSSScore] = Field(
        default=None,
        description="CVSS scoring — None if not yet scored"
    )
    affected_packages: list[AffectedPackage] = Field(
        default_factory=list
    )
    published_at: Optional[datetime] = Field(
        default=None,
        description="When this CVE was first published"
    )
    last_modified_at: Optional[datetime] = Field(
        default=None,
        description="When this CVE record was last updated"
    )
    references: list[str] = Field(
        default_factory=list,
        description="URLs to advisories, patches, write-ups"
    )
    source: str = Field(
        default="unknown",
        description="Which database this record came from: 'nvd', 'osv', 'github'"
    )

    @property
    def severity(self) -> Severity:
        """
        Convenience accessor — agents use this, not cvss.severity directly.
        
        Returns UNKNOWN if CVE has not been scored yet.
        """
        if self.cvss:
            return self.cvss.severity
        return Severity.UNKNOWN

    @property
    def is_critical_or_high(self) -> bool:
        """
        Used by Orchestrator for priority routing decisions.
        
        Critical and High severity findings get routed to fast-track remediation.
        """
        return self.severity in (Severity.CRITICAL, Severity.HIGH)

    def affects_package(
        self,
        package_name: str,
        ecosystem: Ecosystem
    ) -> bool:
        """
        Check if this CVE affects a specific package.
        
        Used by CVE Analyst agent during scanning — given a package
        from a dependency manifest, does this CVE affect it?
        
        Case-insensitive match on package name.
        """
        return any(
            p.package_name.lower() == package_name.lower()
            and p.ecosystem == ecosystem
            for p in self.affected_packages
        )