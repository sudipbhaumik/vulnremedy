"""
Parser — Convert raw API responses into CVERecord models.

Handles the messy work of normalizing NVD and OSV API schemas
into our canonical CVERecord format.

Architectural note:
    This is the boundary enforcement layer.
    Raw external data enters, validated CVERecord objects exit.
    
    If parsing fails for a CVE, we log it and skip it.
    We never let malformed data enter the system.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import ValidationError

from vulnremedy.models.cve import (
    AffectedPackage,
    CVERecord,
    CVSSScore,
    Ecosystem,
    Severity,
)
from vulnremedy.utils.logging import logger


class CVEParser:
    """
    Parses raw vulnerability data from NVD and OSV into CVERecord models.
    
    Usage:
        parser = CVEParser()
        cve = parser.parse_nvd_vulnerability(nvd_json)
    """
    
    def parse_nvd_vulnerability(self, data: dict[str, Any]) -> Optional[CVERecord]:
        """
        Parse an NVD API vulnerability response into CVERecord.
        
        NVD structure:
        {
          "cve": {
            "id": "CVE-2021-44228",
            "descriptions": [...],
            "metrics": {...},
            "configurations": [...],
            "references": [...]
          }
        }
        
        Args:
            data: One vulnerability dict from NVD API response
        
        Returns:
            CVERecord if parsing succeeds, None if it fails
        """
        try:
            cve_data = data.get("cve", {})
            cve_id = cve_data.get("id")
            
            if not cve_id:
                logger.warning("NVD vulnerability missing CVE ID, skipping")
                return None
            
            # Extract description (English preferred)
            description = self._extract_nvd_description(cve_data)
            
            # Extract CVSS score
            cvss = self._extract_nvd_cvss(cve_data)
            
            # Extract affected packages (NVD doesn't structure this well)
            # For now we'll parse what we can from configurations
            affected_packages = self._extract_nvd_affected_packages(cve_data)
            
            # Extract timestamps
            published_at = self._parse_datetime(cve_data.get("published"))
            last_modified_at = self._parse_datetime(cve_data.get("lastModified"))
            
            # Extract references
            references = [
                ref.get("url", "") 
                for ref in cve_data.get("references", [])
                if ref.get("url")
            ]
            
            # Create CVERecord with Pydantic validation
            cve_record = CVERecord(
                cve_id=cve_id,
                description=description,
                cvss=cvss,
                affected_packages=affected_packages,
                published_at=published_at,
                last_modified_at=last_modified_at,
                references=references,
                source="nvd"
            )
            
            logger.debug(f"Successfully parsed NVD CVE: {cve_id}")
            return cve_record
            
        except ValidationError as e:
            logger.error(f"Pydantic validation failed for NVD CVE: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to parse NVD vulnerability: {e}", exc_info=True)
            return None
    
    def parse_osv_vulnerability(self, data: dict[str, Any]) -> Optional[CVERecord]:
        """
        Parse an OSV.dev vulnerability response into CVERecord.
        
        OSV structure:
        {
          "id": "GHSA-xxxx-xxxx-xxxx" or "CVE-2021-44228",
          "summary": "...",
          "details": "...",
          "affected": [...],
          "severity": [...],
          "references": [...]
        }
        
        Args:
            data: One vulnerability dict from OSV API response
        
        Returns:
            CVERecord if parsing succeeds, None if it fails
        """
        try:
            vuln_id = data.get("id", "")
            
            # OSV IDs can be CVE-*, GHSA-*, or ecosystem-specific
            # We prefer CVE IDs for consistency
            cve_id = self._extract_cve_id_from_osv(data)
            
            if not cve_id:
                logger.debug(f"OSV vulnerability {vuln_id} has no CVE ID, skipping")
                return None
            
            # Extract description
            description = data.get("details") or data.get("summary") or "No description available"
            
            # Extract CVSS score from severity field
            cvss = self._extract_osv_cvss(data)
            
            # Extract affected packages
            affected_packages = self._extract_osv_affected_packages(data)
            
            # Extract timestamps
            published_at = self._parse_datetime(data.get("published"))
            last_modified_at = self._parse_datetime(data.get("modified"))
            
            # Extract references
            references = [
                ref.get("url", "")
                for ref in data.get("references", [])
                if ref.get("url")
            ]
            
            # Create CVERecord
            cve_record = CVERecord(
                cve_id=cve_id,
                description=description,
                cvss=cvss,
                affected_packages=affected_packages,
                published_at=published_at,
                last_modified_at=last_modified_at,
                references=references,
                source="osv"
            )
            
            logger.debug(f"Successfully parsed OSV vulnerability as CVE: {cve_id}")
            return cve_record
            
        except ValidationError as e:
            logger.error(f"Pydantic validation failed for OSV vulnerability: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to parse OSV vulnerability: {e}", exc_info=True)
            return None
    
    def _extract_nvd_description(self, cve_data: dict[str, Any]) -> str:
        """Extract English description from NVD CVE data."""
        descriptions = cve_data.get("descriptions", [])
        
        # Prefer English
        for desc in descriptions:
            if desc.get("lang") == "en":
                return desc.get("value", "No description available")
        
        # Fallback to first available
        if descriptions:
            return descriptions[0].get("value", "No description available")
        
        return "No description available"
    
    def _extract_nvd_cvss(self, cve_data: dict[str, Any]) -> Optional[CVSSScore]:
        """
        Extract CVSS score from NVD metrics.
        
        NVD can have multiple CVSS versions (v2, v3.0, v3.1, v4.0).
        We prefer the most recent version.
        """
        metrics = cve_data.get("metrics", {})
        
        # Try CVSS v4.0 first (newest)
        if "cvssMetricV40" in metrics:
            cvss_data = metrics["cvssMetricV40"][0].get("cvssData", {})
            return CVSSScore(
                version="4.0",
                score=cvss_data.get("baseScore", 0.0),
                vector=cvss_data.get("vectorString"),
                severity=Severity.UNKNOWN  # Will be auto-derived
            )
        
        # Try CVSS v3.1
        if "cvssMetricV31" in metrics:
            cvss_data = metrics["cvssMetricV31"][0].get("cvssData", {})
            return CVSSScore(
                version="3.1",
                score=cvss_data.get("baseScore", 0.0),
                vector=cvss_data.get("vectorString"),
                severity=Severity.UNKNOWN
            )
        
        # Try CVSS v3.0
        if "cvssMetricV30" in metrics:
            cvss_data = metrics["cvssMetricV30"][0].get("cvssData", {})
            return CVSSScore(
                version="3.0",
                score=cvss_data.get("baseScore", 0.0),
                vector=cvss_data.get("vectorString"),
                severity=Severity.UNKNOWN
            )
        
        # Try CVSS v2
        if "cvssMetricV2" in metrics:
            cvss_data = metrics["cvssMetricV2"][0].get("cvssData", {})
            return CVSSScore(
                version="2.0",
                score=cvss_data.get("baseScore", 0.0),
                vector=cvss_data.get("vectorString"),
                severity=Severity.UNKNOWN
            )
        
        # No CVSS data available
        return None
    
    def _extract_nvd_affected_packages(
        self, cve_data: dict[str, Any]
    ) -> list[AffectedPackage]:
        """
        Extract affected packages from NVD data.
        
        Note: NVD doesn't structure package data well.
        This is a simplified implementation. OSV has better package data.
        """
        # NVD's "configurations" field is complex and CPE-based
        # For this implementation, we'll return empty list
        # In production, you'd parse CPE strings into package names
        
        # TODO: Implement CPE parsing if needed
        return []
    
    def _extract_osv_cvss(self, data: dict[str, Any]) -> Optional[CVSSScore]:
        """
        Extract CVSS score from OSV severity field.
        
        OSV severity structure:
        "severity": [
          {
            "type": "CVSS_V3",
            "score": "CVSS:3.1/AV:N/AC:L/..."
          }
        ]
        """
        severity_list = data.get("severity", [])
        
        for severity_item in severity_list:
            if severity_item.get("type") in ["CVSS_V3", "CVSS_V31"]:
                vector = severity_item.get("score", "")
                
                # Parse CVSS score from vector string
                # Format: "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H"
                score = self._parse_cvss_score_from_vector(vector)
                
                if score is not None:
                    return CVSSScore(
                        version="3.1",
                        score=score,
                        vector=vector,
                        severity=Severity.UNKNOWN  # Auto-derived
                    )
        
        return None
    
    def _extract_osv_affected_packages(
        self, data: dict[str, Any]
    ) -> list[AffectedPackage]:
        """
        Extract affected packages from OSV data.
        
        OSV has excellent package-level data.
        """
        affected = data.get("affected", [])
        packages = []
        
        for item in affected:
            package_info = item.get("package", {})
            ecosystem_str = package_info.get("ecosystem", "").lower()
            package_name = package_info.get("name", "")
            
            if not package_name:
                continue
            
            # Map OSV ecosystem to our Ecosystem enum
            ecosystem = self._map_osv_ecosystem(ecosystem_str)
            
            # Extract version ranges
            ranges = item.get("ranges", [])
            affected_versions = []
            fixed_version = None
            
            for range_item in ranges:
                events = range_item.get("events", [])
                for event in events:
                    if "introduced" in event:
                        affected_versions.append(f">= {event['introduced']}")
                    if "fixed" in event:
                        fixed_version = event["fixed"]
                        affected_versions.append(f"< {event['fixed']}")
            
            # Extract group_id for Maven
            group_id = None
            if ecosystem == Ecosystem.MAVEN and ":" in package_name:
                parts = package_name.split(":")
                if len(parts) == 2:
                    group_id, package_name = parts
            
            packages.append(
                AffectedPackage(
                    ecosystem=ecosystem,
                    package_name=package_name,
                    group_id=group_id,
                    affected_versions=affected_versions,
                    fixed_version=fixed_version
                )
            )
        
        return packages
    
    def _extract_cve_id_from_osv(self, data: dict[str, Any]) -> Optional[str]:
        """
        Extract CVE ID from OSV data.
        
        OSV ID can be CVE-* itself, or CVE might be in aliases.
        """
        vuln_id = data.get("id", "")
        
        # Check if main ID is a CVE
        if vuln_id.startswith("CVE-"):
            return vuln_id
        
        # Check aliases
        aliases = data.get("aliases", [])
        for alias in aliases:
            if alias.startswith("CVE-"):
                return alias
        
        return None
    
    def _map_osv_ecosystem(self, ecosystem_str: str) -> Ecosystem:
        """Map OSV ecosystem string to our Ecosystem enum."""
        mapping = {
            "maven": Ecosystem.MAVEN,
            "pypi": Ecosystem.PYPI,
            "npm": Ecosystem.NPM,
            "go": Ecosystem.GO,
            "cargo": Ecosystem.RUST,
            "nuget": Ecosystem.NUGET,
        }
        return mapping.get(ecosystem_str, Ecosystem.UNKNOWN)
    
    def _parse_cvss_score_from_vector(self, vector: str) -> Optional[float]:
        """
        Parse numeric CVSS score from vector string.
        
        This is a simplified parser. In production, use a CVSS library.
        """
        # Vector format: "CVSS:3.1/AV:N/AC:L/..."
        # We'd need a full CVSS calculator here
        # For now, return None and rely on pre-calculated scores
        return None
    
    def _parse_datetime(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse ISO datetime string."""
        if not date_str:
            return None
        
        try:
            # Handle both formats: with and without microseconds
            if "." in date_str:
                return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            else:
                return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except Exception as e:
            logger.warning(f"Failed to parse datetime '{date_str}': {e}")
            return None