"""
CVE Analyst Agent — Matches dependencies to CVEs using RAG.

Uses hybrid retrieval to find relevant CVE information for each dependency,
then creates Finding objects for confirmed vulnerability matches.

Pipeline:
    list[Dependency] → RAG queries → CVE matching → list[Finding]
"""

from __future__ import annotations

import re
from typing import Optional
from uuid import UUID, uuid4

from vulnremedy.models.cve import CVERecord, CVSSScore, Severity
from vulnremedy.models.dependency import Dependency
from vulnremedy.models.finding import Finding, FindingStatus
from vulnremedy.rag.embeddings.embedding_service import EmbeddingService
from vulnremedy.rag.retrieval.retriever import HybridRetriever
from vulnremedy.rag.retrieval.vector_store import VectorStore
from vulnremedy.utils.logging import logger

# Minimum blended RAG score to consider a result worth evaluating.
# Results below this threshold are discarded unless a version match is found.
_MIN_SCORE_THRESHOLD = 0.3

# Representative CVSS numeric scores per severity band.
# Used to construct a minimal CVSSScore from the RAG metadata severity string,
# since the vector store stores severity label rather than the raw score.
_SEVERITY_TO_SCORE: dict[str, float] = {
    "critical": 9.5,
    "high": 8.0,
    "medium": 5.5,
    "low": 2.0,
    "none": 0.0,
}


class CVEAnalystAgent:
    """
    Analyzes dependencies for CVE matches using RAG-powered hybrid retrieval.

    Queries the CVE knowledge base for each dependency and creates
    Finding objects for confirmed vulnerability matches.

    Usage:
        analyst = CVEAnalystAgent()
        result = analyst.analyze(
            dependencies=scan_result["dependencies"],
            scan_id=my_scan_id,
            repository="github.com/org/service",
            branch="main",
        )
        findings = result["findings"]
    """

    def __init__(self) -> None:
        self.vector_store = VectorStore()
        self.embedding_service = EmbeddingService()
        self.retriever = HybridRetriever(
            self.vector_store,
            self.embedding_service,
        )

    def analyze(
        self,
        dependencies: list[Dependency],
        scan_id: Optional[UUID] = None,
        repository: str = "unknown",
        branch: str = "main",
    ) -> dict:
        """
        Analyze dependencies for CVE matches.

        Args:
            dependencies: List of Dependency objects from the Scanner Agent.
            scan_id:      UUID correlating this analysis to a scan record.
                          Auto-generated if not provided.
            repository:   Repository identifier e.g. "github.com/org/service".
            branch:       Branch that was scanned.

        Returns:
            Dictionary with:
                - success:               bool
                - findings:              list[Finding]
                - dependencies_analyzed: int
                - vulnerabilities_found: int
                - errors:                list[str]
        """
        scan_id = scan_id or uuid4()

        logger.info(
            "Starting CVE analysis",
            dependencies=len(dependencies),
            repository=repository,
            branch=branch,
            scan_id=str(scan_id),
        )

        findings: list[Finding] = []
        errors: list[str] = []

        for dep in dependencies:
            try:
                dep_findings = self._analyze_dependency(
                    dep=dep,
                    scan_id=scan_id,
                    repository=repository,
                    branch=branch,
                )
                findings.extend(dep_findings)

            except Exception as exc:
                logger.error(
                    "Error analyzing dependency",
                    package=dep.fully_qualified_name,
                    version=dep.version,
                    error=str(exc),
                )
                errors.append(
                    f"Error analyzing {dep.fully_qualified_name}@{dep.version}: {exc}"
                )

        logger.info(
            "CVE analysis complete",
            dependencies_analyzed=len(dependencies),
            findings=len(findings),
            errors=len(errors),
        )

        return {
            "success": True,
            "findings": findings,
            "dependencies_analyzed": len(dependencies),
            "vulnerabilities_found": len(findings),
            "errors": errors,
        }

    # ─── Private orchestration ───────────────────────────────────────────────

    def _analyze_dependency(
        self,
        dep: Dependency,
        scan_id: UUID,
        repository: str,
        branch: str,
    ) -> list[Finding]:
        """Query RAG and create findings for a single dependency."""
        chunks = self._query_for_dependency(dep)

        if not chunks:
            logger.debug(
                "No RAG results for dependency",
                package=dep.fully_qualified_name,
                version=dep.version,
            )
            return []

        logger.info(
            "RAG results for dependency",
            package=dep.fully_qualified_name,
            version=dep.version,
            results=len(chunks),
        )

        # Group chunks by cve_id — one CVE may produce multiple chunks
        chunks_by_cve: dict[str, list[dict]] = {}
        for chunk in chunks:
            cve_id = chunk.get("metadata", {}).get("cve_id", "")
            if not cve_id:
                continue
            chunks_by_cve.setdefault(cve_id, []).append(chunk)

        findings: list[Finding] = []

        for cve_id, cve_chunks in chunks_by_cve.items():
            best_chunk = max(cve_chunks, key=lambda c: c["blended_score"])
            score = best_chunk["blended_score"]
            version_match = self._is_version_match(dep.version, best_chunk["text"])

            # Skip very low-confidence results with no version signal
            if score < _MIN_SCORE_THRESHOLD and not version_match:
                logger.debug(
                    "Skipping low-confidence result",
                    cve_id=cve_id,
                    score=score,
                    package=dep.fully_qualified_name,
                )
                continue

            finding = self._create_finding(
                dep=dep,
                cve_chunks=cve_chunks,
                best_chunk=best_chunk,
                scan_id=scan_id,
                repository=repository,
                branch=branch,
            )

            if finding:
                findings.append(finding)
                logger.info(
                    "Finding created",
                    cve_id=cve_id,
                    package=dep.fully_qualified_name,
                    version=dep.version,
                    severity=finding.severity.value,
                    score=score,
                    version_match=version_match,
                )

        return findings

    def _query_for_dependency(self, dep: Dependency) -> list[dict]:
        """Query RAG for CVEs affecting this dependency."""
        query = f"{dep.fully_qualified_name} {dep.version} vulnerability"

        # Filter to ecosystem-relevant CVEs only
        filters = {"ecosystems": [dep.ecosystem.value]}

        logger.debug(
            "Querying RAG",
            package=dep.fully_qualified_name,
            version=dep.version,
            ecosystem=dep.ecosystem.value,
        )

        return self.retriever.retrieve(
            query=query,
            top_k=5,
            filters=filters,
        )

    def _create_finding(
        self,
        dep: Dependency,
        cve_chunks: list[dict],
        best_chunk: dict,
        scan_id: UUID,
        repository: str,
        branch: str,
    ) -> Optional[Finding]:
        """Build a Finding from a dependency and its matching CVE chunks."""
        metadata = best_chunk.get("metadata", {})
        cve_id = metadata.get("cve_id", "")

        # Guard: CVERecord enforces pattern ^CVE-\d{4}-\d{4,}$
        if not re.match(r"^CVE-\d{4}-\d{4,}$", cve_id):
            logger.warning("Invalid CVE ID format, skipping", cve_id=cve_id)
            return None

        severity_str = metadata.get("severity", "unknown").lower()
        severity = self._parse_severity(severity_str)
        source = metadata.get("source", "unknown")

        # Build description: prefer dedicated description chunks
        description = self._extract_description(cve_chunks, best_chunk)

        # Construct CVSSScore from the severity label we have in metadata.
        # The numeric score is the midpoint of the severity band — sufficient
        # for routing decisions downstream. Week 4 can enrich with real scores.
        cvss: Optional[CVSSScore] = None
        if severity_str in _SEVERITY_TO_SCORE:
            cvss = CVSSScore(
                version="3.1",
                score=_SEVERITY_TO_SCORE[severity_str],
                severity=severity,
            )

        cve_record = CVERecord(
            cve_id=cve_id,
            description=description,
            cvss=cvss,
            source=source,
        )

        # Try to extract the patched version from CVE text
        all_text = " ".join(c["text"] for c in cve_chunks)
        fixed_version = self._extract_fixed_version(all_text)

        affected_dep = dep.to_affected_dependency(
            fixed_version=fixed_version,
            manifest_line=None,
        )

        return Finding(
            scan_id=scan_id,
            repository=repository,
            branch=branch,
            cve=cve_record,
            affected_dependency=affected_dep,
            severity=severity,
            status=FindingStatus.OPEN,
        )

    # ─── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _is_version_match(dep_version: str, cve_chunk_text: str) -> bool:
        """
        Check if dependency version appears verbatim in CVE chunk text.

        Simple substring heuristic — sufficient for portfolio demo.
        TODO Week 4: Replace with semantic version range matching using
        packaging.version.parse() to handle ranges like '>= 2.0, < 2.17.1'.
        """
        return dep_version in cve_chunk_text

    @staticmethod
    def _parse_severity(severity_str: str) -> Severity:
        """Map severity string from RAG metadata to Severity enum."""
        try:
            return Severity(severity_str)
        except ValueError:
            return Severity.UNKNOWN

    @staticmethod
    def _extract_description(cve_chunks: list[dict], best_chunk: dict) -> str:
        """
        Return the most informative description text from retrieved chunks.

        Prefers chunks with chunk_type == 'description'; falls back to the
        highest-scoring chunk's text.
        """
        for chunk in cve_chunks:
            if chunk.get("metadata", {}).get("chunk_type") == "description":
                return chunk["text"]
        return best_chunk["text"]

    @staticmethod
    def _extract_fixed_version(text: str) -> Optional[str]:
        """
        Extract the patched version from CVE text using regex heuristics.

        Looks for patterns like:
            - "fixed in 2.17.1"
            - "upgrade to 2.17.1"
            - "version 2.17.1 resolves"
        """
        patterns = [
            r"fixed in (\d+\.\d+\.?\d*)",
            r"upgrade to (\d+\.\d+\.?\d*)",
            r"version (\d+\.\d+\.?\d*) resolves",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
