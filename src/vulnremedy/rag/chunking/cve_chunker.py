"""
CVE Chunker — Split CVE records into semantically meaningful chunks.

Implements a semantic chunking strategy where each CVE is split
by logical sections rather than arbitrary character limits.

Architectural note:
    We chunk by CVE structure (description, affected packages, fix guidance)
    rather than by token count. This produces higher-quality retrieval
    because semantically related information stays together.
    
    Each chunk gets rich metadata for filtering during retrieval.
"""

from __future__ import annotations

from typing import Any

from vulnremedy.models.cve import CVERecord
from vulnremedy.utils.logging import logger


class CVEChunk:
    """
    A single chunk from a CVE record.
    
    Contains the text content plus metadata for retrieval filtering.
    """
    
    def __init__(
        self,
        text: str,
        metadata: dict[str, Any]
    ):
        self.text = text
        self.metadata = metadata
    
    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to dictionary for storage."""
        return {
            "text": self.text,
            "metadata": self.metadata
        }


class CVEChunker:
    """
    Chunks CVE records into semantically meaningful pieces.
    
    Usage:
        chunker = CVEChunker()
        chunks = chunker.chunk_cve(cve_record)
    """
    
    def __init__(
        self,
        max_chunk_length: Optional[int] = None,
        min_chunk_length: Optional[int] = None,
        overlap: Optional[int] = None
    ):
        """
        Initialize chunker with configurable parameters.
        
        Args:
            max_chunk_length: Maximum characters per chunk (default from config)
            min_chunk_length: Minimum characters for chunk (default from config)
            overlap: Character overlap between chunks (default from config)
        """
        from vulnremedy.utils.config import settings
        
        self.max_chunk_length = max_chunk_length or settings.chunk_max_length
        self.min_chunk_length = min_chunk_length or settings.chunk_min_length
        self.overlap = overlap or settings.chunk_overlap
        
        logger.info(
            "CVE Chunker initialized",
            max_chunk_length=self.max_chunk_length,
            min_chunk_length=self.min_chunk_length,
            overlap=self.overlap
        )
    
    def chunk_cve(self, cve: CVERecord) -> list[CVEChunk]:
        """
        Chunk a CVE record into searchable pieces.
        
        Creates 1+ chunks per CVE depending on content:
        - Description chunk(s) (always present, may be split if long)
        - Affected packages chunk (if packages exist)
        - Fix guidance chunk (if references exist)
        
        Args:
            cve: CVERecord to chunk
        
        Returns:
            List of CVEChunk objects
        """
        chunks = []
        # Extract ecosystems from affected packages
        ecosystems = list(set(
            pkg.ecosystem.value 
            for pkg in cve.affected_packages 
            if pkg.ecosystem
        ))
        
        # Base metadata for all chunks from this CVE
        base_metadata = {
            "cve_id": cve.cve_id,
            "severity": cve.severity.value,
            "source": cve.source,
        }

        # Only add ecosystems if non-empty (ChromaDB doesn't allow empty lists)
        if ecosystems:
            base_metadata["ecosystems"] = ecosystems
        
        # Chunk 1+: Description (returns list now, not single chunk)
        description_chunks = self._create_description_chunk(cve, base_metadata)
        chunks.extend(description_chunks)  # ← Changed from append to extend
        
        # Chunk N: Affected Packages
        packages_chunk = self._create_packages_chunk(cve, base_metadata)
        if packages_chunk:
            chunks.append(packages_chunk)
        
        # Chunk N+1: Fix Guidance
        fix_chunk = self._create_fix_chunk(cve, base_metadata)
        if fix_chunk:
            chunks.append(fix_chunk)
        
        logger.debug(
            f"Chunked {cve.cve_id} into {len(chunks)} chunks"
        )
        
        return chunks
    
    def _create_description_chunk(
        self,
        cve: CVERecord,
        base_metadata: dict[str, Any]
    ) -> list[CVEChunk]:
        """
        Create description chunk(s).
        
        Uses recursive splitting with overlap for long descriptions.
        This ensures no information is lost when descriptions exceed max_chunk_length.
        
        Returns:
            List of CVEChunk objects (1 for short descriptions, multiple for long ones)
        """
        description = cve.description.strip()
        
        # Skip if too short
        if len(description) < self.min_chunk_length:
            logger.debug(f"{cve.cve_id}: Description too short, skipping chunk")
            return []
        
        # Add CVSS context that will appear in first chunk
        cvss_context = ""
        if cve.cvss:
            cvss_context = f"\n\nSeverity: {cve.severity.value.upper()} (CVSS {cve.cvss.score})"
        
        # If description fits in one chunk, return single chunk
        single_chunk_text = f"CVE {cve.cve_id}: {description}{cvss_context}"
        
        if len(single_chunk_text) <= self.max_chunk_length:
            metadata = {
                **base_metadata,
                "chunk_type": "description",
                "chunk_index": 0,
                "total_description_chunks": 1
            }
            return [CVEChunk(text=single_chunk_text, metadata=metadata)]
        
        # Description is too long — split recursively with overlap
        logger.info(
            f"{cve.cve_id}: Description length {len(description)} exceeds max {self.max_chunk_length}, "
            "splitting into multiple chunks"
        )
        
        chunks = []
        overlap = self.overlap  # From config
        
        # Account for the prefix length in first chunk
        first_chunk_prefix = f"CVE {cve.cve_id}: "
        available_first_chunk = self.max_chunk_length - len(first_chunk_prefix) - len(cvss_context)
        
        # First chunk
        first_chunk_text = description[:available_first_chunk]
        full_first_chunk = f"{first_chunk_prefix}{first_chunk_text}{cvss_context}"
        
        metadata = {
            **base_metadata,
            "chunk_type": "description",
            "chunk_index": 0,
            "is_continuation": False,
            # total_description_chunks will be set after we know final count
        }
        chunks.append(CVEChunk(text=full_first_chunk, metadata=metadata))
        
        # Subsequent chunks with overlap
        start_pos = available_first_chunk - overlap  # Start with overlap from first chunk
        chunk_index = 1
        continuation_prefix = f"CVE {cve.cve_id} (continued): "
        available_continuation = self.max_chunk_length - len(continuation_prefix)
        
        while start_pos < len(description):
            # Extract chunk with overlap
            end_pos = start_pos + available_continuation
            chunk_text = description[start_pos:end_pos]
            
            # Skip if remaining text is too short to be useful
            if len(chunk_text) < self.min_chunk_length:
                break
            
            full_chunk_text = f"{continuation_prefix}{chunk_text}"
            
            metadata = {
                **base_metadata,
                "chunk_type": "description",
                "chunk_index": chunk_index,
                "is_continuation": True,
            }
            chunks.append(CVEChunk(text=full_chunk_text, metadata=metadata))
            
            # Move start position forward, accounting for overlap
            start_pos = end_pos - overlap
            chunk_index += 1
        
        # Update all chunks with total count
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk.metadata["total_description_chunks"] = total_chunks
        
        logger.debug(
            f"{cve.cve_id}: Split description into {total_chunks} chunks with {overlap} char overlap"
        )
        
        return chunks
    
    def _create_packages_chunk(
        self,
        cve: CVERecord,
        base_metadata: dict[str, Any]
    ) -> CVEChunk | None:
        """
        Create affected packages chunk.
        
        Lists which packages and versions are vulnerable.
        Critical for matching against scan results.
        """
        if not cve.affected_packages:
            return None
        
        # Build text describing affected packages
        lines = [f"CVE {cve.cve_id} affects the following packages:\n"]
        
        for pkg in cve.affected_packages:
            pkg_line = f"- {pkg.fully_qualified_name} ({pkg.ecosystem.value})"
            
            if pkg.affected_versions:
                versions_str = ", ".join(pkg.affected_versions)
                pkg_line += f" versions {versions_str}"
            
            if pkg.fixed_version:
                pkg_line += f", fixed in {pkg.fixed_version}"
            
            lines.append(pkg_line)
        
        text = "\n".join(lines)
        
        if len(text) < self.min_chunk_length:
            return None
        
        # Truncate if too long
        if len(text) > self.max_chunk_length:
            text = text[:self.max_chunk_length] + "..."
        
        # Extract ecosystems for metadata filtering
        ecosystems = list(set(pkg.ecosystem.value for pkg in cve.affected_packages))
        
        metadata = {
            **base_metadata,
            "chunk_type": "affected_packages",
            "chunk_index": 1,
            "ecosystems": ecosystems,
            "package_names": [pkg.package_name for pkg in cve.affected_packages]
        }
        
        return CVEChunk(text=text, metadata=metadata)
    
    def _create_fix_chunk(
        self,
        cve: CVERecord,
        base_metadata: dict[str, Any]
    ) -> CVEChunk | None:
        """
        Create fix guidance chunk.
        
        Contains remediation information from references.
        Useful for "how do I fix this?" queries.
        """
        if not cve.references:
            return None
        
        # Build text from references
        lines = [
            f"CVE {cve.cve_id} remediation guidance:",
            ""
        ]
        
        # Add fix version info if available
        fixed_versions = [
            pkg.fixed_version 
            for pkg in cve.affected_packages 
            if pkg.fixed_version
        ]
        
        if fixed_versions:
            unique_versions = list(set(fixed_versions))
            lines.append(f"Fixed in versions: {', '.join(unique_versions)}")
            lines.append("")
        
        # Add references
        lines.append("References:")
        for ref in cve.references[:5]:  # Limit to 5 references
            lines.append(f"- {ref}")
        
        text = "\n".join(lines)
        
        if len(text) < self.min_chunk_length:
            return None
        
        # Truncate if too long
        if len(text) > self.max_chunk_length:
            text = text[:self.max_chunk_length] + "..."
        
        metadata = {
            **base_metadata,
            "chunk_type": "fix_guidance",
            "chunk_index": 2,
        }
        
        return CVEChunk(text=text, metadata=metadata)
    
    def chunk_multiple_cves(self, cves: list[CVERecord]) -> list[CVEChunk]:
        """
        Chunk multiple CVE records.
        
        Args:
            cves: List of CVERecord objects
        
        Returns:
            Flat list of all chunks from all CVEs
        """
        all_chunks = []
        
        for cve in cves:
            chunks = self.chunk_cve(cve)
            all_chunks.extend(chunks)
        
        logger.info(
            f"Chunked {len(cves)} CVEs into {len(all_chunks)} total chunks "
            f"(avg {len(all_chunks) / len(cves):.1f} chunks per CVE)"
        )
        
        return all_chunks