#!/usr/bin/env python3
"""
Seed Vector Store Script

Processes downloaded CVE data and loads it into ChromaDB.

Steps:
1. Load raw CVE data from disk
2. Parse into CVERecord models
3. Chunk CVE records
4. Generate embeddings
5. Store in ChromaDB

Usage:
    python scripts/seed_vector_store.py
    python scripts/seed_vector_store.py --reset  # Clear existing data first
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vulnremedy.rag.chunking.cve_chunker import CVEChunker
from vulnremedy.rag.embeddings.embedding_service import EmbeddingService
from vulnremedy.rag.ingestion.parser import CVEParser
from vulnremedy.rag.retrieval.vector_store import VectorStore
from vulnremedy.utils.logging import logger


def load_nvd_files(data_dir: Path) -> list[dict]:
    """Load all NVD JSON files from data directory."""
    nvd_dir = data_dir / "raw" / "nvd"
    
    if not nvd_dir.exists():
        logger.warning(f"NVD directory not found: {nvd_dir}")
        return []
    
    all_vulnerabilities = []
    
    for json_file in nvd_dir.glob("*.json"):
        logger.info(f"Loading NVD file: {json_file.name}")
        
        with open(json_file, "r") as f:
            data = json.load(f)
            vulnerabilities = data.get("vulnerabilities", [])
            all_vulnerabilities.extend(vulnerabilities)
    
    logger.info(f"Loaded {len(all_vulnerabilities)} CVEs from NVD")
    return all_vulnerabilities


def load_osv_files(data_dir: Path) -> list[dict]:
    """Load all OSV JSON files from data directory."""
    osv_dir = data_dir / "raw" / "osv"
    
    if not osv_dir.exists():
        logger.warning(f"OSV directory not found: {osv_dir}")
        return []
    
    all_vulnerabilities = []
    
    for json_file in osv_dir.glob("*.json"):
        logger.info(f"Loading OSV file: {json_file.name}")
        
        with open(json_file, "r") as f:
            vulnerabilities = json.load(f)
            if isinstance(vulnerabilities, list):
                all_vulnerabilities.extend(vulnerabilities)
    
    logger.info(f"Loaded {len(all_vulnerabilities)} vulnerabilities from OSV")
    return all_vulnerabilities


def parse_vulnerabilities(
    nvd_data: list[dict],
    osv_data: list[dict],
    parser: CVEParser
) -> list:
    """Parse raw vulnerability data into CVERecord models."""
    cve_records = []
    
    # Parse NVD data
    logger.info("Parsing NVD data...")
    for vuln in nvd_data:
        cve_record = parser.parse_nvd_vulnerability(vuln)
        if cve_record:
            cve_records.append(cve_record)
    
    logger.info(f"Parsed {len(cve_records)} CVEs from NVD")
    
    # Parse OSV data
    logger.info("Parsing OSV data...")
    osv_count = 0
    for vuln in osv_data:
        cve_record = parser.parse_osv_vulnerability(vuln)
        if cve_record:
            cve_records.append(cve_record)
            osv_count += 1
    
    logger.info(f"Parsed {osv_count} CVEs from OSV")
    
    # Deduplicate by CVE ID (prefer NVD version if duplicate)
    seen_ids = set()
    unique_records = []
    duplicates = 0
    
    for record in cve_records:
        if record.cve_id not in seen_ids:
            seen_ids.add(record.cve_id)
            unique_records.append(record)
        else:
            duplicates += 1
    
    logger.info(
        f"Deduplicated: {len(unique_records)} unique CVEs "
        f"({duplicates} duplicates removed)"
    )
    
    return unique_records


def main():
    parser = argparse.ArgumentParser(
        description="Process CVE data and load into vector store"
    )
    
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset vector store (delete existing data)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Data directory (default: ./data)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("SEEDING VECTOR STORE")
    logger.info("=" * 60)
    
    # Step 1: Load raw data
    logger.info("\n[Step 1/5] Loading raw CVE data from disk...")
    nvd_data = load_nvd_files(args.data_dir)
    osv_data = load_osv_files(args.data_dir)
    
    if not nvd_data and not osv_data:
        logger.error("No data found. Run download_cve_data.py first.")
        sys.exit(1)
    
    # Step 2: Parse into CVERecord models
    logger.info("\n[Step 2/5] Parsing into CVERecord models...")
    cve_parser = CVEParser()
    cve_records = parse_vulnerabilities(nvd_data, osv_data, cve_parser)
    
    if not cve_records:
        logger.error("No CVEs successfully parsed")
        sys.exit(1)
    
    # Step 3: Chunk CVE records
    logger.info("\n[Step 3/5] Chunking CVE records...")
    chunker = CVEChunker()
    chunks = chunker.chunk_multiple_cves(cve_records)
    
    logger.info(f"Generated {len(chunks)} chunks from {len(cve_records)} CVEs")
    
    # Step 4: Generate embeddings
    logger.info("\n[Step 4/5] Generating embeddings...")
    embedding_service = EmbeddingService()
    embedded_chunks = embedding_service.embed_chunks(chunks, show_progress=True)
    
    logger.info(f"Generated {len(embedded_chunks)} embeddings")
    
    # Step 5: Load into vector store
    logger.info("\n[Step 5/5] Loading into ChromaDB...")
    vector_store = VectorStore()
    
    if args.reset:
        logger.warning("Resetting vector store (deleting existing data)...")
        vector_store.reset_collection()
    
    vector_store.add_chunks(embedded_chunks)
    
    # Get stats
    stats = vector_store.get_stats()
    
    # Summary
    logger.info("=" * 60)
    logger.info("SEEDING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"CVEs processed: {len(cve_records)}")
    logger.info(f"Chunks created: {len(chunks)}")
    logger.info(f"Embeddings generated: {len(embedded_chunks)}")
    logger.info(f"Total documents in store: {stats['total_documents']}")
    logger.info("")
    logger.info("Vector store ready for retrieval!")
    logger.info("")
    logger.info("Sample distribution:")
    if "sample_severities" in stats:
        logger.info(f"  Severities: {stats['sample_severities']}")
    if "sample_chunk_types" in stats:
        logger.info(f"  Chunk types: {stats['sample_chunk_types']}")
    if "sample_sources" in stats:
        logger.info(f"  Sources: {stats['sample_sources']}")


if __name__ == "__main__":
    main()