#!/usr/bin/env python3
"""
Download CVE Data Script

Downloads CVE data from NVD and OSV APIs and saves to disk.

Usage:
    python scripts/download_cve_data.py --days-back 90 --max-cves 1000
    python scripts/download_cve_data.py --ecosystems maven pypi npm
"""

import argparse
import sys
from pathlib import Path

# Add src to path so we can import vulnremedy
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vulnremedy.rag.ingestion.nvd_loader import NVDLoader
from vulnremedy.rag.ingestion.osv_loader import OSVLoader
from vulnremedy.utils.logging import logger


def download_nvd_data(days_back: int, max_results: int | None) -> int:
    """
    Download CVE data from NVD.
    
    Args:
        days_back: How many days back to fetch
        max_results: Maximum CVEs to download (None = all)
    
    Returns:
        Number of CVEs downloaded
    """
    logger.info(
        f"Starting NVD download: days_back={days_back}, max_results={max_results}"
    )
    
    with NVDLoader() as loader:
        cves = loader.fetch_cves(
            days_back=days_back,
            max_results=max_results
        )
    
    logger.info(f"NVD download complete: {len(cves)} CVEs")
    return len(cves)


def download_osv_data(ecosystems: list[str], max_per_ecosystem: int | None) -> int:
    """
    Download vulnerability data from OSV.
    
    Args:
        ecosystems: List of ecosystems to download
        max_per_ecosystem: Max vulnerabilities per ecosystem
    
    Returns:
        Number of vulnerabilities downloaded
    """
    logger.info(
        f"Starting OSV download: ecosystems={ecosystems}, "
        f"max_per_ecosystem={max_per_ecosystem}"
    )
    
    with OSVLoader() as loader:
        vulns = loader.fetch_vulnerabilities(
            ecosystems=ecosystems,
            max_per_ecosystem=max_per_ecosystem
        )
    
    logger.info(f"OSV download complete: {len(vulns)} vulnerabilities")
    return len(vulns)


def main():
    parser = argparse.ArgumentParser(
        description="Download CVE data from NVD and OSV APIs"
    )
    
    # NVD options
    parser.add_argument(
        "--days-back",
        type=int,
        default=90,
        help="How many days back to fetch CVEs from NVD (default: 90)"
    )
    parser.add_argument(
        "--max-cves",
        type=int,
        default=1000,
        help="Maximum CVEs to download from NVD (default: 1000)"
    )
    
    # OSV options
    parser.add_argument(
        "--ecosystems",
        nargs="+",
        default=["Maven", "PyPI", "npm"],
        help="Ecosystems to download from OSV (default: Maven PyPI npm)"
    )
    parser.add_argument(
        "--max-per-ecosystem",
        type=int,
        default=300,
        help="Max vulnerabilities per ecosystem from OSV (default: 300)"
    )
    
    # Source selection
    parser.add_argument(
        "--nvd-only",
        action="store_true",
        help="Download only from NVD"
    )
    parser.add_argument(
        "--osv-only",
        action="store_true",
        help="Download only from OSV"
    )
    
    args = parser.parse_args()
    
    # Validate
    if args.nvd_only and args.osv_only:
        logger.error("Cannot specify both --nvd-only and --osv-only")
        sys.exit(1)
    
    total_downloaded = 0
    
    # Download from NVD
    if not args.osv_only:
        logger.info("=" * 60)
        logger.info("DOWNLOADING FROM NVD")
        logger.info("=" * 60)
        
        try:
            nvd_count = download_nvd_data(
                days_back=args.days_back,
                max_results=args.max_cves
            )
            total_downloaded += nvd_count
        except Exception as e:
            logger.error(f"NVD download failed: {e}", exc_info=True)
    
    # Download from OSV
    if not args.nvd_only:
        logger.info("=" * 60)
        logger.info("DOWNLOADING FROM OSV")
        logger.info("=" * 60)
        
        try:
            osv_count = download_osv_data(
                ecosystems=args.ecosystems,
                max_per_ecosystem=args.max_per_ecosystem
            )
            total_downloaded += osv_count
        except Exception as e:
            logger.error(f"OSV download failed: {e}", exc_info=True)
    
    # Summary
    logger.info("=" * 60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total downloaded: {total_downloaded}")
    logger.info("Data saved to:")
    logger.info("  - data/raw/nvd/")
    logger.info("  - data/raw/osv/")
    logger.info("")
    logger.info("Next step: Run seed_vector_store.py to process and load data")


if __name__ == "__main__":
    main()