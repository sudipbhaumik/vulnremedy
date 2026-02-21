"""
Base Dependency Parser — Abstract interface for manifest parsers.

All ecosystem-specific parsers (Maven, Gradle, npm, pip) inherit from this.

Architectural notes:
    - Each parser handles one manifest format
    - Parsers are stateless (pure functions)
    - Parsers return Dependency models from Week 1
    - Parsers handle malformed input gracefully (return empty list, not crash)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from vulnremedy.models.dependency import Dependency
from vulnremedy.utils.logging import logger


class DependencyParser(ABC):
    """
    Abstract base class for dependency parsers.
    
    Each ecosystem (Maven, Gradle, npm, pip) implements a concrete parser.
    """
    
    @abstractmethod
    def parse(self, content: str) -> list[Dependency]:
        """
        Parse manifest file content and extract dependencies.
        
        Args:
            content: Raw manifest file content as string
        
        Returns:
            List of Dependency objects
            
        Note:
            Implementations should handle parsing errors gracefully.
            Return empty list on parse failure, not raise exceptions.
        """
        pass
    
    @property
    @abstractmethod
    def ecosystem(self) -> str:
        """
        Return the ecosystem this parser handles.
        
        Examples: "maven", "gradle", "npm", "pip"
        """
        pass
    
    def _log_parse_error(self, error: Exception, context: str = "") -> None:
        """
        Helper to log parsing errors consistently.
        
        Args:
            error: Exception that occurred
            context: Additional context about what was being parsed
        """
        logger.error(
            f"Failed to parse {self.ecosystem} manifest",
            error=str(error),
            context=context
        )
    
    def _log_parse_success(self, dependency_count: int) -> None:
        """
        Helper to log successful parsing.
        
        Args:
            dependency_count: Number of dependencies extracted
        """
        logger.info(
            f"Parsed {self.ecosystem} manifest",
            dependencies_found=dependency_count
        )