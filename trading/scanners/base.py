"""
Base Scanner - Abstract interface for token scanners.

Usage:
    from trading.scanners import BaseScanner

    class MyScanner(BaseScanner):
        async def scan(self) -> ScanResult:
            # Implement scanning logic
            pass
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
import time

from trading.core.token import Token
from trading.core.filters import TokenFilter, FilterResult


@dataclass
class ScanResult:
    """Result of a scanner run."""

    # All tokens found
    tokens: List[Token] = field(default_factory=list)

    # Tokens that passed filters
    opportunities: List[tuple] = field(default_factory=list)  # List of (Token, FilterResult)

    # Metadata
    scan_time: float = 0.0
    source: str = ""
    error: Optional[str] = None

    @property
    def total_found(self) -> int:
        return len(self.tokens)

    @property
    def total_passed(self) -> int:
        return len(self.opportunities)

    @property
    def pass_rate(self) -> float:
        return self.total_passed / self.total_found if self.total_found > 0 else 0.0

    def best_opportunity(self) -> Optional[tuple]:
        """Get the best opportunity by net edge."""
        if not self.opportunities:
            return None
        return self.opportunities[0]  # Already sorted by edge


class BaseScanner(ABC):
    """
    Abstract base class for token scanners.

    Scanners fetch tokens from various sources (APIs, WebSockets, etc.)
    and filter them for trading opportunities.
    """

    def __init__(self, filter: Optional[TokenFilter] = None):
        self.filter = filter or TokenFilter()
        self.last_scan_time = 0.0
        self.total_scans = 0
        self.total_tokens_found = 0
        self.total_opportunities = 0

    @property
    @abstractmethod
    def name(self) -> str:
        """Scanner name for logging."""
        pass

    @property
    @abstractmethod
    def source(self) -> str:
        """Token source identifier."""
        pass

    @abstractmethod
    async def scan(self) -> ScanResult:
        """
        Scan for tokens and filter for opportunities.

        Returns:
            ScanResult with tokens and filtered opportunities
        """
        pass

    async def scan_with_stats(self) -> ScanResult:
        """
        Scan and update internal statistics.

        Returns:
            ScanResult with tokens and filtered opportunities
        """
        start = time.time()
        result = await self.scan()
        result.scan_time = time.time() - start

        # Update stats
        self.last_scan_time = result.scan_time
        self.total_scans += 1
        self.total_tokens_found += result.total_found
        self.total_opportunities += result.total_passed

        return result

    def filter_tokens(self, tokens: List[Token]) -> List[tuple]:
        """
        Filter tokens and return sorted opportunities.

        Args:
            tokens: List of tokens to filter

        Returns:
            List of (Token, FilterResult) tuples sorted by edge
        """
        return self.filter.get_best_opportunities(tokens)

    def get_stats(self) -> dict:
        """Get scanner statistics."""
        return {
            'name': self.name,
            'source': self.source,
            'total_scans': self.total_scans,
            'total_tokens_found': self.total_tokens_found,
            'total_opportunities': self.total_opportunities,
            'last_scan_time': self.last_scan_time,
            'avg_tokens_per_scan': self.total_tokens_found / self.total_scans if self.total_scans > 0 else 0,
            'opportunity_rate': self.total_opportunities / self.total_tokens_found if self.total_tokens_found > 0 else 0,
        }
