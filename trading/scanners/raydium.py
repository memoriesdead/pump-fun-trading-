"""
Raydium Scanner - Scan for tokens on Raydium via GeckoTerminal API.

Usage:
    from trading.scanners import RaydiumScanner

    scanner = RaydiumScanner()
    result = await scanner.scan()
    for token, filter_result in result.opportunities:
        print(f"{token.symbol}: {filter_result.net_edge:.1%} edge")
"""
import asyncio
import aiohttp
from typing import List, Optional

from trading.core.config import ENDPOINTS
from trading.core.token import Token
from trading.core.filters import TokenFilter
from .base import BaseScanner, ScanResult


class RaydiumScanner(BaseScanner):
    """
    Scanner for Raydium tokens via GeckoTerminal API.

    Fetches from multiple endpoints:
    - New pools (recently created)
    - Trending pools (high volume)
    - Top pools (highest liquidity)
    """

    def __init__(
        self,
        filter: Optional[TokenFilter] = None,
        timeout: float = 15.0,
        include_new: bool = True,
        include_trending: bool = True,
        include_top: bool = True,
    ):
        super().__init__(filter)
        self.timeout = timeout
        self.include_new = include_new
        self.include_trending = include_trending
        self.include_top = include_top

        # Build endpoint list
        self.endpoints = []
        if include_new:
            self.endpoints.append(('new', ENDPOINTS['GECKOTERMINAL_NEW']))
        if include_trending:
            self.endpoints.append(('trending', ENDPOINTS['GECKOTERMINAL_TRENDING']))
        if include_top:
            self.endpoints.append(('pools', ENDPOINTS['GECKOTERMINAL_POOLS']))

    @property
    def name(self) -> str:
        return "RaydiumScanner"

    @property
    def source(self) -> str:
        return "raydium"

    async def scan(self) -> ScanResult:
        """
        Scan all configured endpoints for tokens.

        Returns:
            ScanResult with all found tokens and filtered opportunities
        """
        result = ScanResult(source=self.source)

        try:
            async with aiohttp.ClientSession() as session:
                # Fetch from all endpoints in parallel
                tasks = [
                    self._fetch_endpoint(session, name, url)
                    for name, url in self.endpoints
                ]
                endpoint_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Combine all tokens (dedupe by address)
                seen_addresses = set()
                all_tokens = []

                for pools in endpoint_results:
                    if isinstance(pools, Exception):
                        continue
                    if not pools:
                        continue

                    for pool in pools:
                        token = Token.from_geckoterminal(pool)
                        if token.address and token.address not in seen_addresses:
                            seen_addresses.add(token.address)
                            all_tokens.append(token)

                result.tokens = all_tokens

                # Filter tokens
                result.opportunities = self.filter.get_best_opportunities(
                    all_tokens,
                    max_results=10,
                    min_edge=0.05,  # 5% minimum edge
                )

        except Exception as e:
            result.error = str(e)

        return result

    async def _fetch_endpoint(
        self,
        session: aiohttp.ClientSession,
        name: str,
        url: str,
    ) -> List[dict]:
        """
        Fetch pools from a single endpoint.

        Args:
            session: aiohttp session
            name: Endpoint name for logging
            url: API URL

        Returns:
            List of pool dictionaries
        """
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with session.get(url, timeout=timeout) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                return data.get('data', [])
        except asyncio.TimeoutError:
            return []
        except Exception:
            return []

    async def scan_continuous(
        self,
        interval: float = 5.0,
        callback=None,
    ):
        """
        Continuously scan for tokens.

        Args:
            interval: Seconds between scans
            callback: Optional callback for opportunities
        """
        while True:
            result = await self.scan_with_stats()

            if callback and result.opportunities:
                await callback(result.opportunities)

            await asyncio.sleep(interval)
