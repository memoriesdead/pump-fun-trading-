#!/usr/bin/env python3
"""
TOP TOKENS FETCHER
==================

Fetches top Solana/pump.fun tokens by volume and market cap from FREE APIs.

Sources:
1. DexScreener - Top boosted tokens (FREE, 60 req/min)
2. DexScreener - Search trending pairs (FREE, 300 req/min)
3. Moralis - Graduated pump.fun tokens (FREE tier available)

Usage:
    python data/top_tokens.py --limit 1000
    python data/top_tokens.py --source dexscreener
    python data/top_tokens.py --source moralis --api-key YOUR_KEY
"""

import asyncio
import aiohttp
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TopToken:
    """Token with volume/liquidity data"""
    address: str
    name: str = ""
    symbol: str = ""
    chain: str = "solana"
    volume_24h: float = 0.0
    liquidity: float = 0.0
    market_cap: float = 0.0
    price_usd: float = 0.0
    source: str = ""


class TopTokenFetcher:
    """Fetch top tokens from multiple FREE sources"""

    DEXSCREENER_BOOST = "https://api.dexscreener.com/token-boosts/top/v1"
    DEXSCREENER_SEARCH = "https://api.dexscreener.com/latest/dex/search"
    DEXSCREENER_PAIRS = "https://api.dexscreener.com/latest/dex/pairs/solana"
    DEXSCREENER_TOKENS = "https://api.dexscreener.com/tokens/v1/solana"

    # Moralis endpoints (requires free API key from developers.moralis.com)
    MORALIS_GRADUATED = "https://solana-gateway.moralis.io/token/mainnet/exchange/pumpfun/graduated"
    MORALIS_BONDING = "https://solana-gateway.moralis.io/token/mainnet/exchange/pumpfun/bonding"
    MORALIS_TRENDING = "https://solana-gateway.moralis.io/token/mainnet/tokens/trending"

    def __init__(self, moralis_api_key: str = None):
        self.moralis_api_key = moralis_api_key
        self.tokens: Dict[str, TopToken] = {}

    async def fetch_dexscreener_boosted(self) -> List[TopToken]:
        """Fetch top boosted tokens from DexScreener (FREE)"""
        tokens = []

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.DEXSCREENER_BOOST) as resp:
                    if resp.status == 200:
                        data = await resp.json()

                        for item in data:
                            if item.get('chainId') == 'solana':
                                token = TopToken(
                                    address=item.get('tokenAddress', ''),
                                    name=item.get('name', ''),
                                    symbol=item.get('symbol', ''),
                                    source='dexscreener_boost'
                                )
                                tokens.append(token)

                        logger.info(f"DexScreener Boosted: {len(tokens)} Solana tokens")

            except Exception as e:
                logger.error(f"DexScreener boost error: {e}")

        return tokens

    async def fetch_dexscreener_search(self, queries: List[str] = None) -> List[TopToken]:
        """Search DexScreener for popular tokens (FREE)"""
        tokens = []

        if queries is None:
            # Expanded search for Solana tokens - more queries = more tokens
            queries = [
                # Top DEX tokens
                "SOL", "BONK", "WIF", "POPCAT", "MEW", "BOME", "JUP", "RAY",
                "SLERF", "MYRO", "SAMO", "ORCA", "FARTCOIN", "GOAT", "PNUT",
                "ACT", "MOODENG", "PENGU", "AI16Z", "GRIFFAIN", "ZEREBRO",
                # More meme tokens
                "BRETT", "PEPE", "DOGE", "SHIB", "FLOKI", "TURBO", "LADYS",
                "WOJAK", "CHAD", "GIGA", "BASED", "TOSHI", "MOCHI", "PONKE",
                # Solana ecosystem
                "RENDER", "HNT", "PYTH", "JTO", "TENSOR", "MOBILE", "HONEY",
                "BSOL", "MSOL", "JITOSOL", "LST", "BLZE", "DUST", "FORGE",
                # Pump.fun graduated
                "pump", "fun", "meme", "degen", "ape", "moon", "rocket",
                "lambo", "chad", "wojak", "pepe", "doge", "shiba", "inu",
                # Generic high volume searches
                "solana", "raydium", "jupiter", "orca", "meteora",
                # Letter searches for broad coverage
                "a]", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
                "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"
            ]

        async with aiohttp.ClientSession() as session:
            for query in queries:
                try:
                    url = f"{self.DEXSCREENER_SEARCH}?q={query}"
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            pairs = data.get('pairs', []) or []

                            for pair in pairs:
                                if pair.get('chainId') != 'solana':
                                    continue

                                base = pair.get('baseToken', {})
                                address = base.get('address', '')

                                if address and address not in self.tokens:
                                    token = TopToken(
                                        address=address,
                                        name=base.get('name', ''),
                                        symbol=base.get('symbol', ''),
                                        volume_24h=float(pair.get('volume', {}).get('h24', 0) or 0),
                                        liquidity=float(pair.get('liquidity', {}).get('usd', 0) or 0),
                                        market_cap=float(pair.get('marketCap', 0) or 0),
                                        price_usd=float(pair.get('priceUsd', 0) or 0),
                                        source='dexscreener_search'
                                    )
                                    tokens.append(token)
                                    self.tokens[address] = token

                    # Rate limit: 300 req/min
                    await asyncio.sleep(0.2)

                except Exception as e:
                    logger.warning(f"Search error for {query}: {e}")

        logger.info(f"DexScreener Search: {len(tokens)} tokens")
        return tokens

    async def fetch_dexscreener_pairs(self, addresses: List[str]) -> List[TopToken]:
        """Fetch detailed pair data for specific tokens"""
        tokens = []

        # Batch addresses (max 30 per request)
        batches = [addresses[i:i+30] for i in range(0, len(addresses), 30)]

        async with aiohttp.ClientSession() as session:
            for batch in batches:
                try:
                    addr_str = ','.join(batch)
                    url = f"{self.DEXSCREENER_TOKENS}/{addr_str}"

                    async with session.get(url) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            pairs = data.get('pairs', []) or []

                            for pair in pairs:
                                base = pair.get('baseToken', {})
                                address = base.get('address', '')

                                if address:
                                    token = TopToken(
                                        address=address,
                                        name=base.get('name', ''),
                                        symbol=base.get('symbol', ''),
                                        volume_24h=float(pair.get('volume', {}).get('h24', 0) or 0),
                                        liquidity=float(pair.get('liquidity', {}).get('usd', 0) or 0),
                                        market_cap=float(pair.get('marketCap', 0) or 0),
                                        price_usd=float(pair.get('priceUsd', 0) or 0),
                                        source='dexscreener_pairs'
                                    )
                                    tokens.append(token)

                    await asyncio.sleep(0.2)

                except Exception as e:
                    logger.warning(f"Pairs fetch error: {e}")

        return tokens

    async def fetch_moralis_graduated(self, limit: int = 100) -> List[TopToken]:
        """Fetch graduated pump.fun tokens from Moralis (FREE tier)"""
        if not self.moralis_api_key:
            logger.warning("Moralis API key required. Get free key at developers.moralis.com")
            return []

        tokens = []

        headers = {
            "X-API-Key": self.moralis_api_key,
            "Accept": "application/json"
        }

        async with aiohttp.ClientSession() as session:
            try:
                url = f"{self.MORALIS_GRADUATED}?limit={limit}"
                async with session.get(url, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()

                        for item in data.get('result', []):
                            token = TopToken(
                                address=item.get('tokenAddress', ''),
                                name=item.get('name', ''),
                                symbol=item.get('symbol', ''),
                                liquidity=float(item.get('liquidity', 0) or 0),
                                price_usd=float(item.get('priceUsd', 0) or 0),
                                source='moralis_graduated'
                            )
                            tokens.append(token)

                        logger.info(f"Moralis Graduated: {len(tokens)} tokens")
                    else:
                        logger.error(f"Moralis error: {resp.status}")

            except Exception as e:
                logger.error(f"Moralis error: {e}")

        return tokens

    async def fetch_moralis_trending(self, limit: int = 100) -> List[TopToken]:
        """Fetch trending Solana tokens from Moralis (FREE tier)"""
        if not self.moralis_api_key:
            return []

        tokens = []

        headers = {
            "X-API-Key": self.moralis_api_key,
            "Accept": "application/json"
        }

        async with aiohttp.ClientSession() as session:
            try:
                url = f"{self.MORALIS_TRENDING}?limit={limit}"
                async with session.get(url, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()

                        for item in data.get('result', []):
                            token = TopToken(
                                address=item.get('tokenAddress', ''),
                                name=item.get('name', ''),
                                symbol=item.get('symbol', ''),
                                volume_24h=float(item.get('volume24h', 0) or 0),
                                liquidity=float(item.get('liquidity', 0) or 0),
                                market_cap=float(item.get('marketCap', 0) or 0),
                                price_usd=float(item.get('priceUsd', 0) or 0),
                                source='moralis_trending'
                            )
                            tokens.append(token)

                        logger.info(f"Moralis Trending: {len(tokens)} tokens")

            except Exception as e:
                logger.error(f"Moralis trending error: {e}")

        return tokens

    async def fetch_all(self, limit: int = 1000) -> List[TopToken]:
        """Fetch from all sources and deduplicate"""
        all_tokens = []
        seen = set()

        # 1. DexScreener boosted (always free)
        boosted = await self.fetch_dexscreener_boosted()
        for t in boosted:
            if t.address not in seen:
                all_tokens.append(t)
                seen.add(t.address)

        # 2. DexScreener search (always free)
        searched = await self.fetch_dexscreener_search()
        for t in searched:
            if t.address not in seen:
                all_tokens.append(t)
                seen.add(t.address)

        # 3. Moralis if API key provided
        if self.moralis_api_key:
            graduated = await self.fetch_moralis_graduated(limit=100)
            for t in graduated:
                if t.address not in seen:
                    all_tokens.append(t)
                    seen.add(t.address)

            trending = await self.fetch_moralis_trending(limit=100)
            for t in trending:
                if t.address not in seen:
                    all_tokens.append(t)
                    seen.add(t.address)

        # 4. Get detailed data for boosted tokens
        boosted_addrs = [t.address for t in boosted if t.address]
        if boosted_addrs:
            detailed = await self.fetch_dexscreener_pairs(boosted_addrs)
            # Update existing tokens with detailed data
            for t in detailed:
                if t.address in seen:
                    # Update existing
                    for existing in all_tokens:
                        if existing.address == t.address:
                            existing.volume_24h = t.volume_24h or existing.volume_24h
                            existing.liquidity = t.liquidity or existing.liquidity
                            existing.market_cap = t.market_cap or existing.market_cap
                            break

        # Sort by volume/liquidity
        all_tokens.sort(key=lambda x: (x.volume_24h + x.liquidity), reverse=True)

        logger.info(f"Total unique tokens: {len(all_tokens)}")
        return all_tokens[:limit]

    def save(self, tokens: List[TopToken], path: str = "top_tokens.json"):
        """Save tokens to JSON file"""
        data = {
            "fetched_at": time.time(),
            "count": len(tokens),
            "tokens": [
                {
                    "address": t.address,
                    "name": t.name,
                    "symbol": t.symbol,
                    "volume_24h": t.volume_24h,
                    "liquidity": t.liquidity,
                    "market_cap": t.market_cap,
                    "price_usd": t.price_usd,
                    "source": t.source
                }
                for t in tokens
            ]
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(tokens)} tokens to {path}")

    @staticmethod
    def load(path: str = "top_tokens.json") -> List[TopToken]:
        """Load tokens from JSON file"""
        with open(path) as f:
            data = json.load(f)

        return [
            TopToken(
                address=t['address'],
                name=t.get('name', ''),
                symbol=t.get('symbol', ''),
                volume_24h=t.get('volume_24h', 0),
                liquidity=t.get('liquidity', 0),
                market_cap=t.get('market_cap', 0),
                price_usd=t.get('price_usd', 0),
                source=t.get('source', '')
            )
            for t in data.get('tokens', [])
        ]


async def main():
    parser = argparse.ArgumentParser(description="Fetch top Solana tokens")
    parser.add_argument("--limit", type=int, default=1000, help="Max tokens to fetch")
    parser.add_argument("--moralis-key", type=str, default=None, help="Moralis API key (free at developers.moralis.com)")
    parser.add_argument("--output", type=str, default="data/top_tokens.json", help="Output file")
    args = parser.parse_args()

    fetcher = TopTokenFetcher(moralis_api_key=args.moralis_key)

    print("=" * 60)
    print("TOP TOKENS FETCHER")
    print("=" * 60)
    print(f"Limit: {args.limit}")
    print(f"Moralis API: {'Yes' if args.moralis_key else 'No (get free key at developers.moralis.com)'}")
    print("=" * 60)

    tokens = await fetcher.fetch_all(limit=args.limit)

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fetcher.save(tokens, args.output)

    # Print top 20
    print("\nTOP 20 BY VOLUME + LIQUIDITY:")
    print("-" * 80)
    print(f"{'Symbol':<12} {'Name':<20} {'Volume 24h':>15} {'Liquidity':>15}")
    print("-" * 80)

    for t in tokens[:20]:
        vol = f"${t.volume_24h:,.0f}" if t.volume_24h else "N/A"
        liq = f"${t.liquidity:,.0f}" if t.liquidity else "N/A"
        print(f"{t.symbol:<12} {t.name[:20]:<20} {vol:>15} {liq:>15}")

    print("-" * 80)
    print(f"\nTotal tokens saved: {len(tokens)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
