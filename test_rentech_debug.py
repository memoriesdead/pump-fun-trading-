#!/usr/bin/env python3
"""Debug RenTech filter with real API data."""
import asyncio
import aiohttp
from trading.core.token import Token
from trading.core.rentech_filter import RenTechFilter


async def test_raw():
    url = 'https://api.geckoterminal.com/api/v2/networks/solana/trending_pools'

    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=15) as resp:
            data = await resp.json()
            pools = data.get('data', [])

    print(f'Raw API returned {len(pools)} pools')

    # Parse tokens
    tokens = []
    for pool in pools:
        token = Token.from_geckoterminal(pool)
        if token.address:
            tokens.append(token)

    print(f'Parsed {len(tokens)} tokens')
    print()

    # Test RenTech filter on raw tokens - use $10 trade size (not $1!)
    rentech = RenTechFilter(min_edge_after_friction=0.002, default_trade_size=10.0)

    print('=== RENTECH FILTER ANALYSIS ===')
    print(f'{"Symbol":<12} | {"Mom":>7} | {"Liq":>12} | {"Edge":>7} | {"Net":>7} | {"Dir":>3} | Status')
    print('-' * 80)

    tradeable = 0
    for token in tokens[:15]:
        r = rentech.check(token)  # Uses default $10 trade size
        status = 'PASS' if r.is_tradeable else 'FAIL'
        if r.is_tradeable:
            tradeable += 1
        print(f'{token.symbol:<12} | {token.momentum:>+6.1f}% | ${token.liquidity_usd:>10,.0f} | {r.combined_edge*100:>+6.2f}% | {r.net_edge*100:>+6.2f}% | {r.direction:>3} | {status}')

        # Show rejection reasons for failed tokens
        if not r.is_tradeable and r.reasons:
            print(f'             -> Reasons: {", ".join(r.reasons)}')

    print()
    print(f'Tradeable: {tradeable}/{min(15, len(tokens))} shown')

    # Analyze all tokens
    all_tradeable = 0
    reasons_count = {}
    for token in tokens:
        r = rentech.check(token)  # Uses default $10 trade size
        if r.is_tradeable:
            all_tradeable += 1
        for reason in r.reasons:
            key = reason.split()[0]  # First word
            reasons_count[key] = reasons_count.get(key, 0) + 1

    print(f'All tradeable: {all_tradeable}/{len(tokens)}')
    print()
    print('Rejection reasons:')
    for reason, count in sorted(reasons_count.items(), key=lambda x: -x[1]):
        print(f'  {reason}: {count}')


if __name__ == '__main__':
    asyncio.run(test_raw())
