#!/usr/bin/env python3
"""
Pump.fun Trading System - Unified Entry Point
==============================================

Modular trading system for Solana tokens on Raydium/PumpFun.

Usage:
    # Paper trade with $10
    python main.py --capital 10

    # Paper trade for 5 minutes
    python main.py --capital 10 --duration 300

    # Show configuration only
    python main.py --show-config

    # Scan for opportunities only (no trading)
    python main.py --scan-only
"""
import asyncio
import argparse
import logging
import sys
import time

from trading.modular_engine import ModularEngine
from trading.core.config import Config, FilterConfig, ScalpConfig
from trading.core.friction import FRICTION
from trading.core.filters import TokenFilter
from trading.scanners.raydium import RaydiumScanner


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def show_config(capital: float):
    """Display configuration for given capital size."""
    config = Config.for_account_size(capital)
    friction = FRICTION.compute_total_cost(
        trade_size_usd=capital,
        liquidity_usd=100_000,
        is_pumpfun=False,
    )

    print(f"\n{'='*60}")
    print(f"  CONFIGURATION FOR ${capital:.2f} ACCOUNT")
    print(f"{'='*60}")

    print(f"\nFILTER CRITERIA:")
    print(f"  Min Liquidity:    ${config.filters.min_liquidity:,}")
    print(f"  Max Liquidity:    ${config.filters.max_liquidity:,}")
    print(f"  Min Momentum:     {config.filters.min_momentum}%")
    print(f"  Max Momentum:     {config.filters.max_momentum}%")
    print(f"  Min Volatility:   {config.filters.min_volatility_5m}%")
    print(f"  Min Volume 5m:    ${config.filters.min_volume_5m:,}")
    print(f"  Min Buy Ratio:    {config.filters.min_buy_ratio*100:.0f}%")
    print(f"  Min Txns 5m:      {config.filters.min_txns_5m}")

    print(f"\nSCALP PARAMETERS:")
    print(f"  Target Profit:    {config.scalp.target_profit_pct*100:.0f}%")
    print(f"  Stop Loss:        {config.scalp.max_loss_pct*100:.0f}%")
    print(f"  Max Hold Time:    {config.scalp.max_hold_ms/1000:.0f}s")
    print(f"  Max Positions:    {config.scalp.max_open_positions}")
    print(f"  Max Position %:   {config.scalp.max_position_pct*100:.0f}%")
    print(f"  Min Edge:         {config.scalp.min_expected_edge*100:.0f}%")

    print(f"\nFRICTION MODEL (${capital:.0f} on $100k liquidity):")
    breakdown = friction['breakdown']
    print(f"  Fixed Fees:       {breakdown['fixed_fees_pct']*100:.2f}%")
    print(f"  DEX Fees:         {breakdown['dex_fees_pct']*100:.2f}%")
    print(f"  Slippage:         {breakdown['slippage_pct']*100:.2f}%")
    print(f"  Price Impact:     {breakdown['price_impact_pct']*100:.4f}%")
    print(f"  TOTAL:            {friction['total_cost_pct']*100:.2f}% per leg")
    print(f"  Round-trip:       {friction['total_cost_pct']*200:.2f}%")
    print(f"  Min Move to Win:  {friction['min_profit_pct']*200:.1f}%+")

    print(f"\nEDGE MATH:")
    print(f"  Your filters require {config.filters.min_momentum}%+ momentum")
    print(f"  With 40% continuation rate = {config.filters.min_momentum * 0.4:.1f}% expected move")
    print(f"  Minus {friction['total_cost_pct']*200:.1f}% friction = {config.filters.min_momentum * 0.4 - friction['total_cost_pct']*200:.1f}% net edge")

    print(f"{'='*60}\n")


async def scan_only():
    """Scan for opportunities without trading."""
    print("\nScanning Raydium for opportunities...\n")

    filter = TokenFilter(FilterConfig())
    scanner = RaydiumScanner(filter=filter)

    result = await scanner.scan_with_stats()

    print(f"Found {result.total_found} tokens, {result.total_passed} passed filters\n")

    if result.opportunities:
        print("TOP OPPORTUNITIES:")
        print("-" * 80)
        for i, (token, filter_result) in enumerate(result.opportunities[:10], 1):
            print(
                f"{i}. {token.symbol:12} | "
                f"Price: ${token.price:.8f} | "
                f"Mom: {token.momentum:+6.1f}% | "
                f"Liq: ${token.liquidity_usd:>10,.0f} | "
                f"Edge: {filter_result.net_edge:+5.1%}"
            )
        print("-" * 80)
    else:
        print("No opportunities found matching criteria.")

    # Show rejection analysis
    if result.tokens:
        analysis = filter.analyze_rejections(result.tokens)
        print(f"\nREJECTION ANALYSIS:")
        print(f"  Total: {analysis['total']}, Passed: {analysis['passed']} ({analysis['pass_rate']:.1%})")
        if analysis['rejection_reasons']:
            print(f"  Top rejection reasons:")
            for reason, count in list(analysis['rejection_reasons'].items())[:5]:
                print(f"    {reason}: {count}")


async def run_trading(capital: float, duration: int, paper_mode: bool):
    """Run the trading engine."""
    engine = ModularEngine(
        capital_usd=capital,
        paper_mode=paper_mode,
    )

    print(f"\n{'='*60}")
    print(f"  STARTING {'PAPER' if paper_mode else 'REAL'} TRADING")
    print(f"{'='*60}")
    print(f"  Capital:    ${capital:.2f}")
    print(f"  Duration:   {duration}s")
    print(f"  Momentum:   {engine.config.filters.min_momentum}%+")
    print(f"  Liquidity:  ${engine.config.filters.min_liquidity:,}+")
    print(f"{'='*60}\n")

    try:
        await engine.start()

        start = time.time()
        while time.time() - start < duration:
            await asyncio.sleep(10)

            stats = engine.get_stats()
            elapsed = int(time.time() - start)
            print(
                f"[{elapsed:3}s] "
                f"Trades: {stats['trades']:2} | "
                f"W/L: {stats['wins']}/{stats['losses']} | "
                f"PnL: ${stats['total_pnl_usd']:+.2f} ({stats['total_pnl_pct']:+.1%}) | "
                f"Scans: {stats['scans']}"
            )

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        await engine.stop()
        engine.print_stats()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Pump.fun Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --capital 10 --duration 300
  python main.py --show-config --capital 100
  python main.py --scan-only
        """,
    )

    parser.add_argument(
        "--capital",
        type=float,
        default=10.0,
        help="Starting capital in USD (default: 10)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=120,
        help="Run duration in seconds (default: 120)",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Show configuration and exit",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Scan for opportunities only, no trading",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.show_config:
        show_config(args.capital)
        return 0

    if args.scan_only:
        asyncio.run(scan_only())
        return 0

    asyncio.run(run_trading(args.capital, args.duration, paper_mode=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
