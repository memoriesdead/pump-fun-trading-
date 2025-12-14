#!/usr/bin/env python3
"""
RENTECH UNIFIED TRADING ENGINE
==============================

"Every token on pump.fun is an opportunity."

Single entry point for all modes:
- collect: Capture all data (run on VPS 24/7)
- paper: Paper trading with real data
- live: Real money execution (not implemented yet)

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    UNIFIED ENGINE                                │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │   PumpPortal WebSocket                                          │
    │          │                                                       │
    │          ├──→ Data Lake (every trade stored)                    │
    │          │                                                       │
    │          ├──→ Wallet Tracker (smart money detection)            │
    │          │                                                       │
    │          ├──→ Opportunity Scanner (edge calculation)            │
    │          │                                                       │
    │          └──→ Executor (paper or live)                          │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘

Usage:
    # Mode 1: Collect data only (run on VPS 24/7)
    python trading/run.py --mode collect

    # Mode 2: Paper trading
    python trading/run.py --mode paper --capital 100 --duration 3600

    # Mode 3: Live trading (requires setup)
    python trading/run.py --mode live --capital 100
"""

import asyncio
import json
import time
import sys
import signal
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

# Platform-specific event loop policy
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

PUMPPORTAL_WS = "wss://pumpportal.fun/api/data"
STATS_INTERVAL = 60  # Log stats every minute
TOP_TOKENS_REFRESH = 30  # Refresh top tokens every 30 seconds (aggressive but within API limits)
MAX_TOP_TOKENS = 500  # Subscribe to top 500 volume tokens


# =============================================================================
# UNIFIED ENGINE
# =============================================================================

class UnifiedEngine:
    """
    RenTech-style unified trading engine.

    Combines:
    - Universal data collector
    - Smart wallet tracker
    - Opportunity scanner
    - Paper/live executor

    All in one seamless flow.
    """

    def __init__(
        self,
        mode: str = "paper",
        capital: float = 100.0,
        min_edge: float = 0.55,
        max_positions: int = 5,
        target_pct: float = 0.10,
        stop_pct: float = 0.10,
        max_hold_secs: int = 300
    ):
        self.mode = mode
        self.capital = capital

        # Import components
        from data.lake import get_lake
        from data.wallet_tracker import WalletTracker
        from trading.edge_calculator import EdgeCalculator, ExitCalculator
        from trading.opportunity_scanner import OpportunityScanner
        from trading.executor import PaperExecutor

        # Initialize components
        self.lake = get_lake()
        self.wallet_tracker = WalletTracker()
        self.edge_calc = EdgeCalculator(wallet_tracker=self.wallet_tracker)
        self.exit_calc = ExitCalculator(
            target_pct=target_pct,
            stop_pct=stop_pct,
            max_hold_secs=max_hold_secs
        )
        self.scanner = OpportunityScanner(
            edge_calculator=self.edge_calc,
            exit_calculator=self.exit_calc,
            min_edge=min_edge,
            max_positions=max_positions
        )

        # Choose executor based on mode
        if mode == "live":
            from trading.rentech_executor import RentechExecutor

            # Load config
            config_path = Path(__file__).parent.parent / ".wallet" / "config.json"
            wallet_path = Path(__file__).parent.parent / ".wallet" / "keypair.json"

            if not wallet_path.exists():
                raise ValueError("No wallet found. Run wallet setup first.")

            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                rpc_url = config.get('rpc_url', '')
            else:
                raise ValueError("No config found at .wallet/config.json")

            if "YOUR_HELIUS_API_KEY" in rpc_url:
                raise ValueError("Please update .wallet/config.json with your Helius API key")

            # RenTech executor: tracks REAL P&L from wallet balance
            # Higher target, tighter stop to overcome fees
            self.executor = RentechExecutor(
                wallet_path=str(wallet_path),
                rpc_url=rpc_url,
                capital=capital,
                target_pct=0.15,    # Higher target to overcome ~5% costs
                stop_pct=0.08,      # Tighter stop to limit damage
                max_hold_secs=max_hold_secs,
                slippage_bps=200,   # Tighter slippage (2%)
            )
            self._needs_init = True
        else:
            self.executor = PaperExecutor(
                capital=capital,
                target_pct=target_pct,
                stop_pct=stop_pct,
                max_hold_secs=max_hold_secs
            )
            self._needs_init = False

        # WebSocket state
        self._ws = None
        self._running = False
        self.subscribed = set()

        # Top volume tokens (RenTech style: highest volume first)
        self.top_tokens = []  # List of (address, volume) tuples sorted by volume
        self.top_token_volumes = {}  # address -> volume for priority sorting
        self._last_top_token_refresh = 0

        # Statistics
        self.trades_seen = 0
        self.tokens_seen = 0
        self.start_time = time.time()

    async def fetch_top_tokens(self):
        """
        Fetch top tokens by volume - RenTech style.

        Highest volume = most opportunity.
        """
        try:
            from data.top_tokens import TopTokenFetcher

            logger.info("Fetching top volume tokens (RenTech style)...")

            fetcher = TopTokenFetcher()

            # Fetch from DexScreener (free, no API key needed)
            all_tokens = []

            # 1. Boosted tokens (high attention)
            boosted = await fetcher.fetch_dexscreener_boosted()
            all_tokens.extend(boosted)

            # 2. Search for popular Solana tokens (expanded for 500 token coverage)
            searched = await fetcher.fetch_dexscreener_search([
                # Top pump.fun tokens
                "pump", "fun", "SOL", "BONK", "WIF", "POPCAT", "MEW",
                "SLERF", "MYRO", "FARTCOIN", "GOAT", "PNUT", "MOODENG",
                "AI16Z", "GRIFFAIN", "ZEREBRO", "meme", "degen", "moon",
                # More meme tokens
                "PEPE", "DOGE", "SHIB", "FLOKI", "WOJAK", "CHAD", "GIGA",
                "PONKE", "BOME", "BRETT", "TURBO", "LADYS", "MOCHI",
                # Solana ecosystem
                "RAY", "JUP", "ORCA", "PYTH", "JTO", "TENSOR", "HONEY",
                "RENDER", "HNT", "MOBILE", "DUST", "FORGE", "BLZE",
                # Generic high volume
                "solana", "raydium", "jupiter", "meteora", "ape", "rocket",
                # Letter searches for broad coverage
                "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
                "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x"
            ])
            all_tokens.extend(searched)

            # Deduplicate and sort by volume
            seen = set()
            unique_tokens = []
            for t in all_tokens:
                if t.address and t.address not in seen:
                    seen.add(t.address)
                    unique_tokens.append(t)

            # Sort by volume + liquidity (highest first)
            unique_tokens.sort(key=lambda x: (x.volume_24h + x.liquidity), reverse=True)

            # Take top N
            self.top_tokens = [(t.address, t.volume_24h + t.liquidity) for t in unique_tokens[:MAX_TOP_TOKENS]]
            self.top_token_volumes = {addr: vol for addr, vol in self.top_tokens}

            logger.info(f"Loaded {len(self.top_tokens)} top volume tokens")

            # Log top 5
            for i, (addr, vol) in enumerate(self.top_tokens[:5], 1):
                t = next((x for x in unique_tokens if x.address == addr), None)
                if t:
                    logger.info(f"  {i}. {t.symbol}: ${vol:,.0f} volume+liq")

            self._last_top_token_refresh = time.time()

        except Exception as e:
            logger.warning(f"Error fetching top tokens: {e}")
            # Load from cache if available
            try:
                from data.top_tokens import TopTokenFetcher
                cache_path = Path(__file__).parent.parent / "data" / "top_tokens.json"
                if cache_path.exists():
                    tokens = TopTokenFetcher.load(str(cache_path))
                    self.top_tokens = [(t.address, t.volume_24h + t.liquidity) for t in tokens[:MAX_TOP_TOKENS]]
                    self.top_token_volumes = {addr: vol for addr, vol in self.top_tokens}
                    logger.info(f"Loaded {len(self.top_tokens)} tokens from cache")
            except Exception as e2:
                logger.warning(f"Cache load failed: {e2}")

    async def subscribe_top_tokens(self):
        """Subscribe to top volume tokens."""
        if not self._ws or not self.top_tokens:
            return

        # Get addresses to subscribe (not already subscribed)
        to_subscribe = [addr for addr, _ in self.top_tokens if addr not in self.subscribed]

        if not to_subscribe:
            return

        # Subscribe in batches of 50
        for i in range(0, len(to_subscribe), 50):
            batch = to_subscribe[i:i+50]
            try:
                await self._ws.send_json({
                    "method": "subscribeTokenTrade",
                    "keys": batch
                })
                self.subscribed.update(batch)
                logger.info(f"Subscribed to {len(batch)} top volume tokens (batch {i//50 + 1})")
            except Exception as e:
                logger.warning(f"Subscribe batch error: {e}")

            await asyncio.sleep(0.5)  # Rate limit

    async def run(self, duration: int = 0):
        """
        Main run loop.

        Args:
            duration: Seconds to run (0 = forever)
        """
        self._running = True
        self._print_header()

        # Initialize RenTech executor (get wallet balance)
        if self._needs_init:
            await self.executor.initialize()

        # RENTECH: Fetch top volume tokens BEFORE connecting
        await self.fetch_top_tokens()

        end_time = time.time() + duration if duration > 0 else float('inf')

        try:
            import aiohttp
        except ImportError:
            logger.error("aiohttp required: pip install aiohttp")
            return

        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    PUMPPORTAL_WS,
                    heartbeat=30,
                    receive_timeout=60
                ) as ws:
                    self._ws = ws
                    logger.info("Connected to PumpPortal")

                    # Subscribe to new token firehose (catch new opportunities)
                    await ws.send_json({"method": "subscribeNewToken"})
                    logger.info("Subscribed to new token firehose")

                    # RENTECH: Subscribe to TOP VOLUME tokens (where the money is)
                    await self.subscribe_top_tokens()
                    logger.info(f"Subscribed to {len(self.top_tokens)} high-volume tokens")

                    # Start stats logger
                    stats_task = asyncio.create_task(self._stats_loop())

                    # Start top token refresh task
                    refresh_task = asyncio.create_task(self._top_token_refresh_loop())

                    try:
                        async for msg in ws:
                            if not self._running:
                                break

                            if time.time() > end_time:
                                break

                            if msg.type == aiohttp.WSMsgType.TEXT:
                                await self._process_message(msg.data)

                    finally:
                        stats_task.cancel()
                        refresh_task.cancel()
                        try:
                            await stats_task
                        except asyncio.CancelledError:
                            pass
                        try:
                            await refresh_task
                        except asyncio.CancelledError:
                            pass

        except Exception as e:
            logger.error(f"Engine error: {e}")

        finally:
            self._print_results()

    async def _process_message(self, data: str):
        """Process WebSocket message."""
        try:
            msg = json.loads(data)
        except:
            return

        if not isinstance(msg, dict):
            return

        tx_type = msg.get('txType', '')

        # New token created
        if tx_type == 'create':
            await self._on_new_token(msg)

        # Trade event
        elif tx_type in ('buy', 'sell'):
            await self._on_trade(msg)

    async def _on_new_token(self, msg: dict):
        """Handle new token creation."""
        mint = msg.get('mint', '')
        if not mint or mint in self.subscribed:
            return

        # Rotate subscriptions at limit
        if len(self.subscribed) >= 500:
            oldest = next(iter(self.subscribed))
            self.subscribed.discard(oldest)

        try:
            await self._ws.send_json({
                "method": "subscribeTokenTrade",
                "keys": [mint]
            })
            self.subscribed.add(mint)
            self.tokens_seen += 1

            # Store token
            self.lake.upsert_token({
                'mint': mint,
                'symbol': msg.get('symbol', ''),
                'name': msg.get('name', ''),
                'creator': msg.get('traderPublicKey', ''),
                'created_at': time.time()
            })

        except Exception as e:
            logger.warning(f"Subscribe error: {e}")

    async def _on_trade(self, msg: dict):
        """Handle trade event."""
        # Parse trade
        try:
            sol_amount = float(msg.get('solAmount', 0)) / 1e9
            token_amount = float(msg.get('tokenAmount', 0))

            if token_amount <= 0:
                return

            price = sol_amount / token_amount
            mint = msg.get('mint', '')
            trader = msg.get('traderPublicKey', '')
            side = msg.get('txType', 'buy')

            trade = {
                'timestamp': float(msg.get('timestamp', 0)) or time.time() * 1000,
                'mint': mint,
                'signature': msg.get('signature', ''),
                'trader': trader,
                'side': side,
                'sol_amount': sol_amount,
                'token_amount': token_amount,
                'price_sol': price,
                'market_cap_sol': float(msg.get('marketCapSol', 0)),
                'slot': int(msg.get('slot', 0)),
                'symbol': msg.get('symbol', mint[:8])
            }

        except Exception:
            return

        self.trades_seen += 1

        # Store trade (collect mode or always)
        if self.mode in ('collect', 'paper', 'live'):
            self.lake.ingest_trade(trade)

        # Skip trading logic if collect-only mode
        if self.mode == 'collect':
            return

        # Update wallet tracker
        self.wallet_tracker.update_from_trade(trade)

        # Update scanner state
        self.scanner.update_token(mint, trade)

        # Check exit conditions for open positions
        if mint in self.executor.positions:
            exit_reason = self.executor.check_exit_conditions(mint, price)
            if exit_reason:
                await self.executor.exit(mint, price, exit_reason)

        # Check for new entry opportunities
        elif len(self.executor.positions) < self.scanner.max_positions:
            opportunities = self.scanner.scan()

            # RENTECH: Prioritize by volume (highest volume = most liquidity = best execution)
            if opportunities and self.top_token_volumes:
                # Sort by: volume * edge (high volume + high edge = best opportunity)
                opportunities.sort(
                    key=lambda o: self.top_token_volumes.get(o.mint, 0) * o.edge,
                    reverse=True
                )

            for opp in opportunities[:1]:  # Enter best opportunity
                if opp.mint == mint:  # Only enter on fresh trade
                    # Log if it's a high-volume token
                    volume = self.top_token_volumes.get(mint, 0)
                    if volume > 0:
                        logger.info(f"HIGH VOLUME: {opp.symbol} (${volume:,.0f} vol+liq)")
                    await self.executor.enter(opp, price)

    async def _stats_loop(self):
        """Log statistics periodically."""
        while self._running:
            await asyncio.sleep(STATS_INTERVAL)
            self._log_stats()

    async def _top_token_refresh_loop(self):
        """Periodically refresh top volume tokens."""
        while self._running:
            await asyncio.sleep(TOP_TOKENS_REFRESH)
            try:
                await self.fetch_top_tokens()
                await self.subscribe_top_tokens()
            except Exception as e:
                logger.warning(f"Top token refresh error: {e}")

    def _log_stats(self):
        """Log current statistics."""
        elapsed = (time.time() - self.start_time) / 60
        stats = self.executor.get_stats()

        if self.mode == 'collect':
            lake_stats = self.lake.stats()
            logger.info(
                f"[{elapsed:.1f}m] "
                f"Trades: {self.trades_seen:,} | "
                f"Tokens: {self.tokens_seen:,} | "
                f"Lake: {lake_stats['trades']:,}"
            )
        elif self.mode == 'live':
            # Show REAL P&L from wallet balance
            rejected = stats.get('trades_rejected', 0)
            cost = stats.get('measured_cost', 0)
            logger.info(
                f"[{elapsed:.1f}m] "
                f"Tokens: {self.tokens_seen:,} | "
                f"Pos: {stats['positions_open']}/{self.scanner.max_positions} | "
                f"Trades: {stats['trades']} (skip:{rejected}) | "
                f"WR: {stats['win_rate']:.1%} | "
                f"REAL PnL: {stats['pnl']:+.6f} SOL ({stats.get('real_pnl_pct', 0):+.1%}) | "
                f"Cost: {cost:.1%}"
            )
        else:
            logger.info(
                f"[{elapsed:.1f}m] "
                f"Trades: {self.trades_seen:,} | "
                f"Tokens: {self.tokens_seen:,} | "
                f"Pos: {stats['positions_open']}/{self.scanner.max_positions} | "
                f"Completed: {stats['trades']} | "
                f"WR: {stats['win_rate']:.1%} | "
                f"PnL: ${stats['pnl']:+.2f}"
            )

    def _print_header(self):
        """Print header."""
        print("\n" + "=" * 70)
        print("RENTECH UNIFIED ENGINE")
        print("=" * 70)

        if self.mode == "live":
            print("*" * 70)
            print("***  LIVE MODE - REAL MONEY (COST-AWARE)  ***")
            print("*" * 70)
            print(f"Wallet: {self.executor.pubkey}")
            print(f"Target: +15% / Stop: -8% (asymmetric for fees)")
            print(f"Slippage: 2% max")
            print(f"ONLY trades when: edge > costs + 2%")
        elif self.mode == "paper":
            print("*" * 70)
            print("***  PAPER MODE - REALISTIC COSTS INCLUDED  ***")
            print("*" * 70)
            print("Costs simulated (matches real trading):")
            print("  - Pump.fun fee: 1% buy + 1% sell = 2%")
            print("  - Slippage: ~3% buy + 3% sell = 6%")
            print("  - TOTAL ROUND TRIP: ~8%")
            print("  - Need +8% price move just to BREAK EVEN")

        print(f"Mode: {self.mode.upper()}")
        print(f"Capital: {self.capital:.4f} SOL" if self.mode == "live" else f"Capital: ${self.capital:.2f}")
        print(f"Min Edge: {self.scanner.min_edge:.0%}")
        print(f"Max Positions: {self.scanner.max_positions}")
        if self.mode != "live":
            print(f"Target: +{self.exit_calc.target_pct:.0%} / Stop: -{self.exit_calc.stop_pct:.0%}")
            if self.mode == "paper":
                net_win = self.exit_calc.target_pct - 0.08
                net_loss = -self.exit_calc.stop_pct - 0.08
                print(f"  Net after costs: Win +{net_win:.0%} / Loss {net_loss:.0%}")
        print("-" * 70)
        print("RENTECH DATA SOURCES:")
        print("  - NEW TOKENS: PumpPortal firehose (catch fresh launches)")
        print(f"  - HIGH VOLUME: Top {MAX_TOP_TOKENS} tokens by volume (where money flows)")
        print("  - Priority: volume * edge (liquidity + edge = best execution)")
        print("-" * 70)
        print("EXPLOSIVE TRADE REQUIREMENTS:")
        print("  - Min 2 signals confirming (no single-signal trades)")
        print("  - Volume spike 2x+ average")
        print("  - Buy pressure 65%+ in last 10 trades")
        print("  - Momentum 3%+ recent gain")
        print("  - Token age < 10 minutes (fresh)")
        print("-" * 70)
        print("PATTERNS: REVERSAL (96%) | MOMENTUM (82.8%) | BUY_PRESSURE (81.8%)")
        print("          + 15% live discount applied (historical != live)")
        print("=" * 70 + "\n")

    def _print_results(self):
        """Print final results."""
        print("\n")

        if self.mode == 'collect':
            lake_stats = self.lake.stats()
            elapsed = (time.time() - self.start_time) / 60

            print("=" * 70)
            print("DATA COLLECTION RESULTS")
            print("=" * 70)
            print(f"Duration: {elapsed:.1f} minutes")
            print(f"Trades Captured: {self.trades_seen:,}")
            print(f"Tokens Seen: {self.tokens_seen:,}")
            print(f"Lake Total: {lake_stats['trades']:,}")
            print(f"Database Size: {lake_stats['db_size_mb']:.2f} MB")
            print("=" * 70)

        else:
            self.executor.print_results()


# =============================================================================
# CLI
# =============================================================================

async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="RenTech Unified Trading Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect data only (24/7 VPS mode)
  python trading/run.py --mode collect

  # Paper trading for 1 hour
  python trading/run.py --mode paper --capital 100 --duration 3600

  # Paper trading with custom settings
  python trading/run.py --mode paper --capital 500 --min-edge 0.60 --max-positions 3
        """
    )

    parser.add_argument(
        "--mode", type=str, default="paper",
        choices=["collect", "paper", "live"],
        help="Operating mode (default: paper)"
    )
    parser.add_argument(
        "--capital", type=float, default=100.0,
        help="Starting capital in USD (default: 100)"
    )
    parser.add_argument(
        "--duration", type=int, default=0,
        help="Duration in seconds (0 = forever)"
    )
    parser.add_argument(
        "--min-edge", type=float, default=0.80,
        help="Minimum edge threshold (default: 0.80 for profit after 8%% costs)"
    )
    parser.add_argument(
        "--max-positions", type=int, default=5,
        help="Maximum concurrent positions (default: 5)"
    )
    parser.add_argument(
        "--target", type=float, default=0.15,
        help="Target profit percent (default: 0.15 for +7%% net after 8%% costs)"
    )
    parser.add_argument(
        "--stop", type=float, default=0.08,
        help="Stop loss percent (default: 0.08 for -16%% net after 8%% costs)"
    )
    parser.add_argument(
        "--max-hold", type=int, default=300,
        help="Max hold time in seconds (default: 300)"
    )

    args = parser.parse_args()

    # Create engine
    engine = UnifiedEngine(
        mode=args.mode,
        capital=args.capital,
        min_edge=args.min_edge,
        max_positions=args.max_positions,
        target_pct=args.target,
        stop_pct=args.stop,
        max_hold_secs=args.max_hold
    )

    # Run
    try:
        await engine.run(duration=args.duration)
    except KeyboardInterrupt:
        print("\nShutting down...")
        engine._running = False


if __name__ == "__main__":
    asyncio.run(main())
