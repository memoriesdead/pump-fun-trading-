"""
RenTech-Style Trading Engine - Trade Frequently with Small Edges
================================================================

Key differences from ModularEngine:
1. Uses RenTechFilter instead of TokenFilter
2. No minimum momentum requirement
3. Trades more frequently (target: 50+ trades/day)
4. Smaller position sizes via Kelly Criterion
5. Let law of large numbers work

Usage:
    from trading.rentech_engine import RenTechEngine

    engine = RenTechEngine(capital_usd=10.0)
    await engine.start()
"""
import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .core.rentech_filter import RenTechFilter, RenTechFilterResult
from .core.token import Token
from .scanners.raydium import RaydiumScanner
from .core.filters import TokenFilter, FilterConfig

logger = logging.getLogger(__name__)


@dataclass
class RenTechPosition:
    """Open position with RenTech-style metadata."""
    token: Token
    entry_price: float
    entry_time: float
    size_usd: float
    filter_result: RenTechFilterResult

    @property
    def age_seconds(self) -> float:
        return time.time() - self.entry_time


@dataclass
class RenTechTrade:
    """Completed trade record."""
    token: Token
    entry_price: float
    exit_price: float
    entry_time: float
    exit_time: float
    size_usd: float
    pnl_usd: float
    pnl_pct: float
    exit_reason: str
    edge_at_entry: float
    confidence_at_entry: float


@dataclass
class RenTechStats:
    """Trading statistics."""
    total_scans: int = 0
    tokens_scanned: int = 0
    opportunities_found: int = 0
    trades_taken: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl_usd: float = 0.0

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0


class RenTechEngine:
    """
    RenTech-style trading engine.

    Key insight: It's better to make 1000 trades with 0.5% edge
    than 10 trades with 5% edge. Law of large numbers.

    Expected: 50.75% win rate with many trades = profit
    """

    def __init__(
        self,
        capital_usd: float = 10.0,
        paper_mode: bool = True,
        min_edge: float = 0.005,     # 0.5% minimum edge after friction
        max_positions: int = 5,       # More positions = more trades
        scan_interval_ms: int = 3000, # Faster scanning
        target_profit_pct: float = 0.05,  # 5% target (lower than before)
        max_loss_pct: float = 0.03,       # 3% stop (tighter)
        max_hold_ms: int = 30000,         # 30 second max hold (faster turnover)
    ):
        self.capital_usd = capital_usd
        self.initial_capital = capital_usd
        self.paper_mode = paper_mode

        # Trading params
        self.min_edge = min_edge
        self.max_positions = max_positions
        self.scan_interval_ms = scan_interval_ms
        self.target_profit_pct = target_profit_pct
        self.max_loss_pct = max_loss_pct
        self.max_hold_ms = max_hold_ms

        # Components
        self.filter = RenTechFilter(min_edge_after_friction=min_edge)
        self.legacy_filter = TokenFilter(FilterConfig())
        self.scanner = RaydiumScanner(
            filter=self.legacy_filter,  # Scanner uses legacy filter for basic parsing
            include_new=True,
            include_trending=True,
            include_top=True,
        )

        # State
        self.positions: Dict[str, RenTechPosition] = {}
        self.closed_trades: List[RenTechTrade] = []
        self.stats = RenTechStats()

        # Control
        self._running = False
        self._scan_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the engine."""
        self._running = True
        logger.info(
            f"[RENTECH] Engine started with ${self.capital_usd:.2f} "
            f"(min_edge={self.min_edge:.1%}, max_pos={self.max_positions})"
        )
        self._scan_task = asyncio.create_task(self._scan_loop())

    async def stop(self):
        """Stop the engine."""
        self._running = False
        if self._scan_task:
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass
        logger.info("[RENTECH] Engine stopped")

    async def _scan_loop(self):
        """Main scanning loop."""
        while self._running:
            try:
                # Scan for tokens
                result = await self.scanner.scan()
                self.stats.total_scans += 1
                self.stats.tokens_scanned += len(result.tokens)

                # Refilter with RenTech filter
                opportunities = self.filter.get_best_opportunities(
                    result.tokens,
                    max_results=10,
                    trade_size_usd=self.capital_usd * 0.1,  # 10% position
                )
                self.stats.opportunities_found += len(opportunities)

                if opportunities:
                    logger.info(
                        f"[RENTECH] Scan: {len(result.tokens)} tokens, "
                        f"{len(opportunities)} opportunities"
                    )

                    # Evaluate top opportunities
                    for token, filter_result in opportunities[:5]:
                        await self._evaluate_entry(token, filter_result)

                # Check exits
                await self._check_exits()

                # Wait before next scan
                await asyncio.sleep(self.scan_interval_ms / 1000)

            except Exception as e:
                logger.error(f"[RENTECH] Scan error: {e}")
                await asyncio.sleep(5)

    async def _evaluate_entry(self, token: Token, result: RenTechFilterResult):
        """Evaluate entry opportunity."""
        # Skip if already positioned
        if token.address in self.positions:
            return

        # Skip if at max positions
        if len(self.positions) >= self.max_positions:
            return

        # Use Kelly-derived position size
        position_size = min(
            self.capital_usd * result.position_size_pct,
            self.capital_usd * 0.10,  # Cap at 10% of capital
        )

        if position_size < 1.0:  # Minimum $1 trade
            return

        # Paper trade entry
        if self.paper_mode:
            await self._paper_enter(token, result, position_size)

    async def _paper_enter(
        self,
        token: Token,
        result: RenTechFilterResult,
        size_usd: float,
    ):
        """Execute paper entry."""
        position = RenTechPosition(
            token=token,
            entry_price=token.price,
            entry_time=time.time(),
            size_usd=size_usd,
            filter_result=result,
        )

        self.positions[token.address] = position
        self.capital_usd -= size_usd
        self.stats.trades_taken += 1

        logger.info(
            f"[RENTECH] BUY {token.symbol} @ ${token.price:.8f} | "
            f"Size: ${size_usd:.2f} | "
            f"Edge: {result.net_edge:.2%} | "
            f"Conf: {result.confidence:.0%}"
        )

    async def _check_exits(self):
        """Check exit conditions for all positions."""
        for address, position in list(self.positions.items()):
            # Simulate PnL
            simulated_pnl = self._simulate_pnl(position)

            should_exit = False
            exit_reason = ""

            # Take profit
            if simulated_pnl >= self.target_profit_pct:
                should_exit = True
                exit_reason = "take_profit"

            # Stop loss
            elif simulated_pnl <= -self.max_loss_pct:
                should_exit = True
                exit_reason = "stop_loss"

            # Max hold time
            elif position.age_seconds * 1000 >= self.max_hold_ms:
                should_exit = True
                exit_reason = "max_hold"

            if should_exit:
                await self._paper_exit(position, simulated_pnl, exit_reason)

    def _simulate_pnl(self, position: RenTechPosition) -> float:
        """
        Simulate PnL for paper trading.

        Model: Small edges (0.5-2%) with 50.75% win rate.
        """
        age = position.age_seconds
        edge = position.filter_result.net_edge
        confidence = position.filter_result.confidence

        # Simple model: edge materializes probabilistically
        # Over 30 seconds, expect edge to either work or not

        if age < 30:
            # Expected move scales with time
            time_factor = age / 30
            # Random walk with drift
            drift = edge * time_factor
            noise = 0.01 * time_factor * (0.5 - hash(str(position.entry_time)) % 100 / 100)
            simulated_move = drift + noise
        else:
            # After 30s, more likely to hit target or stop
            base = edge * confidence
            decay = (age - 30) / 60
            simulated_move = base * (1 - decay * 0.5)

        # Account for friction (round trip)
        friction = position.filter_result.friction_cost * 2

        return simulated_move - friction

    async def _paper_exit(
        self,
        position: RenTechPosition,
        pnl_pct: float,
        reason: str,
    ):
        """Execute paper exit."""
        exit_price = position.entry_price * (1 + pnl_pct)
        pnl_usd = position.size_usd * pnl_pct

        # Record trade
        trade = RenTechTrade(
            token=position.token,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_time=position.entry_time,
            exit_time=time.time(),
            size_usd=position.size_usd,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            edge_at_entry=position.filter_result.net_edge,
            confidence_at_entry=position.filter_result.confidence,
        )
        self.closed_trades.append(trade)

        # Update stats
        self.capital_usd += position.size_usd + pnl_usd
        self.stats.total_pnl_usd += pnl_usd
        if pnl_usd > 0:
            self.stats.wins += 1
        else:
            self.stats.losses += 1

        # Remove position
        del self.positions[position.token.address]

        logger.info(
            f"[RENTECH] SELL {position.token.symbol} @ ${exit_price:.8f} | "
            f"PnL: ${pnl_usd:+.2f} ({pnl_pct:+.1%}) | "
            f"Reason: {reason}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get trading statistics."""
        return {
            'paper_mode': self.paper_mode,
            'capital_usd': self.capital_usd,
            'initial_capital': self.initial_capital,
            'total_pnl_usd': self.stats.total_pnl_usd,
            'total_pnl_pct': (self.capital_usd - self.initial_capital) / self.initial_capital if self.initial_capital > 0 else 0,
            'trades': self.stats.trades_taken,
            'wins': self.stats.wins,
            'losses': self.stats.losses,
            'win_rate': self.stats.win_rate,
            'open_positions': len(self.positions),
            'scans': self.stats.total_scans,
            'tokens_scanned': self.stats.tokens_scanned,
            'opportunities': self.stats.opportunities_found,
        }

    def print_stats(self):
        """Print formatted statistics."""
        stats = self.get_stats()
        print(f"\n{'='*60}")
        print(f"  RENTECH TRADING STATS ({'PAPER' if stats['paper_mode'] else 'REAL'})")
        print(f"{'='*60}")
        print(f"  Capital: ${stats['capital_usd']:.2f} (started ${stats['initial_capital']:.2f})")
        print(f"  PnL: ${stats['total_pnl_usd']:+.2f} ({stats['total_pnl_pct']:+.1%})")
        print(f"  Trades: {stats['trades']} (W:{stats['wins']} L:{stats['losses']})")
        print(f"  Win Rate: {stats['win_rate']:.1%}")
        print(f"  Scans: {stats['scans']} ({stats['tokens_scanned']} tokens)")
        print(f"  Opportunities: {stats['opportunities']}")

        # Show RenTech comparison
        print(f"\n  RENTECH BENCHMARK:")
        print(f"  Target Win Rate: 50.75%")
        print(f"  Your Win Rate: {stats['win_rate']:.2%}")
        print(f"  At scale (1000 trades), edge compounds!")
        print(f"{'='*60}\n")


async def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="RenTech-Style Trading Engine")
    parser.add_argument("--capital", type=float, default=10.0, help="Starting capital USD")
    parser.add_argument("--duration", type=int, default=120, help="Run duration seconds")
    args = parser.parse_args()

    engine = RenTechEngine(capital_usd=args.capital)

    print(f"\n{'='*60}")
    print(f"  RENTECH-STYLE TRADING ENGINE")
    print(f"{'='*60}")
    print(f"  Capital: ${args.capital}")
    print(f"  Min Edge: {engine.min_edge:.1%} (NOT 50% momentum!)")
    print(f"  Max Positions: {engine.max_positions}")
    print(f"  Target: {engine.target_profit_pct:.0%} / Stop: {engine.max_loss_pct:.0%}")
    print(f"  Strategy: Many trades with small edges")
    print(f"{'='*60}\n")

    try:
        await engine.start()

        start = time.time()
        while time.time() - start < args.duration:
            await asyncio.sleep(10)

            stats = engine.get_stats()
            elapsed = int(time.time() - start)
            print(
                f"[{elapsed}s] Trades: {stats['trades']} | "
                f"W/L: {stats['wins']}/{stats['losses']} | "
                f"PnL: ${stats['total_pnl_usd']:+.2f} | "
                f"WR: {stats['win_rate']:.0%}"
            )

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        await engine.stop()
        engine.print_stats()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    asyncio.run(main())
