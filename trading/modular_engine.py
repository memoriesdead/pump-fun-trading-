"""
Modular Trading Engine - Clean Architecture
=============================================

Integrates:
- trading.core.token - Token dataclass
- trading.core.filters - TokenFilter for edge calculation
- trading.core.friction - Friction model
- trading.core.config - Configuration
- trading.scanners - Token discovery

Usage:
    from trading.modular_engine import ModularEngine

    engine = ModularEngine(capital_usd=10.0)
    await engine.run()
"""
import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .core.config import Config, FilterConfig, ScalpConfig, ENDPOINTS
from .core.friction import FRICTION
from .core.token import Token, TokenSource
from .core.filters import TokenFilter, FilterResult
from .scanners.raydium import RaydiumScanner
from .scanners.base import ScanResult

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Open position tracking."""
    token: Token
    entry_price: float
    entry_time: float
    size_usd: float
    filter_result: FilterResult

    @property
    def age_seconds(self) -> float:
        return time.time() - self.entry_time


@dataclass
class ClosedTrade:
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


@dataclass
class EngineStats:
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


class ModularEngine:
    """
    Clean modular trading engine.

    Uses the new modular components:
    - TokenFilter for opportunity detection
    - RaydiumScanner for token discovery
    - FrictionModel for realistic cost calculation
    """

    def __init__(
        self,
        capital_usd: float = 10.0,
        paper_mode: bool = True,
        config: Optional[Config] = None,
    ):
        """
        Initialize engine.

        Args:
            capital_usd: Starting capital in USD
            paper_mode: True for paper trading
            config: Optional configuration override
        """
        self.capital_usd = capital_usd
        self.initial_capital = capital_usd
        self.paper_mode = paper_mode
        self.config = config or Config.for_account_size(capital_usd)

        # Components
        self.filter = TokenFilter(self.config.filters)
        self.scanner = RaydiumScanner(
            filter=self.filter,
            include_new=True,
            include_trending=True,
            include_top=True,
        )

        # State
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[ClosedTrade] = []
        self.stats = EngineStats()

        # Control
        self._running = False
        self._scan_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the trading engine."""
        self._running = True
        logger.info(f"[{'PAPER' if self.paper_mode else 'REAL'}] Engine started with ${self.capital_usd:.2f}")
        self._scan_task = asyncio.create_task(self._scan_loop())

    async def stop(self):
        """Stop the trading engine."""
        self._running = False
        if self._scan_task:
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass
        logger.info("Engine stopped")

    async def _scan_loop(self):
        """Main scanning loop."""
        while self._running:
            try:
                # Scan for opportunities
                result = await self.scanner.scan_with_stats()
                self.stats.total_scans += 1
                self.stats.tokens_scanned += result.total_found
                self.stats.opportunities_found += result.total_passed

                # Log scan results
                if result.opportunities:
                    logger.info(
                        f"Scan: {result.total_found} tokens, "
                        f"{result.total_passed} passed filters"
                    )

                    # Evaluate top opportunities
                    for token, filter_result in result.opportunities[:3]:
                        await self._evaluate_entry(token, filter_result)

                # Check exits for open positions
                await self._check_exits()

                # Wait before next scan
                await asyncio.sleep(self.config.scalp.scan_interval_ms / 1000)

            except Exception as e:
                logger.error(f"Scan error: {e}")
                await asyncio.sleep(5)

    async def _evaluate_entry(self, token: Token, filter_result: FilterResult):
        """
        Evaluate whether to enter a position.

        Args:
            token: Token to evaluate
            filter_result: Pre-computed filter result
        """
        # Skip if already positioned
        if token.address in self.positions:
            return

        # Skip if at max positions
        if len(self.positions) >= self.config.scalp.max_open_positions:
            return

        # Check minimum edge requirement
        if filter_result.net_edge < self.config.scalp.min_expected_edge:
            logger.debug(f"Skip {token.symbol}: edge {filter_result.net_edge:.1%} < min")
            return

        # Calculate position size
        position_size = self._calculate_position_size(filter_result)

        if position_size < 1.0:  # Minimum $1 trade
            return

        # Enter position (paper mode)
        if self.paper_mode:
            await self._paper_enter(token, filter_result, position_size)
        else:
            # Real trading would go here
            logger.warning("Real trading not implemented")

    def _calculate_position_size(self, filter_result: FilterResult) -> float:
        """
        Calculate position size based on edge and Kelly criterion.

        Kelly: f* = (p*b - q) / b
        where p = win probability, b = win/loss ratio, q = 1-p
        """
        # Estimate win probability from edge
        # Higher edge = higher confidence
        edge = filter_result.net_edge

        # Conservative: assume edge translates to ~60% win rate at 5% edge
        # Scale linearly with edge (capped)
        win_prob = min(0.55 + edge * 0.5, 0.75)

        # Expected profit vs loss (target/stop ratio)
        target = self.config.scalp.target_profit_pct
        stop = self.config.scalp.max_loss_pct
        win_loss_ratio = target / stop if stop > 0 else 1.0

        # Kelly formula
        q = 1 - win_prob
        kelly = (win_prob * win_loss_ratio - q) / win_loss_ratio

        # Use fraction of Kelly for safety
        kelly_fraction = 0.25  # 1/4 Kelly

        # Calculate size
        max_size = self.capital_usd * self.config.scalp.max_position_pct
        size = min(self.capital_usd * kelly * kelly_fraction, max_size)

        return max(0, size)

    async def _paper_enter(
        self,
        token: Token,
        filter_result: FilterResult,
        size_usd: float,
    ):
        """Execute paper entry."""
        position = Position(
            token=token,
            entry_price=token.price,
            entry_time=time.time(),
            size_usd=size_usd,
            filter_result=filter_result,
        )

        self.positions[token.address] = position
        self.capital_usd -= size_usd
        self.stats.trades_taken += 1

        logger.info(
            f"BUY {token.symbol} @ ${token.price:.8f} | "
            f"Size: ${size_usd:.2f} | "
            f"Edge: {filter_result.net_edge:.1%} | "
            f"Mom: {token.momentum:+.1f}%"
        )

    async def _check_exits(self):
        """Check exit conditions for all positions."""
        config = self.config.scalp

        for address, position in list(self.positions.items()):
            # In paper mode, simulate price movement based on momentum
            # Real mode would fetch actual price
            simulated_pnl = self._simulate_pnl(position)

            should_exit = False
            exit_reason = ""

            # Check take profit
            if simulated_pnl >= config.target_profit_pct:
                should_exit = True
                exit_reason = "take_profit"

            # Check stop loss
            elif simulated_pnl <= -config.max_loss_pct:
                should_exit = True
                exit_reason = "stop_loss"

            # Check max hold time
            elif position.age_seconds * 1000 >= config.max_hold_ms:
                should_exit = True
                exit_reason = "max_hold"

            if should_exit:
                await self._paper_exit(position, simulated_pnl, exit_reason)

    def _simulate_pnl(self, position: Position) -> float:
        """
        Simulate PnL for paper trading.

        Model: momentum tends to continue in first 30s, then mean reverts.
        """
        age = position.age_seconds
        momentum = position.token.momentum / 100  # Convert to decimal

        # Momentum continuation model
        if age < 30:
            # First 30s: momentum continues at ~40% of rate
            continuation = 0.4
            simulated_move = momentum * continuation * (age / 30)
        else:
            # After 30s: momentum exhaustion, add noise
            continuation = 0.4
            base_move = momentum * continuation
            # Mean reversion factor
            reversion = (age - 30) / 60 * 0.5
            simulated_move = base_move * (1 - reversion)

        # Subtract friction for round trip
        friction = position.filter_result.friction_cost * 2

        return simulated_move - friction

    async def _paper_exit(
        self,
        position: Position,
        pnl_pct: float,
        reason: str,
    ):
        """Execute paper exit."""
        exit_price = position.entry_price * (1 + pnl_pct)
        pnl_usd = position.size_usd * pnl_pct

        # Record trade
        trade = ClosedTrade(
            token=position.token,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_time=position.entry_time,
            exit_time=time.time(),
            size_usd=position.size_usd,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            exit_reason=reason,
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
            f"SELL {position.token.symbol} @ ${exit_price:.8f} | "
            f"PnL: ${pnl_usd:+.2f} ({pnl_pct:+.1%}) | "
            f"Reason: {reason}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
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
        print(f"\n{'='*50}")
        print(f"  TRADING STATS ({'PAPER' if stats['paper_mode'] else 'REAL'})")
        print(f"{'='*50}")
        print(f"  Capital: ${stats['capital_usd']:.2f} (started ${stats['initial_capital']:.2f})")
        print(f"  PnL: ${stats['total_pnl_usd']:+.2f} ({stats['total_pnl_pct']:+.1%})")
        print(f"  Trades: {stats['trades']} (W:{stats['wins']} L:{stats['losses']})")
        print(f"  Win Rate: {stats['win_rate']:.1%}")
        print(f"  Scans: {stats['scans']} ({stats['tokens_scanned']} tokens)")
        print(f"  Opportunities: {stats['opportunities']}")
        print(f"{'='*50}\n")


async def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Modular Trading Engine")
    parser.add_argument("--capital", type=float, default=10.0, help="Starting capital USD")
    parser.add_argument("--duration", type=int, default=120, help="Run duration seconds")
    parser.add_argument("--real", action="store_true", help="Real trading mode (disabled)")
    args = parser.parse_args()

    # Initialize engine
    engine = ModularEngine(
        capital_usd=args.capital,
        paper_mode=not args.real,
    )

    # Print config
    print(f"\nStarting Modular Engine")
    print(f"  Capital: ${args.capital}")
    print(f"  Mode: {'PAPER' if engine.paper_mode else 'REAL'}")
    print(f"  Duration: {args.duration}s")
    print(f"  Min Momentum: {engine.config.filters.min_momentum}%")
    print(f"  Min Liquidity: ${engine.config.filters.min_liquidity:,}")
    print()

    try:
        await engine.start()

        # Run for duration
        start = time.time()
        while time.time() - start < args.duration:
            await asyncio.sleep(10)

            # Print progress
            stats = engine.get_stats()
            elapsed = int(time.time() - start)
            print(
                f"[{elapsed}s] Trades: {stats['trades']} | "
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
