"""
RenTech Trading Engine v3.1
===========================

Helius LaserStream + Jito Bundles + VALIDATED Formulas = The Edge

Architecture:
- LaserStream: <50ms data latency (vs 200-500ms WebSocket)
- Jito: Priority execution with tips
- VALIDATED signals from 269,830 historical trades

"We're right 50.75% of the time, but we're 100% right 50.75% of the time."

VALIDATED ENTRY SIGNALS (Dec 2024 backtest):
- PF-520: Mean Reversion - 82.5% win rate (26,345 signals) - PRIMARY
- PF-530: Buy Pressure   - 52.8% win rate (34,146 signals) - SECONDARY

VALIDATED EXIT SIGNALS:
- PF-511: Volume Dry-Up  - 62.4% win rate (5,838 signals)

REMOVED (INVALIDATED - were causing losses):
- Volume Spike: 9% win rate - FAILS
- Whale Following: 15.2% win rate, -22.8% avg return - LOSES MONEY

Usage:
    # Paper trading
    python trading/rentech_v3.py --paper --duration 3600

    # Live trading
    python trading/rentech_v3.py --capital 100 --duration 0
"""

import asyncio
import argparse
import os
import sys
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.helius_feed import HeliusFeed, PumpTrade, TokenCreate
from trading.jito_executor import JitoExecutor, TipLevel, ExecutionResult

# Import VALIDATED formulas from historical data discovery (269,830 trades backtested)
from formulas.pumpfun_formulas import (
    PricePoint,
    pf520_mean_reversion,      # 82.5% win rate - BEST PERFORMER
    pf530_buy_pressure,        # 52.8% win rate
    pf511_volume_dryup_warning,  # 62.4% win rate - EXIT signal
    # NOTE: PF-510 (volume spike) and PF-512 (whale following) are INVALIDATED
    # PF-510: 9% win rate - FAILS
    # PF-512: 15.2% win rate, -22.8% avg return - LOSES MONEY
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class TokenState:
    """Track state for a single token"""
    mint: str
    created_at: datetime

    # Bonding curve state
    virtual_sol: float = 30.0
    virtual_tokens: float = 1_000_000_000

    # Trade history (last 100)
    trades: deque = field(default_factory=lambda: deque(maxlen=100))

    # Aggregated signals
    total_buys: int = 0
    total_sells: int = 0
    total_buy_sol: float = 0.0
    total_sell_sol: float = 0.0

    # Derived metrics (updated on each trade)
    price: float = 0.0
    mcap_sol: float = 0.0
    sol_flow_1m: float = 0.0
    sol_flow_5m: float = 0.0
    buy_pressure: float = 0.5
    avg_trade_size: float = 0.0
    whale_trades_5m: int = 0

    def update(self, trade: PumpTrade):
        """Update state with new trade"""
        self.trades.append(trade)

        # Update curve
        self.virtual_sol = trade.virtual_sol
        self.virtual_tokens = trade.virtual_tokens
        self.price = self.virtual_sol / max(1, self.virtual_tokens)
        self.mcap_sol = self.price * 1_000_000_000

        # Update aggregates
        if trade.is_buy:
            self.total_buys += 1
            self.total_buy_sol += trade.sol_amount
        else:
            self.total_sells += 1
            self.total_sell_sol += trade.sol_amount

        # Calculate rolling metrics
        self._update_rolling_metrics()

    def _update_rolling_metrics(self):
        """Update time-windowed metrics"""
        now = datetime.now()

        # 1-minute window
        sol_flow_1m = 0.0
        # 5-minute window
        sol_flow_5m = 0.0
        whale_count = 0

        for trade in self.trades:
            age = (now - trade.timestamp).total_seconds()
            sol = trade.sol_amount if trade.is_buy else -trade.sol_amount

            if age <= 60:
                sol_flow_1m += sol
            if age <= 300:
                sol_flow_5m += sol
                if trade.sol_amount >= 5.0:  # Whale = 5+ SOL
                    whale_count += 1

        self.sol_flow_1m = sol_flow_1m
        self.sol_flow_5m = sol_flow_5m
        self.whale_trades_5m = whale_count

        # Buy pressure
        total_trades = self.total_buys + self.total_sells
        if total_trades > 0:
            self.buy_pressure = self.total_buys / total_trades

        # Average trade size
        if len(self.trades) > 0:
            self.avg_trade_size = sum(t.sol_amount for t in self.trades) / len(self.trades)

    @property
    def curve_progress(self) -> float:
        """Progress through bonding curve (0-100%)"""
        # Curve starts at 30 SOL, graduates at 85 SOL
        return min(100, max(0, (self.virtual_sol - 30) / 55 * 100))

    def to_price_points(self) -> List[PricePoint]:
        """
        Convert trade history to PricePoint format for validated formulas.

        This bridges the gap between our live trade data and the formula
        interface that was validated on 269,830 historical trades.
        """
        points = []
        for t in self.trades:
            price = t.virtual_sol / max(1, t.virtual_tokens)
            points.append(PricePoint(
                timestamp=t.timestamp.timestamp(),
                price=price,
                virtual_sol=t.virtual_sol,
                virtual_tokens=t.virtual_tokens,
                sol_amount=t.sol_amount,
                is_buy=t.is_buy
            ))
        return points


@dataclass
class Position:
    """Active trading position"""
    mint: str
    entry_price: float
    entry_time: datetime
    token_amount: float
    sol_invested: float

    # For tracking
    peak_price: float = 0.0
    current_price: float = 0.0

    def __post_init__(self):
        self.peak_price = self.entry_price
        self.current_price = self.entry_price

    @property
    def pnl_pct(self) -> float:
        """Current P&L percentage"""
        if self.entry_price <= 0:
            return 0
        return (self.current_price - self.entry_price) / self.entry_price * 100

    @property
    def hold_time_sec(self) -> float:
        """Seconds since entry"""
        return (datetime.now() - self.entry_time).total_seconds()


@dataclass
class TradeResult:
    """Completed trade for stats"""
    mint: str
    is_win: bool
    pnl_pct: float
    pnl_sol: float
    hold_time_sec: float
    entry_time: datetime
    exit_time: datetime
    exit_reason: str


class RenTechV3:
    """
    RenTech-style trading engine with Helius + Jito infrastructure.

    The edge:
    - See trades 400ms faster than WebSocket users
    - Execute with priority via Jito bundles
    - Same simple signals, but we can actually act on them now
    """

    def __init__(
        self,
        capital: float = 100.0,
        target_pct: float = 0.05,
        stop_pct: float = 0.05,
        min_win_prob: float = 0.52,
        max_positions: int = 3,
        max_hold_sec: float = 300.0,
        position_size_pct: float = 0.10,
        paper: bool = True,
        helius_api_key: Optional[str] = None,
        keypair_path: Optional[str] = None
    ):
        # Capital
        self.initial_capital = capital
        self.capital = capital

        # Risk parameters
        self.target_pct = target_pct
        self.stop_pct = stop_pct
        self.min_win_prob = min_win_prob
        self.max_positions = max_positions
        self.max_hold_sec = max_hold_sec
        self.position_size_pct = position_size_pct

        # Mode
        self.paper = paper

        # Infrastructure
        self.feed = HeliusFeed(api_key=helius_api_key)
        self.executor = JitoExecutor(keypair_path=keypair_path)

        # State
        self.tokens: Dict[str, TokenState] = {}
        self.positions: Dict[str, Position] = {}
        self.completed_trades: List[TradeResult] = []

        # Stats
        self.trades_seen = 0
        self.signals_generated = 0
        self.start_time: Optional[datetime] = None

        # Running flag
        self._running = False

    async def start(self, duration_sec: float = 0):
        """
        Start the trading engine.

        Args:
            duration_sec: How long to run (0 = forever)
        """
        self.start_time = datetime.now()
        self._running = True

        # Connect infrastructure
        logger.info("=" * 60)
        logger.info("RENTECH V3.1 - VALIDATED FORMULAS INTEGRATED")
        logger.info("=" * 60)
        logger.info(f"Mode: {'PAPER' if self.paper else 'LIVE'}")
        logger.info(f"Capital: ${self.capital:.2f}")
        logger.info(f"Target: +{self.target_pct*100:.1f}% / Stop: -{self.stop_pct*100:.1f}%")
        logger.info(f"Min Win Prob: {self.min_win_prob*100:.1f}%")
        logger.info(f"Max Positions: {self.max_positions}")
        logger.info("-" * 60)
        logger.info("ENTRY SIGNALS (from 269,830 trade backtest):")
        logger.info("  PF-520: Mean Reversion - 82.5% win rate")
        logger.info("  PF-530: Buy Pressure   - 52.8% win rate")
        logger.info("EXIT SIGNALS:")
        logger.info("  PF-511: Volume Dry-Up  - 62.4% win rate")
        logger.info("-" * 60)
        logger.info("REMOVED (invalidated patterns):")
        logger.info("  Volume Spike: 9% win rate - FAILS")
        logger.info("  Whale Following: -22.8% return - LOSES MONEY")
        logger.info("=" * 60)

        await self.feed.connect()
        if not self.paper:
            await self.executor.connect()

        # Register callbacks
        self.feed.on_trade(self._on_trade)
        self.feed.on_create(self._on_create)

        # Start monitoring loop
        monitor_task = asyncio.create_task(self._monitor_positions())

        # Calculate end time
        end_time = None
        if duration_sec > 0:
            end_time = datetime.now() + timedelta(seconds=duration_sec)
            logger.info(f"Running for {duration_sec} seconds...")
        else:
            logger.info("Running indefinitely (Ctrl+C to stop)...")

        try:
            # Main event loop
            async for event in self.feed.stream():
                if not self._running:
                    break
                if end_time and datetime.now() > end_time:
                    break
                # Events are processed via callbacks

        except KeyboardInterrupt:
            logger.info("\nShutting down...")
        finally:
            self._running = False
            monitor_task.cancel()
            await self._shutdown()

    def _on_trade(self, trade: PumpTrade):
        """Process incoming trade from LaserStream"""
        self.trades_seen += 1

        # Get or create token state
        if trade.mint not in self.tokens:
            self.tokens[trade.mint] = TokenState(
                mint=trade.mint,
                created_at=trade.timestamp
            )

        state = self.tokens[trade.mint]
        state.update(trade)

        # Update existing position
        if trade.mint in self.positions:
            self.positions[trade.mint].current_price = state.price
            self.positions[trade.mint].peak_price = max(
                self.positions[trade.mint].peak_price,
                state.price
            )

        # Check for entry signal
        if trade.mint not in self.positions:
            self._check_entry(state, trade)

    def _on_create(self, create: TokenCreate):
        """Process new token creation"""
        logger.info(f"NEW TOKEN: {create.symbol} - {create.name} ({create.mint[:8]}...)")

        # Initialize token state
        self.tokens[create.mint] = TokenState(
            mint=create.mint,
            created_at=create.timestamp
        )

    def _check_entry(self, state: TokenState, trade: PumpTrade):
        """Check if we should enter a position"""
        # Skip if at max positions
        if len(self.positions) >= self.max_positions:
            return

        # Skip if too early (need some history)
        if len(state.trades) < 5:
            return

        # Calculate win probability
        win_prob, signals = self._calculate_signals(state, trade)

        if win_prob >= self.min_win_prob:
            self.signals_generated += 1
            asyncio.create_task(self._enter_position(state, win_prob, signals))

    def _calculate_signals(
        self,
        state: TokenState,
        trade: PumpTrade
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate entry signals using VALIDATED formulas from historical data.

        Backtested on 269,830 trades (Dec 2024).

        VALIDATED SIGNALS:
        - PF-520: Mean Reversion (82.5% win rate) - PRIMARY
        - PF-530: Buy Pressure (52.8% win rate) - SECONDARY

        REMOVED (INVALIDATED):
        - Volume Spike: 9% win rate - FAILS
        - Whale Following: 15.2% win rate, -22.8% return - LOSES MONEY

        Returns: (win_probability, signal_breakdown)
        """
        signals = {}

        # Convert to PricePoint format for validated formulas
        points = state.to_price_points()

        # ====================================================================
        # VALIDATED ENTRY SIGNALS (from 269,830 trade backtest)
        # ====================================================================

        # PF-520: Mean Reversion (82.5% win rate) - BEST PERFORMER
        # Logic: Buy after 30%+ drop in 5 trades (oversold bounce)
        mr_enter, mr_conf, mr_breakdown = pf520_mean_reversion(points)
        signals['pf520_mean_reversion'] = mr_conf if mr_enter else 0.0
        signals['pf520_fired'] = 1.0 if mr_enter else 0.0

        # PF-530: Buy Pressure (52.8% win rate)
        # Logic: Buy when >70% of recent trades are buys (momentum)
        bp_enter, bp_conf, bp_breakdown = pf530_buy_pressure(points)
        signals['pf530_buy_pressure'] = bp_conf if bp_enter else 0.0
        signals['pf530_fired'] = 1.0 if bp_enter else 0.0

        # ====================================================================
        # SUPPORTING SIGNALS (logical, not backtested)
        # ====================================================================

        # Curve position - Sweet spot is 10-50% (liquidity + room to grow)
        curve_signal = 0.0
        progress = state.curve_progress
        if 10 <= progress <= 50:
            curve_signal = 0.6  # Optimal zone
        elif 50 < progress <= 70:
            curve_signal = 0.4  # Still ok
        elif progress < 10:
            curve_signal = 0.2  # Too early, risky
        # >70% = too late, near graduation
        signals['curve_position'] = curve_signal

        # Confirming signal - Is this trade a buy?
        signals['is_buy'] = 0.5 if trade.is_buy else 0.0

        # ====================================================================
        # WIN PROBABILITY CALCULATION
        # ====================================================================

        # If PF-520 (mean reversion) fires, it's our strongest signal
        # 82.5% historical win rate - this alone justifies entry
        if mr_enter:
            # Mean reversion fired - high confidence entry
            # Base: 55% (conservative estimate from 82.5% backtest)
            # Boosted by buy pressure if also present
            base_prob = 0.55
            if bp_enter:
                base_prob = 0.58  # Both signals = higher confidence
            win_prob = base_prob + (curve_signal * 0.05)  # Small curve boost
            signals['entry_reason'] = 'PF-520_MEAN_REVERSION'

        elif bp_enter:
            # Buy pressure only - still valid but lower confidence
            # 52.8% historical win rate
            base_prob = 0.52
            win_prob = base_prob + (curve_signal * 0.03)
            signals['entry_reason'] = 'PF-530_BUY_PRESSURE'

        else:
            # No validated signal fired
            # Use supporting signals only (lower probability)
            win_prob = 0.45 + (curve_signal * 0.03) + (signals['is_buy'] * 0.02)
            signals['entry_reason'] = 'NO_VALIDATED_SIGNAL'

        # Ensure bounds
        win_prob = max(0.40, min(0.65, win_prob))
        signals['win_probability'] = win_prob

        return win_prob, signals

    async def _enter_position(
        self,
        state: TokenState,
        win_prob: float,
        signals: Dict[str, float]
    ):
        """Enter a new position"""
        # Calculate position size
        size_sol = self.capital * self.position_size_pct
        size_sol = min(size_sol, self.capital * 0.2)  # Never more than 20%

        # Determine tip based on conviction
        if win_prob > 0.60:
            tip = TipLevel.HIGH
        elif win_prob > 0.55:
            tip = TipLevel.MEDIUM
        else:
            tip = TipLevel.LOW

        # Log entry with formula IDs
        entry_reason = signals.get('entry_reason', 'UNKNOWN')
        pf520_fired = signals.get('pf520_fired', 0) > 0
        pf530_fired = signals.get('pf530_fired', 0) > 0

        logger.info(f"ENTRY SIGNAL: {state.mint[:8]}... | Win Prob: {win_prob*100:.1f}%")
        logger.info(f"  Reason: {entry_reason}")
        logger.info(f"  PF-520 Mean Reversion (82.5%): {'FIRED' if pf520_fired else 'no'}")
        logger.info(f"  PF-530 Buy Pressure (52.8%):   {'FIRED' if pf530_fired else 'no'}")
        logger.info(f"  Curve: {state.curve_progress:.1f}% | Size: {size_sol:.4f} SOL | Tip: {tip.name}")

        if self.paper:
            # Paper trade - simulate execution
            result = ExecutionResult(
                success=True,
                signature="paper",
                slot=0,
                error=None,
                latency_ms=50,
                tip_paid=tip.value
            )
        else:
            # Live trade via Jito
            result = await self.executor.buy(
                mint=state.mint,
                sol_amount=size_sol,
                tip=tip
            )

        if result.success:
            # Calculate tokens received (estimate from curve)
            tokens = size_sol / state.price if state.price > 0 else 0

            self.positions[state.mint] = Position(
                mint=state.mint,
                entry_price=state.price,
                entry_time=datetime.now(),
                token_amount=tokens,
                sol_invested=size_sol
            )

            self.capital -= size_sol

            logger.info(f"  ENTERED @ {state.price:.10f} | Latency: {result.latency_ms:.0f}ms")
        else:
            logger.error(f"  ENTRY FAILED: {result.error}")

    async def _check_exit(self, position: Position) -> Optional[str]:
        """
        Check if we should exit a position.

        Uses VALIDATED exit signals from historical data:
        - PF-511: Volume Dry-Up Warning (62.4% win rate)

        Returns: Exit reason or None
        """
        pnl = position.pnl_pct / 100

        # ====================================================================
        # STANDARD EXITS (always check first)
        # ====================================================================

        # Take profit
        if pnl >= self.target_pct:
            return "TARGET"

        # Stop loss
        if pnl <= -self.stop_pct:
            return "STOP"

        # Time limit
        if position.hold_time_sec >= self.max_hold_sec:
            return "TIMEOUT"

        # ====================================================================
        # VALIDATED EXIT SIGNALS (from 269,830 trade backtest)
        # ====================================================================

        state = self.tokens.get(position.mint)
        if not state:
            return None

        # Convert to PricePoint format for validated formulas
        points = state.to_price_points()

        # PF-511: Volume Dry-Up Warning (62.4% win rate)
        # Logic: Exit when volume drops >80% from average
        # This catches dying tokens before they crash
        if len(points) >= 11:  # Need enough data for lookback
            should_exit, confidence, breakdown = pf511_volume_dryup_warning(points)
            if should_exit:
                return "PF511_VOLUME_DRYUP"

        # ====================================================================
        # SUPPORTING EXIT SIGNALS (logical, kept from original)
        # ====================================================================

        # Momentum reversal - strong selling pressure
        if state.sol_flow_1m < -2.0:
            return "MOMENTUM_FLIP"

        # Near graduation (80%+) - price behavior changes
        if state.curve_progress > 80:
            return "GRADUATION_RISK"

        return None

    async def _exit_position(self, position: Position, reason: str):
        """Exit a position"""
        # Determine tip (exits are important!)
        if reason in ["STOP", "MOMENTUM_FLIP"]:
            tip = TipLevel.URGENT  # Get out fast
        else:
            tip = TipLevel.HIGH

        logger.info(f"EXIT: {position.mint[:8]}... | Reason: {reason}")

        if self.paper:
            result = ExecutionResult(
                success=True,
                signature="paper",
                slot=0,
                error=None,
                latency_ms=50,
                tip_paid=tip.value
            )
        else:
            result = await self.executor.sell(
                mint=position.mint,
                token_amount=position.token_amount,
                tip=tip
            )

        if result.success:
            # Calculate final P&L
            exit_value = position.sol_invested * (1 + position.pnl_pct / 100)
            pnl_sol = exit_value - position.sol_invested

            self.capital += exit_value

            # Record trade
            trade_result = TradeResult(
                mint=position.mint,
                is_win=position.pnl_pct > 0,
                pnl_pct=position.pnl_pct,
                pnl_sol=pnl_sol,
                hold_time_sec=position.hold_time_sec,
                entry_time=position.entry_time,
                exit_time=datetime.now(),
                exit_reason=reason
            )
            self.completed_trades.append(trade_result)

            # Remove position
            del self.positions[position.mint]

            status = "WIN" if trade_result.is_win else "LOSS"
            logger.info(f"  {status}: {position.pnl_pct:+.2f}% ({pnl_sol:+.4f} SOL) | Hold: {position.hold_time_sec:.0f}s")
        else:
            logger.error(f"  EXIT FAILED: {result.error}")

    async def _monitor_positions(self):
        """Background task to monitor and exit positions"""
        while self._running:
            try:
                # Check each position
                for mint in list(self.positions.keys()):
                    position = self.positions.get(mint)
                    if not position:
                        continue

                    exit_reason = await self._check_exit(position)
                    if exit_reason:
                        await self._exit_position(position, exit_reason)

                await asyncio.sleep(0.1)  # Check 10x per second

            except Exception as e:
                logger.error(f"Monitor error: {e}")
                await asyncio.sleep(1)

    async def _shutdown(self):
        """Clean shutdown"""
        # Close all positions
        for mint in list(self.positions.keys()):
            position = self.positions[mint]
            await self._exit_position(position, "SHUTDOWN")

        # Print final stats
        self._print_stats()

        # Close connections
        await self.feed.close()
        if not self.paper:
            await self.executor.close()

    def _print_stats(self):
        """Print session statistics"""
        logger.info("\n" + "=" * 60)
        logger.info("SESSION COMPLETE")
        logger.info("=" * 60)

        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            logger.info(f"Duration: {duration:.0f} seconds")

        logger.info(f"Trades Seen: {self.trades_seen}")
        logger.info(f"Signals Generated: {self.signals_generated}")
        logger.info(f"Trades Executed: {len(self.completed_trades)}")

        if self.completed_trades:
            wins = sum(1 for t in self.completed_trades if t.is_win)
            total = len(self.completed_trades)
            win_rate = wins / total * 100

            total_pnl = sum(t.pnl_sol for t in self.completed_trades)
            avg_pnl = sum(t.pnl_pct for t in self.completed_trades) / total
            avg_hold = sum(t.hold_time_sec for t in self.completed_trades) / total

            logger.info(f"\nWin Rate: {win_rate:.1f}% ({wins}/{total})")
            logger.info(f"Total P&L: {total_pnl:+.4f} SOL")
            logger.info(f"Avg P&L: {avg_pnl:+.2f}%")
            logger.info(f"Avg Hold: {avg_hold:.0f}s")

            # Exit reasons
            logger.info("\nExit Reasons:")
            reasons = {}
            for t in self.completed_trades:
                reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
            for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                logger.info(f"  {reason}: {count}")

        pnl_pct = (self.capital - self.initial_capital) / self.initial_capital * 100
        logger.info(f"\nFinal Capital: ${self.capital:.2f} ({pnl_pct:+.2f}%)")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="RenTech V3 Trading Engine")
    parser.add_argument("--capital", type=float, default=100, help="Starting capital in USD")
    parser.add_argument("--duration", type=float, default=3600, help="Run duration in seconds (0=forever)")
    parser.add_argument("--target", type=float, default=0.05, help="Take profit percentage")
    parser.add_argument("--stop", type=float, default=0.05, help="Stop loss percentage")
    parser.add_argument("--min-prob", type=float, default=0.52, help="Minimum win probability")
    parser.add_argument("--max-positions", type=int, default=3, help="Maximum concurrent positions")
    parser.add_argument("--paper", action="store_true", help="Paper trading mode")
    parser.add_argument("--live", action="store_true", help="Live trading mode")

    args = parser.parse_args()

    # Paper mode unless explicitly live
    paper = not args.live

    engine = RenTechV3(
        capital=args.capital,
        target_pct=args.target,
        stop_pct=args.stop,
        min_win_prob=args.min_prob,
        max_positions=args.max_positions,
        paper=paper,
        helius_api_key=os.getenv("HELIUS_API_KEY"),
        keypair_path=os.getenv("KEYPAIR_PATH")
    )

    asyncio.run(engine.start(duration_sec=args.duration))


if __name__ == "__main__":
    main()
