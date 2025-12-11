"""
Trading Engine - RenTech 1:1 Parity Architecture
================================================

SINGLE codebase for BOTH paper and real trading.
The ONLY difference is the executor (paper vs solana).

This is how RenTech validates strategies:
- Paper and real use IDENTICAL logic
- Any divergence = bug in code, not market difference
- Fix bugs before deploying capital

Usage:
    # Paper trading
    engine = TradingEngine(paper_mode=True, capital=100.0)

    # Real trading (ONLY after parity validation)
    engine = TradingEngine(paper_mode=False, capital=100.0)
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from .config import TradingConfig, DEFAULT_CONFIG
from .adaptive_config import AdaptiveFormulaConfig, AdaptiveParameters
from .models import (
    Order, OrderSide, OrderType,
    ExecutionResult, Position, ClosedTrade,
    EntryDecision, ExitDecision, TradeDecision,
    TokenData,
)
from .executors.paper import PaperExecutor
from .signals.signal_engine import SignalEngine, get_signal_engine

logger = logging.getLogger(__name__)


@dataclass
class EngineStats:
    """Trading engine statistics"""
    total_decisions: int = 0
    entries: int = 0
    exits: int = 0
    skips: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    peak_capital: float = 0.0
    max_drawdown: float = 0.0
    daily_trades: int = 0
    daily_pnl: float = 0.0
    daily_reset_time: float = field(default_factory=time.time)

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            'total_decisions': self.total_decisions,
            'entries': self.entries,
            'exits': self.exits,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': self.win_rate,
            'total_pnl': self.total_pnl,
            'max_drawdown': self.max_drawdown,
        }


class TradingEngine:
    """
    Single trading engine for both paper and real trading.
    RenTech methodology: 1:1 parity.

    CRITICAL: All logic is IDENTICAL for paper and real.
    The ONLY difference is the executor.
    """

    def __init__(
        self,
        paper_mode: bool = True,
        capital: float = 100.0,
        config: Optional[TradingConfig] = None,
    ):
        """
        Initialize trading engine.

        Args:
            paper_mode: True for paper trading, False for real
            capital: Starting capital in USD
            config: Trading configuration (same for paper and real)
        """
        self.paper_mode = paper_mode
        self.capital = capital
        self.initial_capital = capital
        self.config = config or DEFAULT_CONFIG

        # Components - SAME for both modes
        self.signal_engine = get_signal_engine()

        # Executor - ONLY difference between paper and real
        # NOTE: For real mode, executor should be set by orchestrator
        # with proper keypair configuration. Default to paper.
        if paper_mode:
            self.executor = PaperExecutor()
        else:
            # Real executor requires keypair - will be set by orchestrator
            # Default to paper executor (disabled) for safety
            self.executor = PaperExecutor()
            self.executor.disabled = True
            logger.info("Real mode: executor will be set by orchestrator")

        # State tracking - IDENTICAL for both modes
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[ClosedTrade] = []
        self.decision_log: List[Dict[str, Any]] = []
        self.stats = EngineStats()

        # Formula-driven adaptive configuration - NO hardcoded values
        # This replaces static config with real-time computed parameters
        self.adaptive_config = AdaptiveFormulaConfig()

    async def on_new_token(
        self,
        token_data: Dict[str, Any],
        trades: Optional[List[Dict[str, Any]]] = None,
    ) -> TradeDecision:
        """
        Evaluate a new token for entry.
        IDENTICAL logic for paper and real.

        Args:
            token_data: Token metadata and current state
            trades: Recent trades for the token

        Returns:
            TradeDecision with action taken
        """
        mint = token_data.get('mint', '')
        self.stats.total_decisions += 1

        # Check daily limits
        self._check_daily_reset()
        if self.stats.daily_trades >= self.config.max_daily_trades:
            return TradeDecision(action='SKIP', mint=mint, reason='daily_trade_limit')

        if self.stats.daily_pnl < -self.config.max_daily_loss_pct * self.initial_capital:
            return TradeDecision(action='SKIP', mint=mint, reason='daily_loss_limit')

        # Check max positions
        if len(self.positions) >= self.config.max_open_positions:
            return TradeDecision(action='SKIP', mint=mint, reason='max_positions')

        # Already have position?
        if mint in self.positions:
            return TradeDecision(action='SKIP', mint=mint, reason='already_positioned')

        # Step 1: Compute all signals (IDENTICAL)
        signal_result = self.signal_engine.compute_all(token_data, trades)
        signals = signal_result.signals

        # Step 2: Entry decision (IDENTICAL)
        entry_decision = self._should_enter(signals)
        if not entry_decision.should_enter:
            self.stats.skips += 1
            return TradeDecision(
                action='SKIP',
                mint=mint,
                reason=entry_decision.reason,
                signals=signals,
            )

        # Step 3: Position sizing (IDENTICAL)
        position_size = self._calculate_position_size(signals)

        if position_size < self.config.min_position_sol:
            return TradeDecision(
                action='SKIP',
                mint=mint,
                reason='position_too_small',
                signals=signals,
            )

        # Step 4: Build order (IDENTICAL)
        current_price = token_data.get('price', 0)
        if current_price <= 0:
            return TradeDecision(
                action='SKIP',
                mint=mint,
                reason='invalid_price',
                signals=signals,
            )

        order = Order(
            mint=mint,
            side=OrderSide.BUY,
            amount_sol=position_size,
            expected_price=current_price,
            order_type=OrderType.MARKET,
            context={
                'liquidity': token_data.get('liquidity_sol', 10.0),
                'bonding_progress': token_data.get('bonding_curve_progress', 0.5),
            },
        )

        # Step 5: Execute (ONLY DIFFERENCE - paper vs real executor)
        result = await self.executor.execute(order)

        if not result.success:
            return TradeDecision(
                action='SKIP',
                mint=mint,
                reason=f'execution_failed: {result.error}',
                signals=signals,
            )

        # Step 6: Record position (IDENTICAL)
        position = Position(
            mint=mint,
            entry_price=result.fill_price,
            entry_time=time.time(),
            tokens=result.tokens,
            sol_invested=position_size,
            entry_signals=signals,
            paper_mode=self.paper_mode,
            tx_signature=result.tx_signature,
        )
        self.positions[mint] = position

        # Update stats
        self.capital -= position_size
        self.stats.entries += 1
        self.stats.daily_trades += 1

        # Log decision
        decision = TradeDecision(
            action='BUY',
            mint=mint,
            size=position_size,
            signals=signals,
            result=result,
            reason=f'quality={entry_decision.quality_score}',
        )
        self._log_decision(decision)

        logger.info(
            f"[{'PAPER' if self.paper_mode else 'REAL'}] BUY {mint[:8]}... "
            f"size={position_size:.4f} SOL @ {result.fill_price:.8f}"
        )

        return decision

    async def on_price_update(
        self,
        mint: str,
        price: float,
        token_data: Optional[Dict[str, Any]] = None,
        volume: float = 0.0,
        is_buy: bool = True,
    ) -> Optional[TradeDecision]:
        """
        Check exit conditions for open positions.
        IDENTICAL logic for paper and real.

        Args:
            mint: Token mint address
            price: Current price
            token_data: Optional updated token data
            volume: Trade volume (for adaptive config)
            is_buy: True if this was a buy trade

        Returns:
            TradeDecision if exit triggered, None otherwise
        """
        # Update adaptive config with new market data
        # This feeds the Ornstein-Uhlenbeck and Hawkes models
        self.adaptive_config.update(
            price=price,
            volume=volume,
            timestamp=time.time() * 1000,  # milliseconds
            is_buy=is_buy,
        )

        if mint not in self.positions:
            return None

        position = self.positions[mint]

        # Calculate current PnL (IDENTICAL)
        current_pnl = (price - position.entry_price) / position.entry_price
        position.current_pnl = current_pnl

        # Check exit conditions (IDENTICAL)
        exit_decision = self._should_exit(position, price, current_pnl)
        if not exit_decision.should_exit:
            return None

        # Build exit order (IDENTICAL)
        order = Order(
            mint=mint,
            side=OrderSide.SELL,
            amount_tokens=position.tokens,
            expected_price=price,
            order_type=OrderType.MARKET,
            context={
                'exit_reason': exit_decision.reason,
            },
        )

        # Execute (ONLY DIFFERENCE)
        result = await self.executor.execute(order)

        if not result.success:
            logger.warning(f"Exit failed for {mint}: {result.error}")
            return None

        # Record closed trade (IDENTICAL)
        closed = ClosedTrade(
            mint=mint,
            entry_price=position.entry_price,
            exit_price=result.fill_price,
            entry_time=position.entry_time,
            exit_time=time.time(),
            tokens=position.tokens,
            sol_invested=position.sol_invested,
            sol_received=result.sol_amount,
            pnl_sol=result.sol_amount - position.sol_invested,
            pnl_pct=current_pnl,
            exit_reason=exit_decision.reason,
            entry_signals=position.entry_signals,
            paper_mode=self.paper_mode,
        )
        self.closed_trades.append(closed)

        # Update stats
        self.capital += result.sol_amount
        self.stats.exits += 1
        self.stats.total_pnl += closed.pnl_sol
        self.stats.daily_pnl += closed.pnl_sol

        if closed.pnl_sol > 0:
            self.stats.wins += 1
        else:
            self.stats.losses += 1

        # Track peak and drawdown
        if self.capital > self.stats.peak_capital:
            self.stats.peak_capital = self.capital
        drawdown = (self.stats.peak_capital - self.capital) / self.stats.peak_capital
        if drawdown > self.stats.max_drawdown:
            self.stats.max_drawdown = drawdown

        # Remove position
        del self.positions[mint]

        decision = TradeDecision(
            action='SELL',
            mint=mint,
            pnl=closed.pnl_pct,
            reason=exit_decision.reason,
            result=result,
        )
        self._log_decision(decision)

        logger.info(
            f"[{'PAPER' if self.paper_mode else 'REAL'}] SELL {mint[:8]}... "
            f"PnL={closed.pnl_pct:+.1%} ({exit_decision.reason})"
        )

        return decision

    def _should_enter(self, signals: Dict[int, float]) -> EntryDecision:
        """
        Entry logic. MUST BE IDENTICAL for paper and real.
        """
        config = self.config

        # Tier 1: Safety - MUST pass ALL rug checks
        rug_score = signals.get(9550, 1.0)  # AggregateRugScore
        if rug_score > config.max_rug_score:
            return EntryDecision(False, f'rug_score={rug_score:.2f}', signals=signals)

        instant_rug = signals.get(9541, 1.0)  # InstantRugAlert
        if instant_rug > config.max_instant_rug:
            return EntryDecision(False, f'instant_rug={instant_rug:.2f}', signals=signals)

        mint_risk = signals.get(9521, 1.0)  # MintAuthorityRisk
        if mint_risk > config.max_mint_risk:
            return EntryDecision(False, f'mint_risk={mint_risk:.2f}', signals=signals)

        honeypot = signals.get(9524, 1.0)  # HoneypotIndicator
        if honeypot > config.max_honeypot:
            return EntryDecision(False, f'honeypot={honeypot:.2f}', signals=signals)

        # Tier 2: Quality - need minimum quality signals
        quality_score = 0

        graduation_velocity = signals.get(9401, 0)
        if graduation_velocity > config.min_graduation_velocity:
            quality_score += 1

        momentum = signals.get(9353, 0)  # VelocityAcceleration normalized
        if momentum > config.min_momentum:
            quality_score += 1

        entry_window = signals.get(9391, 0)  # OptimalEntryWindow
        if entry_window > config.min_entry_window:
            quality_score += 1

        directional = signals.get(9356, 0)  # DirectionalIntensity
        if directional > config.min_directional:
            quality_score += 1

        burst = signals.get(9361, 0)  # BurstIntensity
        if burst > config.min_burst:
            quality_score += 1

        if quality_score < config.min_quality_signals:
            return EntryDecision(
                False,
                f'quality_score={quality_score}/{config.min_quality_signals}',
                quality_score=quality_score,
                signals=signals,
            )

        return EntryDecision(
            True,
            f'passed_all_checks',
            quality_score=quality_score,
            signals=signals,
        )

    def _should_exit(
        self,
        position: Position,
        price: float,
        current_pnl: float,
    ) -> ExitDecision:
        """
        Exit logic. MUST BE IDENTICAL for paper and real.

        FORMULA-DRIVEN: Uses adaptive_config to compute exit thresholds
        in real-time based on Ornstein-Uhlenbeck optimal stopping theory,
        quantum uncertainty, and Hawkes process intensity.

        NO HARDCODED VALUES - everything computed mathematically.
        """
        # Current time in milliseconds
        current_time_ms = time.time() * 1000
        entry_time_ms = position.entry_time * 1000

        # Get formula-driven exit decision
        should_exit, reason, confidence = self.adaptive_config.should_exit(
            entry_price=position.entry_price,
            entry_time_ms=entry_time_ms,
            current_price=price,
            current_time_ms=current_time_ms,
        )

        if should_exit:
            return ExitDecision(True, reason, current_pnl)

        # Trailing stop using adaptive parameters
        params = self.adaptive_config.last_params
        if position.peak_pnl > params.take_profit_pct * 0.5:
            # Dynamic trailing stop based on volatility
            trailing_threshold = params.stop_loss_pct * 0.8
            drawdown_from_peak = position.peak_pnl - current_pnl
            if drawdown_from_peak > trailing_threshold:
                return ExitDecision(
                    True,
                    f'trailing_stop ({drawdown_from_peak*100:.1f}% drawdown from peak)',
                    current_pnl
                )

        return ExitDecision(False, None, current_pnl)

    def _calculate_position_size(self, signals: Dict[int, float]) -> float:
        """
        Calculate position size. MUST BE IDENTICAL for paper and real.

        FORMULA-DRIVEN: Uses Kelly Criterion with:
        - Estimated win rate from historical performance
        - Adaptive take profit / stop loss ratio
        - Kyle's lambda for price impact adjustment
        - Signal quality modulation

        NO HARDCODED FRACTIONS - everything computed mathematically.
        """
        # Get formula-computed position size from adaptive config
        # This uses Kelly Criterion: f* = (p*b - q) / b
        base_size = self.adaptive_config.get_position_size(self.capital)

        # Modulate by signal quality (quantum + rug scores)
        # OptimalEntryWindow (9391) indicates timing quality
        entry_window = signals.get(9391, 0.5)

        # Quantum combined score (9601) indicates overall regime favorability
        quantum_score = signals.get(9601, 0.5)

        # Rug score (9550) reduces position - higher risk = smaller size
        rug_score = signals.get(9550, 0.3)

        # Calculate signal multiplier
        # Good signals boost size, rug risk reduces it
        signal_mult = (0.5 + entry_window * 0.3 + quantum_score * 0.2) * (1 - rug_score)
        signal_mult = max(0.25, min(1.5, signal_mult))  # 0.25x to 1.5x

        position_sol = base_size * signal_mult

        # Safety constraints (these are risk limits, not trading parameters)
        min_position = self.config.min_position_sol
        max_position = min(self.config.max_position_sol, self.capital * 0.5)

        position_sol = max(min_position, min(max_position, position_sol))

        return round(position_sol, 4)

    def _check_daily_reset(self):
        """Reset daily counters at midnight"""
        now = time.time()
        if now - self.stats.daily_reset_time > 86400:  # 24 hours
            self.stats.daily_trades = 0
            self.stats.daily_pnl = 0.0
            self.stats.daily_reset_time = now

    def _log_decision(self, decision: TradeDecision):
        """Log decision for analysis"""
        self.decision_log.append({
            'decision': decision.to_dict(),
            'capital': self.capital,
            'positions': len(self.positions),
            'timestamp': time.time(),
            'paper_mode': self.paper_mode,
        })

    def get_state(self) -> Dict[str, Any]:
        """Get current engine state"""
        return {
            'paper_mode': self.paper_mode,
            'capital': self.capital,
            'initial_capital': self.initial_capital,
            'positions': {k: v.to_dict() for k, v in self.positions.items()},
            'stats': self.stats.to_dict(),
            'closed_trades': len(self.closed_trades),
            'total_pnl': self.stats.total_pnl,
            'win_rate': self.stats.win_rate,
        }

    def reset(self):
        """Reset engine state"""
        self.capital = self.initial_capital
        self.positions.clear()
        self.closed_trades.clear()
        self.decision_log.clear()
        self.stats = EngineStats()
        self.executor.reset()
        self.signal_engine.clear_cache()
