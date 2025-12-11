#!/usr/bin/env python3
"""
RENTECH PORTFOLIO OPTIMIZER
============================

Kelly criterion position sizing + signal correlation-based portfolio optimization.

Based on:
- Kelly Criterion (1956) - Optimal betting fraction
- Markowitz Portfolio Theory - Mean-variance optimization
- Black-Litterman Model - Signal-based views
- Risk Parity - Equal risk contribution

Features:
1. Kelly criterion with fractional sizing
2. Signal correlation matrix for diversification
3. Dynamic position limits based on volatility
4. Drawdown-adjusted position sizing
5. Regime-aware portfolio weights
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import json
from pathlib import Path
from collections import defaultdict
import time


# ============================================================
# CONFIGURATION
# ============================================================

# Kelly Criterion Settings
FULL_KELLY_FRACTION = 0.25  # Use 1/4 Kelly for safety
MIN_TRADES_FOR_KELLY = 30   # Minimum sample size
MAX_POSITION_FRACTION = 0.10  # Max 10% of capital per trade

# Correlation Settings
CORRELATION_LOOKBACK = 100   # Trades to calculate correlation
DIVERSIFICATION_BONUS = 0.1  # Bonus for uncorrelated signals

# Risk Settings
MAX_PORTFOLIO_RISK = 0.20    # Max 20% VaR at 95%
MAX_CONCENTRATION = 0.30     # Max 30% in any single token
MAX_CORRELATED_EXPOSURE = 0.50  # Max 50% in correlated positions

# Regime Settings
REGIME_LOOKBACK = 50         # Trades to detect regime


# ============================================================
# DATA STRUCTURES
# ============================================================

class MarketRegime(Enum):
    """Market regime classification"""
    BULL = "bull"          # High activity, positive momentum
    BEAR = "bear"          # Low activity, negative momentum
    SIDEWAYS = "sideways"  # Normal activity
    VOLATILE = "volatile"  # High volatility spikes


@dataclass
class SignalStats:
    """Statistics for a trading signal"""
    name: str
    win_rate: float
    avg_win: float      # In SOL or % terms
    avg_loss: float
    sharpe: float
    trade_count: int
    kelly_fraction: float = 0.0
    correlation_with_portfolio: float = 0.0


@dataclass
class PositionSizing:
    """Position sizing recommendation"""
    signal_name: str
    base_size_sol: float
    kelly_adjusted: float
    volatility_adjusted: float
    regime_adjusted: float
    final_size_sol: float
    confidence: float
    rationale: str


@dataclass
class PortfolioState:
    """Current portfolio state"""
    cash_sol: float
    positions: Dict[str, float]  # token -> size in SOL
    total_value_sol: float
    current_drawdown: float
    max_drawdown: float
    regime: MarketRegime
    risk_utilization: float  # 0-1 scale


@dataclass
class OptimalWeights:
    """Optimal signal weights for portfolio"""
    signal_weights: Dict[str, float]  # signal -> weight (0-1)
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    diversification_ratio: float


# ============================================================
# KELLY CRITERION CALCULATOR
# ============================================================

class KellyCriterion:
    """
    Kelly Criterion position sizing.

    Full Kelly: f* = (p * b - q) / b
    Where:
    - f* = fraction of capital to bet
    - p = probability of winning
    - b = odds (avg_win / avg_loss)
    - q = probability of losing = 1 - p

    We use fractional Kelly (1/4) for safety.
    """

    def __init__(self, fraction: float = FULL_KELLY_FRACTION):
        self.fraction = fraction

    def calculate(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        trade_count: int = 0
    ) -> float:
        """
        Calculate Kelly fraction.

        Returns value between 0 and max_position_fraction.
        """
        # Need minimum sample size
        if trade_count < MIN_TRADES_FOR_KELLY:
            return 0.02  # Conservative 2% default

        # Validate inputs
        if win_rate <= 0 or win_rate >= 1:
            return 0.02
        if avg_loss <= 0:
            return 0.02

        p = win_rate
        q = 1 - p
        b = avg_win / avg_loss  # Odds

        # Kelly formula
        kelly = (p * b - q) / b

        # Apply fractional Kelly
        kelly *= self.fraction

        # Confidence adjustment based on sample size
        # More trades = more confidence in estimate
        confidence = min(1.0, trade_count / 200)
        kelly *= confidence

        # Clamp to valid range
        kelly = max(0.01, min(MAX_POSITION_FRACTION, kelly))

        return kelly

    def calculate_from_returns(
        self,
        returns: List[float],
        risk_free_rate: float = 0.0
    ) -> float:
        """
        Calculate Kelly from historical returns.

        Uses the simpler formula: f* = mu / sigma^2
        Where mu = mean excess return, sigma = std dev
        """
        if len(returns) < MIN_TRADES_FOR_KELLY:
            return 0.02

        returns_arr = np.array(returns)
        mu = np.mean(returns_arr) - risk_free_rate
        sigma = np.std(returns_arr)

        if sigma <= 0:
            return 0.02

        # Kelly fraction
        kelly = mu / (sigma ** 2)

        # Apply fractional Kelly
        kelly *= self.fraction

        # Clamp
        kelly = max(0.01, min(MAX_POSITION_FRACTION, kelly))

        return kelly


# ============================================================
# SIGNAL CORRELATION ANALYZER
# ============================================================

class SignalCorrelation:
    """
    Analyzes correlation between trading signals.

    Low correlation = good diversification.
    """

    def __init__(self, lookback: int = CORRELATION_LOOKBACK):
        self.lookback = lookback
        self.signal_returns: Dict[str, List[float]] = defaultdict(list)

    def add_signal_return(self, signal_name: str, return_pct: float):
        """Add a return for a signal"""
        self.signal_returns[signal_name].append(return_pct)
        # Trim to lookback
        if len(self.signal_returns[signal_name]) > self.lookback:
            self.signal_returns[signal_name] = \
                self.signal_returns[signal_name][-self.lookback:]

    def correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate pairwise correlation matrix.

        Returns dict of dict with correlation values.
        """
        signals = list(self.signal_returns.keys())
        matrix = {}

        for s1 in signals:
            matrix[s1] = {}
            r1 = np.array(self.signal_returns[s1])

            for s2 in signals:
                if s1 == s2:
                    matrix[s1][s2] = 1.0
                    continue

                r2 = np.array(self.signal_returns[s2])

                # Align lengths
                min_len = min(len(r1), len(r2))
                if min_len < 10:
                    matrix[s1][s2] = 0.0
                    continue

                r1_aligned = r1[-min_len:]
                r2_aligned = r2[-min_len:]

                # Pearson correlation
                corr = np.corrcoef(r1_aligned, r2_aligned)[0, 1]
                if np.isnan(corr):
                    corr = 0.0

                matrix[s1][s2] = corr

        return matrix

    def find_uncorrelated_signals(
        self,
        threshold: float = 0.3
    ) -> List[Tuple[str, str]]:
        """Find pairs of signals with low correlation"""
        matrix = self.correlation_matrix()
        pairs = []

        signals = list(matrix.keys())
        for i, s1 in enumerate(signals):
            for s2 in signals[i+1:]:
                corr = abs(matrix[s1].get(s2, 0))
                if corr < threshold:
                    pairs.append((s1, s2))

        return pairs

    def portfolio_correlation(self, active_signals: List[str]) -> float:
        """
        Calculate average correlation of active signals.

        Lower = more diversified.
        """
        if len(active_signals) < 2:
            return 0.0

        matrix = self.correlation_matrix()
        correlations = []

        for i, s1 in enumerate(active_signals):
            for s2 in active_signals[i+1:]:
                corr = matrix.get(s1, {}).get(s2, 0)
                correlations.append(abs(corr))

        return np.mean(correlations) if correlations else 0.0


# ============================================================
# REGIME DETECTOR
# ============================================================

class RegimeDetector:
    """
    Detects current market regime from recent trades.
    """

    def __init__(self, lookback: int = REGIME_LOOKBACK):
        self.lookback = lookback
        self.returns: List[float] = []
        self.volumes: List[float] = []

    def add_observation(self, return_pct: float, volume_sol: float):
        """Add market observation"""
        self.returns.append(return_pct)
        self.volumes.append(volume_sol)

        # Trim
        if len(self.returns) > self.lookback:
            self.returns = self.returns[-self.lookback:]
            self.volumes = self.volumes[-self.lookback:]

    def detect(self) -> MarketRegime:
        """Detect current regime"""
        if len(self.returns) < 20:
            return MarketRegime.SIDEWAYS

        returns = np.array(self.returns)
        volumes = np.array(self.volumes)

        # Calculate metrics
        avg_return = np.mean(returns)
        volatility = np.std(returns)
        avg_volume = np.mean(volumes)
        recent_volume = np.mean(volumes[-10:])

        # Volatility spike detection
        if volatility > 2 * np.mean([abs(r) for r in returns]):
            return MarketRegime.VOLATILE

        # Volume trend
        volume_increasing = recent_volume > avg_volume * 1.2

        # Return trend
        if avg_return > 0.02 and volume_increasing:  # 2% positive
            return MarketRegime.BULL
        elif avg_return < -0.02:  # 2% negative
            return MarketRegime.BEAR
        else:
            return MarketRegime.SIDEWAYS

    def regime_multiplier(self) -> float:
        """
        Position sizing multiplier based on regime.

        Bull: slightly increase
        Bear: decrease
        Sideways: neutral
        Volatile: decrease significantly
        """
        regime = self.detect()

        multipliers = {
            MarketRegime.BULL: 1.1,
            MarketRegime.BEAR: 0.7,
            MarketRegime.SIDEWAYS: 1.0,
            MarketRegime.VOLATILE: 0.5
        }

        return multipliers[regime]


# ============================================================
# PORTFOLIO OPTIMIZER
# ============================================================

class PortfolioOptimizer:
    """
    Main portfolio optimization engine.

    Combines Kelly criterion, correlation analysis, and regime detection
    for optimal position sizing.
    """

    def __init__(
        self,
        initial_capital: float = 10.0,  # SOL
        max_position_pct: float = MAX_POSITION_FRACTION,
        kelly_fraction: float = FULL_KELLY_FRACTION
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_pct = max_position_pct

        # Components
        self.kelly = KellyCriterion(fraction=kelly_fraction)
        self.correlation = SignalCorrelation()
        self.regime = RegimeDetector()

        # Signal statistics
        self.signal_stats: Dict[str, SignalStats] = {}

        # Portfolio state
        self.positions: Dict[str, float] = {}
        self.equity_curve: List[float] = [initial_capital]
        self.max_equity = initial_capital

    def update_signal_stats(
        self,
        signal_name: str,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        sharpe: float,
        trade_count: int
    ):
        """Update statistics for a signal"""
        kelly_f = self.kelly.calculate(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            trade_count=trade_count
        )

        self.signal_stats[signal_name] = SignalStats(
            name=signal_name,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            sharpe=sharpe,
            trade_count=trade_count,
            kelly_fraction=kelly_f
        )

    def add_trade_result(
        self,
        signal_name: str,
        return_pct: float,
        volume_sol: float
    ):
        """Record trade result for correlation and regime analysis"""
        self.correlation.add_signal_return(signal_name, return_pct)
        self.regime.add_observation(return_pct, volume_sol)

    def calculate_position_size(
        self,
        signal_name: str,
        signal_confidence: float = 1.0
    ) -> PositionSizing:
        """
        Calculate optimal position size for a signal.

        Combines:
        1. Kelly criterion base size
        2. Volatility adjustment
        3. Regime adjustment
        4. Correlation penalty
        5. Drawdown adjustment
        """
        stats = self.signal_stats.get(signal_name)

        if not stats:
            return PositionSizing(
                signal_name=signal_name,
                base_size_sol=0.0,
                kelly_adjusted=0.0,
                volatility_adjusted=0.0,
                regime_adjusted=0.0,
                final_size_sol=0.0,
                confidence=0.0,
                rationale="No stats available for signal"
            )

        # 1. Base Kelly size
        base_fraction = stats.kelly_fraction
        base_size = self.current_capital * base_fraction

        # 2. Signal confidence adjustment
        kelly_adjusted = base_size * signal_confidence

        # 3. Volatility adjustment (reduce in volatile regimes)
        regime_mult = self.regime.regime_multiplier()
        volatility_adjusted = kelly_adjusted * regime_mult

        # 4. Correlation penalty
        active_signals = list(self.positions.keys())
        if active_signals:
            avg_corr = self.correlation.portfolio_correlation(
                active_signals + [signal_name]
            )
            # High correlation = reduce size
            corr_penalty = 1.0 - (avg_corr * 0.3)
            volatility_adjusted *= max(0.5, corr_penalty)

        # 5. Drawdown adjustment
        current_dd = self._current_drawdown()
        if current_dd > 0.10:  # >10% drawdown
            dd_mult = max(0.5, 1.0 - current_dd)
            regime_adjusted = volatility_adjusted * dd_mult
        else:
            regime_adjusted = volatility_adjusted

        # 6. Apply absolute limits
        max_size = self.current_capital * self.max_position_pct
        final_size = min(regime_adjusted, max_size)

        # Confidence in this sizing
        confidence = min(1.0, stats.trade_count / 100) * signal_confidence

        # Build rationale
        rationale = (
            f"Kelly: {base_fraction:.1%}, "
            f"Regime: {self.regime.detect().value}, "
            f"Mult: {regime_mult:.1f}, "
            f"DD: {current_dd:.1%}"
        )

        return PositionSizing(
            signal_name=signal_name,
            base_size_sol=base_size,
            kelly_adjusted=kelly_adjusted,
            volatility_adjusted=volatility_adjusted,
            regime_adjusted=regime_adjusted,
            final_size_sol=final_size,
            confidence=confidence,
            rationale=rationale
        )

    def _current_drawdown(self) -> float:
        """Calculate current drawdown from peak"""
        if not self.equity_curve:
            return 0.0

        current = self.equity_curve[-1]
        peak = max(self.equity_curve)

        if peak <= 0:
            return 0.0

        return (peak - current) / peak

    def optimize_weights(
        self,
        target_signals: List[str]
    ) -> OptimalWeights:
        """
        Calculate optimal weights for a set of signals.

        Uses mean-variance optimization with correlation constraints.
        """
        n = len(target_signals)
        if n == 0:
            return OptimalWeights({}, 0.0, 0.0, 0.0, 0.0)

        # Get expected returns (using Sharpe as proxy)
        returns = []
        for sig in target_signals:
            stats = self.signal_stats.get(sig)
            if stats:
                returns.append(stats.sharpe)
            else:
                returns.append(0.0)

        returns = np.array(returns)

        # Get correlation matrix
        corr_matrix = self.correlation.correlation_matrix()

        # Build covariance matrix (approximation)
        cov = np.zeros((n, n))
        for i, s1 in enumerate(target_signals):
            for j, s2 in enumerate(target_signals):
                if i == j:
                    cov[i, j] = 1.0
                else:
                    cov[i, j] = corr_matrix.get(s1, {}).get(s2, 0.0)

        # Simple optimization: inverse volatility weighting with correlation
        # Weight inversely proportional to correlation with others
        weights = np.zeros(n)
        for i in range(n):
            avg_corr = np.mean(np.abs(cov[i, :]))
            kelly_f = self.signal_stats.get(
                target_signals[i], SignalStats("", 0, 0, 0, 0, 0)
            ).kelly_fraction
            weights[i] = kelly_f * (1.0 - avg_corr * 0.3)

        # Normalize to sum to 1 if needed
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)

        # Calculate portfolio metrics
        expected_return = np.dot(weights, returns)
        expected_risk = np.sqrt(np.dot(weights, np.dot(cov, weights)))
        sharpe = expected_return / expected_risk if expected_risk > 0 else 0

        # Diversification ratio
        weighted_avg_vol = np.sum(weights * np.diag(cov))
        div_ratio = weighted_avg_vol / expected_risk if expected_risk > 0 else 1

        return OptimalWeights(
            signal_weights=dict(zip(target_signals, weights)),
            expected_return=expected_return,
            expected_risk=expected_risk,
            sharpe_ratio=sharpe,
            diversification_ratio=div_ratio
        )

    def risk_budget(
        self,
        target_risk: float = MAX_PORTFOLIO_RISK
    ) -> Dict[str, float]:
        """
        Calculate risk budget for each signal.

        Allocates risk equally by default (risk parity approach).
        """
        signals = list(self.signal_stats.keys())
        n = len(signals)

        if n == 0:
            return {}

        # Equal risk contribution
        risk_per_signal = target_risk / n

        # Adjust by signal quality (Sharpe)
        sharpes = [self.signal_stats[s].sharpe for s in signals]
        total_sharpe = sum(sharpes) if any(sharpes) else 1.0

        budget = {}
        for sig in signals:
            stats = self.signal_stats[sig]
            quality_mult = stats.sharpe / total_sharpe if total_sharpe > 0 else 1/n
            budget[sig] = risk_per_signal * quality_mult * n

        return budget

    def update_equity(self, new_equity: float):
        """Update equity curve after trades"""
        self.current_capital = new_equity
        self.equity_curve.append(new_equity)
        self.max_equity = max(self.max_equity, new_equity)

    def get_state(self) -> PortfolioState:
        """Get current portfolio state"""
        return PortfolioState(
            cash_sol=self.current_capital - sum(self.positions.values()),
            positions=self.positions.copy(),
            total_value_sol=self.current_capital,
            current_drawdown=self._current_drawdown(),
            max_drawdown=self._max_drawdown(),
            regime=self.regime.detect(),
            risk_utilization=self._risk_utilization()
        )

    def _max_drawdown(self) -> float:
        """Calculate maximum historical drawdown"""
        if len(self.equity_curve) < 2:
            return 0.0

        peak = self.equity_curve[0]
        max_dd = 0.0

        for val in self.equity_curve:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            max_dd = max(max_dd, dd)

        return max_dd

    def _risk_utilization(self) -> float:
        """Calculate how much of risk budget is being used"""
        total_position = sum(self.positions.values())
        max_allowed = self.current_capital * MAX_PORTFOLIO_RISK
        return total_position / max_allowed if max_allowed > 0 else 0.0


# ============================================================
# RISK MANAGER
# ============================================================

class RiskManager:
    """
    Portfolio-level risk management.

    Enforces limits and hedging rules.
    """

    def __init__(
        self,
        max_position_pct: float = MAX_POSITION_FRACTION,
        max_concentration: float = MAX_CONCENTRATION,
        max_correlated_exposure: float = MAX_CORRELATED_EXPOSURE
    ):
        self.max_position_pct = max_position_pct
        self.max_concentration = max_concentration
        self.max_correlated_exposure = max_correlated_exposure

    def check_position(
        self,
        sizing: PositionSizing,
        portfolio: PortfolioOptimizer
    ) -> Tuple[bool, str, float]:
        """
        Check if position is allowed.

        Returns: (allowed, reason, adjusted_size)
        """
        state = portfolio.get_state()
        final_size = sizing.final_size_sol

        # 1. Max position size
        max_size = state.total_value_sol * self.max_position_pct
        if final_size > max_size:
            return (True, "Size capped at max position", max_size)

        # 2. Concentration check
        existing_position = state.positions.get(sizing.signal_name, 0)
        new_total = existing_position + final_size
        if new_total > state.total_value_sol * self.max_concentration:
            allowed_add = max(
                0,
                state.total_value_sol * self.max_concentration - existing_position
            )
            if allowed_add < final_size * 0.1:
                return (False, "Would exceed concentration limit", 0)
            return (True, "Reduced for concentration", allowed_add)

        # 3. Drawdown check
        if state.current_drawdown > 0.15:  # >15% drawdown
            reduced = final_size * 0.5
            return (True, "Reduced 50% due to drawdown", reduced)

        # 4. Risk utilization
        if state.risk_utilization > 0.8:  # >80% of risk budget
            reduced = final_size * 0.7
            return (True, "Reduced for risk budget", reduced)

        return (True, "OK", final_size)

    def should_hedge(self, portfolio: PortfolioOptimizer) -> bool:
        """Determine if hedging is needed"""
        state = portfolio.get_state()

        # Hedge if high drawdown
        if state.current_drawdown > 0.20:
            return True

        # Hedge if volatile regime
        if state.regime == MarketRegime.VOLATILE:
            return True

        return False

    def emergency_reduce(
        self,
        portfolio: PortfolioOptimizer
    ) -> Dict[str, float]:
        """
        Calculate emergency position reductions.

        Called when risk limits are breached.
        """
        state = portfolio.get_state()
        reductions = {}

        # Reduce all positions by drawdown excess
        if state.current_drawdown > 0.15:
            reduce_pct = min(0.5, state.current_drawdown)
            for token, size in state.positions.items():
                reductions[token] = size * reduce_pct

        return reductions


# ============================================================
# MAIN RUNNER
# ============================================================

def create_portfolio_optimizer(
    initial_capital: float = 10.0,
    kelly_fraction: float = FULL_KELLY_FRACTION
) -> Tuple[PortfolioOptimizer, RiskManager]:
    """Create portfolio optimizer with risk manager"""
    optimizer = PortfolioOptimizer(
        initial_capital=initial_capital,
        kelly_fraction=kelly_fraction
    )
    risk_mgr = RiskManager()

    return optimizer, risk_mgr


def example_usage():
    """Example of portfolio optimization"""
    print("=" * 60)
    print("  RENTECH PORTFOLIO OPTIMIZER")
    print("=" * 60)
    print()

    # Create optimizer
    optimizer, risk_mgr = create_portfolio_optimizer(initial_capital=10.0)

    # Add some signal stats
    signals = [
        ("momentum_5m", 0.55, 0.08, 0.05, 1.2, 150),
        ("whale_follow", 0.52, 0.12, 0.06, 1.5, 200),
        ("bonding_curve", 0.48, 0.15, 0.08, 1.1, 180),
        ("volume_spike", 0.45, 0.20, 0.10, 0.9, 120),
        ("smart_money", 0.58, 0.10, 0.06, 1.8, 250),
    ]

    for name, win_rate, avg_win, avg_loss, sharpe, count in signals:
        optimizer.update_signal_stats(
            signal_name=name,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            sharpe=sharpe,
            trade_count=count
        )

    # Add some trade results for correlation
    np.random.seed(42)
    for _ in range(100):
        for name, _, _, _, _, _ in signals:
            ret = np.random.normal(0.02, 0.05)
            vol = np.random.uniform(0.5, 2.0)
            optimizer.add_trade_result(name, ret, vol)

    # Calculate position sizes
    print("Position Sizing Recommendations:")
    print("-" * 60)

    for name, _, _, _, _, _ in signals:
        sizing = optimizer.calculate_position_size(name, signal_confidence=0.8)
        allowed, reason, adjusted = risk_mgr.check_position(sizing, optimizer)

        print(f"{name}:")
        print(f"  Kelly fraction: {optimizer.signal_stats[name].kelly_fraction:.1%}")
        print(f"  Final size: {sizing.final_size_sol:.4f} SOL")
        print(f"  Risk check: {reason} -> {adjusted:.4f} SOL")
        print()

    # Optimize weights
    print("Optimal Signal Weights:")
    print("-" * 60)

    weights = optimizer.optimize_weights([s[0] for s in signals])
    for sig, weight in sorted(weights.signal_weights.items(), key=lambda x: -x[1]):
        print(f"  {sig}: {weight:.1%}")

    print()
    print(f"Expected Sharpe: {weights.sharpe_ratio:.2f}")
    print(f"Diversification: {weights.diversification_ratio:.2f}")
    print()

    # Portfolio state
    state = optimizer.get_state()
    print("Portfolio State:")
    print("-" * 60)
    print(f"  Capital: {state.total_value_sol:.2f} SOL")
    print(f"  Regime: {state.regime.value}")
    print(f"  Current DD: {state.current_drawdown:.1%}")
    print(f"  Risk Usage: {state.risk_utilization:.1%}")


if __name__ == "__main__":
    example_usage()
