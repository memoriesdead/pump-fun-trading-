"""
RenTech-Style Token Filter - Trade Frequently with Small Edges
==============================================================

Key differences from standard filter:
1. NO minimum momentum requirement (was 50%)
2. Multi-signal edge extraction instead of simple momentum
3. Trade more frequently with smaller position sizes
4. Let law of large numbers work for us

RenTech made 66% annually with 50.75% win rate by:
- Trading millions of times per year
- Small edges compound with many trades
- Kelly Criterion position sizing
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from .config import FilterConfig
from .token import Token
from .friction import FRICTION


@dataclass
class RenTechFilterResult:
    """Result of RenTech-style filtering."""
    passed: bool
    reasons: List[str]

    # Multi-signal edge
    combined_edge: float
    confidence: float
    direction: int  # 1=long, -1=short, 0=neutral

    # Cost analysis
    friction_cost: float
    net_edge: float

    # Position sizing (Kelly-derived)
    position_size_pct: float

    # Individual signals
    signals: dict

    @property
    def is_tradeable(self) -> bool:
        return self.passed and self.net_edge > 0 and self.direction != 0


class RenTechFilter:
    """
    RenTech-style multi-signal filter.

    Instead of waiting for 50%+ momentum:
    1. Combine multiple weak signals
    2. Trade when combined edge exceeds friction
    3. Scale position size by confidence

    Expected: 50+ trades/day with 50.75% win rate
    """

    def __init__(
        self,
        min_liquidity: float = 25000,       # $25k min
        max_liquidity: float = 50_000_000,  # $50M max (was $10M - too restrictive)
        min_edge_after_friction: float = 0.002,  # 0.2% net edge (was 0.5% - too strict)
        kelly_fraction: float = 0.25,       # Quarter Kelly
        default_trade_size: float = 10.0,   # Default $10 trade (not $1!)
    ):
        self.min_liquidity = min_liquidity
        self.max_liquidity = max_liquidity
        self.min_edge_after_friction = min_edge_after_friction
        self.kelly_fraction = kelly_fraction
        self.default_trade_size = default_trade_size

        # Signal weights (can be optimized with backtesting)
        self.signal_weights = {
            'momentum_short': 0.15,
            'momentum_medium': 0.10,
            'mean_reversion': 0.20,
            'volatility_regime': 0.15,
            'order_flow': 0.25,
            'liquidity_signal': 0.15,
        }

    def check(self, token: Token, trade_size_usd: float = None) -> RenTechFilterResult:
        """
        Check token using RenTech-style multi-signal approach.

        Args:
            token: Token to evaluate
            trade_size_usd: Expected trade size for friction calculation

        Returns:
            RenTechFilterResult with edge and position sizing
        """
        # Use default trade size if not specified
        if trade_size_usd is None:
            trade_size_usd = self.default_trade_size

        reasons = []
        signals = {}

        # Basic sanity filters (keep these strict)
        if token.liquidity_usd < self.min_liquidity:
            reasons.append(f"liquidity ${token.liquidity_usd:,.0f} < ${self.min_liquidity:,.0f}")
        if token.liquidity_usd > self.max_liquidity:
            reasons.append(f"liquidity ${token.liquidity_usd:,.0f} > ${self.max_liquidity:,.0f}")

        # Age filter (not too new - skip too_old for trending pools)
        if token.age_seconds < 30:
            reasons.append(f"too_new {token.age_seconds:.0f}s")
        # Removed too_old filter - trending pools are often established tokens
        # RenTech trades based on edge, not age

        # Calculate multi-signal edge
        signals['momentum_short'] = self._momentum_signal(token)
        signals['momentum_medium'] = self._momentum_reversal_signal(token)
        signals['mean_reversion'] = self._mean_reversion_signal(token)
        signals['volatility_regime'] = self._volatility_regime_signal(token)
        signals['order_flow'] = self._order_flow_signal(token)
        signals['liquidity_signal'] = self._liquidity_signal(token)

        # Combine signals with weights
        combined_edge = sum(
            signals[name] * self.signal_weights[name]
            for name in signals
        )

        # Calculate confidence from signal alignment
        positive_signals = sum(1 for v in signals.values() if v > 0)
        negative_signals = sum(1 for v in signals.values() if v < 0)
        total_nonzero = positive_signals + negative_signals

        if total_nonzero > 0:
            alignment = max(positive_signals, negative_signals) / total_nonzero
        else:
            alignment = 0.5

        avg_strength = np.mean(np.abs(list(signals.values()))) * 10
        strength_factor = min(1.0, avg_strength)
        confidence = alignment * 0.7 + strength_factor * 0.3

        # Determine direction
        if combined_edge > 0.005:  # 0.5% threshold
            direction = 1  # Long
        elif combined_edge < -0.005:
            direction = -1  # Short (if supported)
        else:
            direction = 0  # No trade

        # Calculate friction
        friction = FRICTION.compute_total_cost(
            trade_size_usd=trade_size_usd,
            liquidity_usd=token.liquidity_usd,
            is_pumpfun=token.is_pumpfun,
        )
        friction_cost = friction['total_cost_pct']

        # Net edge after friction
        net_edge = abs(combined_edge) - friction_cost

        # Check minimum edge requirement
        if net_edge < self.min_edge_after_friction and direction != 0:
            reasons.append(f"net_edge {net_edge:.2%} < {self.min_edge_after_friction:.2%}")

        # Calculate Kelly position size
        if net_edge > 0 and direction != 0:
            win_prob = 0.5 + net_edge / 2  # Edge translates to win prob
            position_size_pct = self._kelly_size(win_prob, confidence)
        else:
            position_size_pct = 0.0

        return RenTechFilterResult(
            passed=len(reasons) == 0,
            reasons=reasons,
            combined_edge=combined_edge,
            confidence=confidence,
            direction=direction,
            friction_cost=friction_cost,
            net_edge=net_edge,
            position_size_pct=position_size_pct,
            signals=signals,
        )

    def _momentum_signal(self, token: Token) -> float:
        """
        Short-term momentum signal.

        Lower threshold than before - even 5% momentum is useful
        when combined with other signals.
        """
        momentum = token.momentum

        if abs(momentum) < 2:  # Less than 2% - no signal
            return 0.0

        # Continuation factor (40% of momentum continues in short term)
        continuation = 0.4

        # Cap at reasonable levels
        capped = np.sign(momentum) * min(abs(momentum), 100)

        return (capped / 100) * continuation * 0.5

    def _momentum_reversal_signal(self, token: Token) -> float:
        """
        Detect momentum exhaustion.

        When momentum is extreme, expect reversal.
        """
        momentum = abs(token.momentum)

        # Extreme momentum (>100%) suggests exhaustion
        if momentum > 100:
            return -np.sign(token.momentum) * 0.05

        return 0.0

    def _mean_reversion_signal(self, token: Token) -> float:
        """
        Mean reversion signal based on volatility.

        High volatility suggests overshooting, expect reversion.
        """
        volatility = abs(token.change_5m)
        momentum = token.momentum

        # If big move but momentum is losing steam, expect reversion
        if volatility > 20 and abs(momentum) < volatility * 0.5:
            return -np.sign(momentum) * 0.03

        return 0.0

    def _volatility_regime_signal(self, token: Token) -> float:
        """
        Volatility regime signal.

        High volatility = more opportunities.
        """
        volatility = abs(token.change_5m)

        if volatility > 30:
            return 0.05  # High vol regime - bullish for trading
        elif volatility > 10:
            return 0.02  # Normal vol
        elif volatility < 2:
            return -0.02  # Low vol squeeze

        return 0.01

    def _order_flow_signal(self, token: Token) -> float:
        """
        Order flow imbalance signal.

        More buyers than sellers = bullish.
        """
        buy_ratio = token.buy_ratio

        # Imbalance from 50% neutral
        imbalance = buy_ratio - 0.5

        # Only signal if significant imbalance
        if abs(imbalance) < 0.05:
            return 0.0

        # Scale to reasonable signal strength
        return imbalance * 0.3

    def _liquidity_signal(self, token: Token) -> float:
        """
        Liquidity sweet spot signal.

        Not too liquid (efficient), not too illiquid (slippage).
        """
        liq = token.liquidity_usd

        # Sweet spot: $50k - $500k
        if 50000 <= liq <= 500000:
            return 0.03  # In sweet spot
        elif liq < 50000:
            return -0.02 * (1 - liq / 50000)  # Illiquid penalty
        else:
            return -0.01  # Too efficient

    def _kelly_size(self, win_prob: float, confidence: float) -> float:
        """
        Calculate Kelly-optimal position size.

        f* = (p * b - q) / b

        We use quarter Kelly for safety and adjust by confidence.
        """
        if win_prob <= 0.5:
            return 0.0

        # Assume 1:1 risk/reward
        b = 1.0
        q = 1 - win_prob

        # Kelly formula
        kelly = (win_prob * b - q) / b

        # Apply safety fraction and confidence
        safe_kelly = kelly * self.kelly_fraction * confidence

        # Cap at 10% of capital per trade (conservative)
        return min(safe_kelly, 0.10)

    def get_best_opportunities(
        self,
        tokens: List[Token],
        max_results: int = 10,
        trade_size_usd: float = 10.0,
    ) -> List[Tuple[Token, RenTechFilterResult]]:
        """
        Get best trading opportunities from a batch.

        Args:
            tokens: List of tokens to evaluate
            max_results: Max opportunities to return
            trade_size_usd: Expected trade size

        Returns:
            List of (token, result) sorted by net edge
        """
        results = []

        for token in tokens:
            result = self.check(token, trade_size_usd)
            if result.is_tradeable:
                results.append((token, result))

        # Sort by net edge (highest first)
        results.sort(key=lambda x: x[1].net_edge, reverse=True)

        return results[:max_results]

    def analyze_opportunities(self, tokens: List[Token]) -> dict:
        """
        Analyze opportunity distribution.

        Shows how many tokens pass each filter level.
        """
        total = len(tokens)
        passed_liquidity = 0
        passed_age = 0
        has_direction = 0
        has_edge = 0
        fully_tradeable = 0

        for token in tokens:
            result = self.check(token)

            # Check individual criteria
            if self.min_liquidity <= token.liquidity_usd <= self.max_liquidity:
                passed_liquidity += 1
            if 30 <= token.age_seconds <= 86400 * 7:
                passed_age += 1
            if result.direction != 0:
                has_direction += 1
            if result.net_edge > 0:
                has_edge += 1
            if result.is_tradeable:
                fully_tradeable += 1

        return {
            'total': total,
            'passed_liquidity': passed_liquidity,
            'passed_age': passed_age,
            'has_direction': has_direction,
            'has_positive_edge': has_edge,
            'fully_tradeable': fully_tradeable,
            'trade_rate': fully_tradeable / total if total > 0 else 0,
        }
