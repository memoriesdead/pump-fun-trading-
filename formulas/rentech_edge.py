"""
RenTech Edge Extraction - Mathematical Trading Edge at Scale
=============================================================

Renaissance Technologies' Medallion Fund approach:
- 50.75% win rate with MILLIONS of trades (law of large numbers)
- 12.5-20x leverage (Kelly-optimized)
- Multiple signal categories combined
- Trade MORE frequently with SMALLER edges

The key insight: You don't need 80% win rate. You need:
  Expected Value = (Win% * Avg_Win) - (Loss% * Avg_Loss) > 0

With proper position sizing (Kelly), even 50.75% with 1:1 risk/reward prints money.

Formula IDs: 9000-9099 (RenTech Edge Series)
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from enum import Enum


class MarketRegime(Enum):
    """HMM-detected market regimes."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class EdgeSignal:
    """Computed edge signal with confidence."""
    edge_pct: float          # Expected edge in percent
    confidence: float        # 0-1 confidence level
    direction: int           # 1 = long, -1 = short, 0 = neutral
    regime: MarketRegime     # Current detected regime
    position_size: float     # Kelly-optimal position size (fraction)
    signals: Dict[str, float]  # Individual signal contributions


class RenTechEdgeExtractor:
    """
    Formula ID: 9000

    Core edge extraction using RenTech-style multi-signal approach.

    Instead of waiting for 50%+ momentum (too rare), we:
    1. Combine multiple weak signals into strong edge
    2. Trade frequently with small position sizes
    3. Let law of large numbers work for us
    4. Use leverage to amplify small edges

    Expected annual return with 50.75% win rate:
    - 1000 trades/year, 1:1 risk/reward, 1% per trade
    - Edge per trade = 0.0075 * 0.01 = 0.0075%
    - With 12x leverage = 0.09% per trade
    - 1000 trades * 0.09% = 90% annual return

    This is why RenTech made 66% annually for 30 years.
    """

    def __init__(
        self,
        lookback_periods: int = 20,
        min_edge_threshold: float = 0.005,  # 0.5% minimum edge
        kelly_fraction: float = 0.25,        # Quarter Kelly for safety
        max_leverage: float = 3.0,           # Max 3x for crypto (conservative)
    ):
        self.lookback_periods = lookback_periods
        self.min_edge_threshold = min_edge_threshold
        self.kelly_fraction = kelly_fraction
        self.max_leverage = max_leverage

        # Signal weights (learned from data, these are example weights)
        self.signal_weights = {
            'momentum_short': 0.15,
            'momentum_medium': 0.10,
            'mean_reversion': 0.20,
            'volatility_regime': 0.15,
            'order_flow': 0.25,
            'liquidity_signal': 0.15,
        }

    def compute_edge(
        self,
        prices: List[float],
        volumes: List[float],
        buy_ratio: float,
        liquidity_usd: float,
        volatility_5m: float,
        momentum_5m: float,
    ) -> EdgeSignal:
        """
        Compute trading edge from multiple signals.

        Args:
            prices: Recent price history (newest last)
            volumes: Recent volume history
            buy_ratio: Buy/total ratio (0-1)
            liquidity_usd: Pool liquidity in USD
            volatility_5m: 5-minute volatility %
            momentum_5m: 5-minute price change %

        Returns:
            EdgeSignal with combined edge and optimal sizing
        """
        signals = {}

        # 1. Short-term momentum (1-5 min) - trend continuation
        signals['momentum_short'] = self._momentum_signal(
            momentum_5m,
            threshold=5.0,  # Much lower than 50%!
            continuation_factor=0.3
        )

        # 2. Medium-term momentum reversal detection
        signals['momentum_medium'] = self._momentum_reversal_signal(
            prices,
            lookback=10
        )

        # 3. Mean reversion signal (Ornstein-Uhlenbeck based)
        signals['mean_reversion'] = self._mean_reversion_signal(
            prices,
            volatility_5m
        )

        # 4. Volatility regime signal
        signals['volatility_regime'] = self._volatility_regime_signal(
            volatility_5m,
            historical_vol=self._calculate_historical_vol(prices)
        )

        # 5. Order flow imbalance (buy pressure)
        signals['order_flow'] = self._order_flow_signal(
            buy_ratio,
            threshold=0.55  # Only need 55% buy ratio for edge
        )

        # 6. Liquidity-adjusted signal
        signals['liquidity_signal'] = self._liquidity_signal(
            liquidity_usd,
            optimal_range=(50000, 500000)  # Sweet spot
        )

        # Combine signals with weights
        combined_edge = sum(
            signals[name] * self.signal_weights[name]
            for name in signals
        )

        # Determine direction
        if combined_edge > self.min_edge_threshold:
            direction = 1  # Long
        elif combined_edge < -self.min_edge_threshold:
            direction = -1  # Short (if supported)
        else:
            direction = 0  # No trade

        # Calculate confidence (how aligned are signals)
        signal_values = list(signals.values())

        # Confidence based on:
        # 1. Signal alignment (same direction)
        # 2. Signal strength (magnitude)
        positive_signals = sum(1 for v in signal_values if v > 0)
        negative_signals = sum(1 for v in signal_values if v < 0)

        # Alignment: how many signals agree on direction
        total_nonzero = positive_signals + negative_signals
        if total_nonzero > 0:
            alignment = max(positive_signals, negative_signals) / total_nonzero
        else:
            alignment = 0.5

        # Strength: how strong is the average signal
        avg_strength = np.mean(np.abs(signal_values)) * 10  # Scale up
        strength_factor = min(1.0, avg_strength)

        # Combined confidence
        confidence = alignment * 0.7 + strength_factor * 0.3

        # Detect regime
        regime = self._detect_regime(volatility_5m, momentum_5m, prices)

        # Calculate Kelly-optimal position size
        position_size = self._kelly_position_size(
            edge=abs(combined_edge),
            win_prob=0.5 + abs(combined_edge) / 2,  # Edge translates to win prob
            confidence=confidence
        )

        return EdgeSignal(
            edge_pct=combined_edge,
            confidence=confidence,
            direction=direction,
            regime=regime,
            position_size=position_size,
            signals=signals
        )

    def _momentum_signal(
        self,
        momentum_pct: float,
        threshold: float,
        continuation_factor: float
    ) -> float:
        """
        Formula ID: 9001

        Short-term momentum with continuation probability.

        Research shows momentum continues ~40% of initial move in first 30s,
        then mean reverts. We capture the continuation.
        """
        if abs(momentum_pct) < threshold:
            return 0.0

        # Positive momentum -> expect continuation (but diminishing)
        # Cap at reasonable levels to avoid chasing
        capped_momentum = np.sign(momentum_pct) * min(abs(momentum_pct), 100)

        # Expected continuation = momentum * continuation_factor
        # But with diminishing returns for extreme momentum
        continuation = continuation_factor * np.tanh(capped_momentum / 50)

        return continuation

    def _momentum_reversal_signal(
        self,
        prices: List[float],
        lookback: int
    ) -> float:
        """
        Formula ID: 9002

        Detect momentum exhaustion for reversal trades.

        When price moves too fast too quickly, expect reversal.
        """
        if len(prices) < lookback:
            return 0.0

        recent = prices[-lookback:]
        returns = np.diff(recent) / recent[:-1]

        # Cumulative return
        cum_return = (recent[-1] / recent[0]) - 1

        # If strong trend with diminishing returns per period, expect reversal
        if len(returns) >= 3:
            trend = np.sign(cum_return)
            recent_slope = returns[-3:]

            # Momentum exhaustion: returns getting smaller while still trending
            if all(np.sign(r) == trend for r in recent_slope):
                if abs(recent_slope[-1]) < abs(recent_slope[0]) * 0.5:
                    # Exhaustion detected - reversal signal opposite to trend
                    return -trend * 0.1

        return 0.0

    def _mean_reversion_signal(
        self,
        prices: List[float],
        current_vol: float
    ) -> float:
        """
        Formula ID: 9003

        Ornstein-Uhlenbeck inspired mean reversion.

        dX_t = theta * (mu - X_t) * dt + sigma * dW_t

        When price deviates from mean, expect reversion proportional to deviation.
        """
        if len(prices) < 10:
            return 0.0

        prices_arr = np.array(prices)

        # Estimate mean (using EMA for adaptivity)
        alpha = 0.3
        ema = prices_arr[0]
        for p in prices_arr[1:]:
            ema = alpha * p + (1 - alpha) * ema

        # Current deviation from mean
        current_price = prices_arr[-1]
        deviation = (current_price - ema) / ema

        # Mean reversion strength (theta)
        # Higher volatility = faster reversion expected
        theta = 0.5 * (1 + current_vol / 100)

        # Reversion signal: negative of deviation (expect price to revert)
        # But cap to avoid extreme signals
        reversion_signal = -theta * np.tanh(deviation * 10)

        return reversion_signal

    def _volatility_regime_signal(
        self,
        current_vol: float,
        historical_vol: float
    ) -> float:
        """
        Formula ID: 9004

        Volatility regime trading.

        - High vol relative to history = opportunities (trade more)
        - Low vol = squeeze building (prepare for breakout)
        """
        if historical_vol < 1:
            historical_vol = 1

        vol_ratio = current_vol / historical_vol

        # High vol regime: more opportunity but more risk
        if vol_ratio > 2.0:
            # High vol = trade smaller but more frequently
            return 0.05 * (vol_ratio - 1)
        elif vol_ratio < 0.5:
            # Low vol = squeeze, prepare for breakout
            return 0.03
        else:
            # Normal regime
            return 0.01

    def _order_flow_signal(
        self,
        buy_ratio: float,
        threshold: float
    ) -> float:
        """
        Formula ID: 9005

        Order flow imbalance signal.

        More buyers than sellers = upward pressure.
        Simple but effective.
        """
        # Normalize around 0.5 (neutral)
        imbalance = buy_ratio - 0.5

        # Only signal if imbalance exceeds threshold
        if abs(imbalance) < (threshold - 0.5):
            return 0.0

        # Scale to reasonable signal strength
        return imbalance * 0.5

    def _liquidity_signal(
        self,
        liquidity_usd: float,
        optimal_range: Tuple[float, float]
    ) -> float:
        """
        Formula ID: 9006

        Liquidity sweet spot signal.

        - Too low liquidity = high slippage, avoid
        - Too high liquidity = efficient market, less edge
        - Sweet spot = enough liquidity for trades, still inefficient
        """
        min_liq, max_liq = optimal_range

        if liquidity_usd < min_liq:
            # Too illiquid - negative signal
            return -0.05 * (1 - liquidity_usd / min_liq)
        elif liquidity_usd > max_liq:
            # Too liquid - efficient market, less edge
            return -0.02 * min(1, (liquidity_usd - max_liq) / max_liq)
        else:
            # Sweet spot - positive signal
            # Optimal at middle of range
            mid = (min_liq + max_liq) / 2
            distance_from_mid = abs(liquidity_usd - mid) / (max_liq - min_liq)
            return 0.05 * (1 - distance_from_mid)

    def _calculate_historical_vol(self, prices: List[float]) -> float:
        """Calculate historical volatility from prices."""
        if len(prices) < 2:
            return 10.0  # Default

        returns = np.diff(prices) / prices[:-1]
        return float(np.std(returns) * 100 * np.sqrt(len(prices)))

    def _detect_regime(
        self,
        volatility: float,
        momentum: float,
        prices: List[float]
    ) -> MarketRegime:
        """
        Formula ID: 9007

        Simplified HMM regime detection.

        Full HMM would use Baum-Welch for parameter estimation,
        but for real-time we use heuristic classification.
        """
        # High volatility regime
        if volatility > 50:
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < 5:
            return MarketRegime.LOW_VOLATILITY

        # Trend vs mean reversion
        if len(prices) >= 5:
            recent_trend = (prices[-1] / prices[-5]) - 1

            if recent_trend > 0.10:  # 10%+ up
                return MarketRegime.TRENDING_UP
            elif recent_trend < -0.10:  # 10%+ down
                return MarketRegime.TRENDING_DOWN

        return MarketRegime.MEAN_REVERTING

    def _kelly_position_size(
        self,
        edge: float,
        win_prob: float,
        confidence: float
    ) -> float:
        """
        Formula ID: 9008

        Kelly Criterion with confidence adjustment.

        f* = (p * b - q) / b

        where:
            p = win probability
            q = 1 - p
            b = win/loss ratio (assuming 1:1 for simplicity)

        We use fractional Kelly (25%) for safety and adjust by confidence.
        """
        if edge <= 0 or win_prob <= 0.5:
            return 0.0

        # Assume 1:1 risk/reward for simplicity
        b = 1.0
        q = 1 - win_prob

        # Kelly formula
        kelly = (win_prob * b - q) / b

        # Apply safety fraction and confidence
        safe_kelly = kelly * self.kelly_fraction * confidence

        # Cap at max leverage
        return min(safe_kelly, 1.0 / self.max_leverage)


class ScaleTrader:
    """
    Formula ID: 9010

    Trade at scale with small edges.

    RenTech's insight: It's better to make 1000 trades with 0.5% edge
    than 10 trades with 5% edge. Law of large numbers reduces variance.

    Expected value is the same, but with 1000 trades:
    - Variance reduced by sqrt(1000/10) = 10x
    - More consistent returns
    - Faster capital deployment
    """

    def __init__(
        self,
        min_edge: float = 0.005,       # 0.5% minimum edge
        max_position_pct: float = 0.10, # 10% max per trade
        target_trades_per_day: int = 50,
        friction_per_trade: float = 0.003,  # 0.3% friction
    ):
        self.min_edge = min_edge
        self.max_position_pct = max_position_pct
        self.target_trades_per_day = target_trades_per_day
        self.friction_per_trade = friction_per_trade

    def should_trade(self, edge_signal: EdgeSignal) -> Tuple[bool, float, str]:
        """
        Determine if we should trade based on edge signal.

        Returns:
            (should_trade, position_size, reason)
        """
        # Net edge after friction
        net_edge = abs(edge_signal.edge_pct) - self.friction_per_trade

        if net_edge < self.min_edge:
            return False, 0.0, f"Insufficient edge: {net_edge:.2%} < {self.min_edge:.2%}"

        if edge_signal.direction == 0:
            return False, 0.0, "No directional signal"

        if edge_signal.confidence < 0.3:
            return False, 0.0, f"Low confidence: {edge_signal.confidence:.2%}"

        # Position size based on Kelly and confidence
        position_size = min(
            edge_signal.position_size,
            self.max_position_pct
        )

        # Scale down for lower confidence
        position_size *= edge_signal.confidence

        return True, position_size, f"Trade: {net_edge:.2%} edge, {edge_signal.regime.value}"

    def expected_daily_return(
        self,
        avg_edge: float = 0.0075,  # 0.75% average edge
        trades_per_day: int = 50,
        avg_position: float = 0.05,  # 5% average position
    ) -> Dict[str, float]:
        """
        Calculate expected daily return from scale trading.

        Formula ID: 9011

        This is the math that makes RenTech work:
        - Many small edges compound to large returns
        - Law of large numbers smooths variance
        """
        # Net edge after friction
        net_edge = avg_edge - self.friction_per_trade

        # Expected return per trade
        return_per_trade = net_edge * avg_position

        # Daily expected return
        daily_return = return_per_trade * trades_per_day

        # Variance reduction from many trades
        # Single trade variance ~ edge^2
        single_trade_variance = (avg_edge ** 2) * (avg_position ** 2)

        # With N trades, variance scales as 1/sqrt(N)
        daily_std = np.sqrt(single_trade_variance * trades_per_day) / np.sqrt(trades_per_day)

        # Sharpe ratio (annualized)
        annual_return = daily_return * 252
        annual_std = daily_std * np.sqrt(252)
        sharpe = annual_return / annual_std if annual_std > 0 else 0

        return {
            'net_edge_per_trade': net_edge,
            'trades_per_day': trades_per_day,
            'expected_daily_return': daily_return,
            'daily_std': daily_std,
            'expected_annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'win_rate_needed': 0.5 + net_edge / 2,  # For 1:1 risk/reward
        }


class AdaptiveEdgeOptimizer:
    """
    Formula ID: 9020

    Adaptive optimization of edge parameters based on market conditions.

    RenTech continuously updated their models. We do the same by:
    1. Tracking trade outcomes
    2. Adjusting signal weights
    3. Updating regime detection thresholds
    """

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.trade_history: List[Dict] = []
        self.signal_performance: Dict[str, List[float]] = {
            'momentum_short': [],
            'momentum_medium': [],
            'mean_reversion': [],
            'volatility_regime': [],
            'order_flow': [],
            'liquidity_signal': [],
        }

    def record_trade(
        self,
        edge_signal: EdgeSignal,
        outcome: float,  # Actual PnL percent
    ):
        """Record trade outcome for learning."""
        self.trade_history.append({
            'signals': edge_signal.signals.copy(),
            'predicted_edge': edge_signal.edge_pct,
            'actual_outcome': outcome,
            'regime': edge_signal.regime,
        })

        # Track individual signal performance
        for name, value in edge_signal.signals.items():
            # Signal was correct if sign matches outcome
            correct = np.sign(value) == np.sign(outcome)
            self.signal_performance[name].append(1 if correct else 0)

    def get_optimized_weights(self) -> Dict[str, float]:
        """
        Formula ID: 9021

        Calculate optimized signal weights based on performance.
        """
        if len(self.trade_history) < 10:
            # Not enough data, return default weights
            return {
                'momentum_short': 0.15,
                'momentum_medium': 0.10,
                'mean_reversion': 0.20,
                'volatility_regime': 0.15,
                'order_flow': 0.25,
                'liquidity_signal': 0.15,
            }

        # Calculate hit rate for each signal
        weights = {}
        total_weight = 0

        for name, outcomes in self.signal_performance.items():
            if len(outcomes) >= 5:
                hit_rate = np.mean(outcomes[-50:])  # Recent 50 trades
                # Weight based on hit rate (above 50% is positive)
                weight = max(0.05, hit_rate)
                weights[name] = weight
                total_weight += weight
            else:
                weights[name] = 0.15
                total_weight += 0.15

        # Normalize to sum to 1
        for name in weights:
            weights[name] /= total_weight

        return weights


# Convenience functions for trading engine integration

def compute_rentech_edge(
    prices: List[float],
    volumes: List[float],
    buy_ratio: float,
    liquidity_usd: float,
    volatility_5m: float,
    momentum_5m: float,
) -> EdgeSignal:
    """
    Convenience function to compute RenTech-style edge.

    Usage:
        from formulas.rentech_edge import compute_rentech_edge

        signal = compute_rentech_edge(
            prices=[100, 101, 102, 103],
            volumes=[1000, 1100, 1200, 1300],
            buy_ratio=0.65,
            liquidity_usd=100000,
            volatility_5m=25.0,
            momentum_5m=8.0
        )

        if signal.direction != 0:
            print(f"Trade signal: {signal.edge_pct:.2%} edge")
    """
    extractor = RenTechEdgeExtractor()
    return extractor.compute_edge(
        prices=prices,
        volumes=volumes,
        buy_ratio=buy_ratio,
        liquidity_usd=liquidity_usd,
        volatility_5m=volatility_5m,
        momentum_5m=momentum_5m,
    )


def get_trade_decision(
    edge_signal: EdgeSignal,
    friction_pct: float = 0.003,
) -> Tuple[bool, float, str]:
    """
    Get trade decision from edge signal.

    Returns:
        (should_trade, position_size_pct, reason)
    """
    trader = ScaleTrader(friction_per_trade=friction_pct)
    return trader.should_trade(edge_signal)


# Example usage and testing
if __name__ == "__main__":
    print("RenTech Edge Extractor Test")
    print("=" * 60)

    # Simulate some price data
    base_price = 0.001
    prices = [base_price * (1 + 0.02 * i + 0.01 * np.random.randn()) for i in range(20)]
    volumes = [10000 * (1 + 0.1 * np.random.randn()) for _ in range(20)]

    # Compute edge
    signal = compute_rentech_edge(
        prices=prices,
        volumes=volumes,
        buy_ratio=0.65,
        liquidity_usd=100000,
        volatility_5m=25.0,
        momentum_5m=8.0
    )

    print(f"\nEdge Signal:")
    print(f"  Combined Edge: {signal.edge_pct:.2%}")
    print(f"  Direction: {'Long' if signal.direction > 0 else 'Short' if signal.direction < 0 else 'Neutral'}")
    print(f"  Confidence: {signal.confidence:.2%}")
    print(f"  Regime: {signal.regime.value}")
    print(f"  Position Size: {signal.position_size:.2%}")
    print(f"\nIndividual Signals:")
    for name, value in signal.signals.items():
        print(f"  {name}: {value:+.4f}")

    # Get trade decision
    should_trade, size, reason = get_trade_decision(signal)
    print(f"\nTrade Decision:")
    print(f"  Should Trade: {should_trade}")
    print(f"  Position Size: {size:.2%}")
    print(f"  Reason: {reason}")

    # Show expected returns at scale
    trader = ScaleTrader()
    returns = trader.expected_daily_return()
    print(f"\nExpected Returns at Scale:")
    print(f"  Net Edge/Trade: {returns['net_edge_per_trade']:.2%}")
    print(f"  Trades/Day: {returns['trades_per_day']}")
    print(f"  Expected Daily: {returns['expected_daily_return']:.2%}")
    print(f"  Expected Annual: {returns['expected_annual_return']:.1%}")
    print(f"  Sharpe Ratio: {returns['sharpe_ratio']:.2f}")
    print(f"  Win Rate Needed: {returns['win_rate_needed']:.1%}")
