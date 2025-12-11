"""
Trading Signals Module
======================

Real-time signal generation for pump.fun trading.

Components:
- EarlyPatternScorer: Score tokens in first 5 minutes
- SignalAggregator: Combine multiple signal sources
- SignalEngine: Unified 96+ signal computation
- FormulaEngine: Apply 683 mathematical formulas
"""

from .signal_engine import SignalEngine, SignalResult, get_signal_engine

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from math import log10


@dataclass
class TokenScore:
    """Comprehensive token score with components"""
    mint: str
    total_score: float
    confidence: float
    signal: str  # 'strong_buy', 'buy', 'hold', 'avoid'
    components: Dict[str, float] = field(default_factory=dict)
    raw_metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "mint": self.mint,
            "score": self.total_score,
            "confidence": self.confidence,
            "signal": self.signal,
            "components": self.components,
            "timestamp": self.timestamp.isoformat()
        }


class EarlyPatternScorer:
    """
    Score tokens based on early trading patterns (first 5 minutes).

    Proven thresholds from historical pump.fun analysis:
    - Wallet diversity indicates organic interest
    - Buyer depth shows demand strength
    - Volume velocity indicates momentum
    - Buy pressure shows directional bias
    """

    # Minimum thresholds for consideration
    MIN_WALLETS = 15
    MIN_BUYERS = 10
    MIN_SOL = 5.0
    MIN_BUY_PRESSURE = 0.6

    # Scoring weights
    WEIGHT_WALLETS = 0.30
    WEIGHT_BUYERS = 0.25
    WEIGHT_VOLUME = 0.25
    WEIGHT_PRESSURE = 0.20

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode

    def score(
        self,
        unique_wallets: int,
        unique_buyers: int,
        total_sol: float,
        buy_sol: float,
        trade_count: int = 0,
        age_seconds: float = 0,
        price_change: float = 0,
    ) -> TokenScore:
        """
        Calculate comprehensive token score.

        Args:
            unique_wallets: Number of unique trading wallets
            unique_buyers: Number of unique buying wallets
            total_sol: Total SOL volume
            buy_sol: SOL volume from buys only
            trade_count: Total number of trades
            age_seconds: Token age in seconds
            price_change: Price change since first trade

        Returns:
            TokenScore with signal and confidence
        """
        components = {}

        # Wallet diversity score (0-1)
        wallet_score = min(1.0, unique_wallets / 30)
        components["wallet_diversity"] = wallet_score

        # Buyer depth score (0-1)
        buyer_score = min(1.0, unique_buyers / 20)
        components["buyer_depth"] = buyer_score

        # Volume score (log scale, 0-1)
        volume_score = min(1.0, log10(total_sol + 1) / 2) if total_sol > 0 else 0
        components["volume"] = volume_score

        # Buy pressure score (0-1)
        pressure = buy_sol / total_sol if total_sol > 0 else 0
        pressure_score = min(1.0, pressure / 0.8)
        components["buy_pressure"] = pressure_score

        # Calculate weighted total
        total_score = (
            self.WEIGHT_WALLETS * wallet_score +
            self.WEIGHT_BUYERS * buyer_score +
            self.WEIGHT_VOLUME * volume_score +
            self.WEIGHT_PRESSURE * pressure_score
        )

        # Velocity bonus (if age data available)
        if age_seconds > 0 and trade_count > 0:
            trades_per_minute = (trade_count / age_seconds) * 60
            velocity_bonus = min(0.1, trades_per_minute / 100)
            total_score += velocity_bonus
            components["velocity_bonus"] = velocity_bonus

        # Price momentum bonus
        if price_change > 0:
            momentum_bonus = min(0.1, price_change / 2)  # Max 10% bonus for 200%+ gain
            total_score += momentum_bonus
            components["momentum_bonus"] = momentum_bonus

        # Determine signal based on score and thresholds
        signal = self._determine_signal(
            total_score, unique_wallets, unique_buyers, total_sol, pressure
        )

        # Calculate confidence
        confidence = self._calculate_confidence(
            unique_wallets, unique_buyers, total_sol, trade_count
        )

        return TokenScore(
            mint="",  # Set by caller
            total_score=round(total_score, 4),
            confidence=round(confidence, 4),
            signal=signal,
            components=components,
            raw_metrics={
                "unique_wallets": unique_wallets,
                "unique_buyers": unique_buyers,
                "total_sol": total_sol,
                "buy_sol": buy_sol,
                "trade_count": trade_count,
                "age_seconds": age_seconds,
                "buy_pressure": pressure,
            }
        )

    def _determine_signal(
        self,
        score: float,
        wallets: int,
        buyers: int,
        sol: float,
        pressure: float,
    ) -> str:
        """Determine trading signal based on score and thresholds"""
        if self.strict_mode:
            # Must meet minimum thresholds
            if wallets < self.MIN_WALLETS:
                return "avoid"
            if buyers < self.MIN_BUYERS:
                return "avoid"
            if sol < self.MIN_SOL:
                return "avoid"
            if pressure < self.MIN_BUY_PRESSURE:
                return "hold"

        # Score-based signals
        if score >= 0.85:
            return "strong_buy"
        elif score >= 0.70:
            return "buy"
        elif score >= 0.50:
            return "hold"
        else:
            return "avoid"

    def _calculate_confidence(
        self,
        wallets: int,
        buyers: int,
        sol: float,
        trades: int,
    ) -> float:
        """Calculate confidence based on data quality"""
        # More data = higher confidence
        wallet_conf = min(1.0, wallets / 50)
        buyer_conf = min(1.0, buyers / 30)
        volume_conf = min(1.0, sol / 20)
        trade_conf = min(1.0, trades / 100) if trades > 0 else 0

        return (wallet_conf + buyer_conf + volume_conf + trade_conf) / 4


class SignalAggregator:
    """
    Aggregate signals from multiple sources using Bayesian methods.

    Uses log-odds aggregation (Formula 721) for combining
    independent probability estimates.
    """

    def __init__(self, prior: float = 0.5):
        """
        Args:
            prior: Prior probability of a good trade (0.5 = no bias)
        """
        self.prior = prior

    def aggregate(self, signals: List[TokenScore]) -> TokenScore:
        """
        Combine multiple TokenScores into unified signal.

        Uses log-odds Bayesian aggregation:
        log(p/(1-p)) = log(prior/(1-prior)) + sum(log(p_i/(1-p_i)))
        """
        if not signals:
            return TokenScore(
                mint="", total_score=0.5, confidence=0,
                signal="hold", components={}
            )

        if len(signals) == 1:
            return signals[0]

        # Convert scores to probabilities (assume score = P(good trade))
        log_odds = []
        for sig in signals:
            p = max(0.01, min(0.99, sig.total_score))  # Clamp to avoid log(0)
            log_odds.append(log10(p / (1 - p)))

        # Aggregate
        prior_odds = log10(self.prior / (1 - self.prior))
        total_log_odds = prior_odds + sum(log_odds)

        # Convert back to probability
        combined_prob = 10**total_log_odds / (1 + 10**total_log_odds)

        # Combine components
        combined_components = {}
        for sig in signals:
            for k, v in sig.components.items():
                if k not in combined_components:
                    combined_components[k] = []
                combined_components[k].append(v)

        # Average components
        avg_components = {k: sum(v)/len(v) for k, v in combined_components.items()}

        # Confidence is average of inputs
        avg_confidence = sum(s.confidence for s in signals) / len(signals)

        # Determine signal
        if combined_prob >= 0.85:
            signal = "strong_buy"
        elif combined_prob >= 0.70:
            signal = "buy"
        elif combined_prob >= 0.50:
            signal = "hold"
        else:
            signal = "avoid"

        return TokenScore(
            mint=signals[0].mint,
            total_score=round(combined_prob, 4),
            confidence=round(avg_confidence, 4),
            signal=signal,
            components=avg_components
        )


class PositionSizer:
    """
    Kelly criterion position sizing with constraints.

    Based on:
    - Formula 724: KellyCriterionWithEdge
    - Formula 725: DrawdownConstrainedKelly
    """

    def __init__(
        self,
        max_position_pct: float = 0.05,  # Max 5% per trade
        max_drawdown: float = 0.20,      # Max 20% portfolio drawdown
        win_rate: float = 0.55,          # Historical win rate
    ):
        self.max_position_pct = max_position_pct
        self.max_drawdown = max_drawdown
        self.win_rate = win_rate

    def calculate(
        self,
        score: float,
        confidence: float,
        portfolio_value: float,
    ) -> float:
        """
        Calculate position size in SOL.

        Args:
            score: Signal score (0-1)
            confidence: Confidence in signal (0-1)
            portfolio_value: Current portfolio value in SOL

        Returns:
            Position size in SOL
        """
        # Convert score to edge estimate
        edge = score - 0.5  # Score > 0.5 implies positive edge

        if edge <= 0:
            return 0

        # Adjust edge by confidence
        adjusted_edge = edge * confidence

        # Kelly fraction
        if adjusted_edge > 0:
            kelly = (self.win_rate * adjusted_edge - (1 - self.win_rate)) / adjusted_edge
        else:
            kelly = 0

        # Drawdown constraint (Grossman-Zhou optimal)
        safe_kelly = kelly * (1 - self.max_drawdown)

        # Apply maximum position limit
        position_pct = max(0, min(safe_kelly, self.max_position_pct))

        # Calculate actual position size
        position_sol = portfolio_value * position_pct

        return round(position_sol, 4)


__all__ = [
    'TokenScore',
    'EarlyPatternScorer',
    'SignalAggregator',
    'PositionSizer',
]
