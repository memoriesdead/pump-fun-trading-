"""
INFINITE POSSIBILITIES ENGINE - Formula IDs 700-706
====================================================
Mathematical Framework for Capturing ALL Trading Opportunities

Based on:
- Whitrow (2007): Algorithms for optimal allocation of bets on many simultaneous events
- AAAI 2025: Adaptive Multi-Scale Decomposition Framework
- Clemen & Winkler (1999): Combining Probability Forecasts
- Easley: Optimal Execution Horizon
- Kelly (1956): Optimal Growth Criterion

PURPOSE:
With $100 capital, capture EVERY mathematical edge across ALL timescales
simultaneously. Evaluate infinite possibilities every second.

ID MAPPING:
700: SimultaneousKellyWhitrow     - Multi-bet optimal allocation
701: AdaptiveScaleSelector        - Which timescale is working NOW
702: BayesianProbabilityAggregator - Combine probabilities from all scales
703: HotScaleDetector             - Real-time scale performance tracking
704: SignalFreshnessIndex         - Signal age decay
705: CrossScaleCorrelationMonitor - Detect alignment/divergence
706: InfinitePossibilitiesController - MASTER combining all above
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import time
from scipy.optimize import minimize, Bounds
from scipy.special import expit  # sigmoid function

from .base import BaseFormula, FormulaRegistry


# =============================================================================
# ID 700: SIMULTANEOUS KELLY (WHITROW 2007)
# =============================================================================

class SimultaneousKellyWhitrow(BaseFormula):
    """
    ID 700: Optimal allocation across MULTIPLE simultaneous bets.

    Paper: Whitrow, C. (2007). "Algorithms for optimal allocation of bets
           on many simultaneous events." Journal of the Royal Statistical
           Society: Series C, 56: 607-623.

    Key Insight: When betting on multiple events simultaneously, optimal
    allocations are SMALLER than individual Kelly bets and distributed
    proportionally to edge.

    Formula:
        Maximize: E[log(W)] = E[log(1 + Σ f_i * R_i)]
        Subject to: Σ f_i ≤ 1 (total ≤ bankroll)

    Solution: Numerical optimization via gradient descent
    """

    FORMULA_ID = 700
    NAME = "SimultaneousKellyWhitrow"
    CATEGORY = "infinite_possibilities"

    def __init__(self, max_total_allocation: float = 0.8, min_bet: float = 0.02, **kwargs):
        """
        Args:
            max_total_allocation: Maximum total capital to risk (default 80%)
            min_bet: Minimum bet size as fraction of capital (default 2%)
        """
        super().__init__(**kwargs)
        self.max_total_allocation = max_total_allocation
        self.min_bet = min_bet
        self.n_simulations = 1000  # Monte Carlo simulations for gradient
        self.last_result = {}

    def _compute(self) -> None:
        """Required by BaseFormula - uses last calculate() result."""
        pass  # Signal computed via calculate()

    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate optimal allocation across multiple simultaneous bets.

        Args:
            data: {
                'opportunities': [
                    {'win_prob': 0.55, 'win_amount': 1.5, 'loss_amount': 1.0, 'timescale': 1},
                    {'win_prob': 0.60, 'win_amount': 1.2, 'loss_amount': 1.0, 'timescale': 5},
                    ...
                ],
                'capital': 100.0
            }

        Returns:
            {
                'allocations': [0.10, 0.15, ...],  # Fraction per bet
                'dollar_amounts': [10.0, 15.0, ...],
                'expected_log_growth': 0.05,
                'total_allocation': 0.45
            }
        """
        opportunities = data.get('opportunities', [])
        capital = data.get('capital', 100.0)

        if not opportunities:
            return {'allocations': [], 'dollar_amounts': [], 'expected_log_growth': 0.0, 'total_allocation': 0.0}

        n_bets = len(opportunities)

        # Extract parameters
        win_probs = np.array([o['win_prob'] for o in opportunities])
        win_amounts = np.array([o.get('win_amount', 1.5) for o in opportunities])
        loss_amounts = np.array([o.get('loss_amount', 1.0) for o in opportunities])

        # Calculate edges
        edges = win_probs * win_amounts - (1 - win_probs) * loss_amounts

        # Filter to positive edge only
        positive_mask = edges > 0
        if not np.any(positive_mask):
            return {'allocations': [0.0] * n_bets, 'dollar_amounts': [0.0] * n_bets,
                    'expected_log_growth': 0.0, 'total_allocation': 0.0}

        # Objective: Maximize expected log wealth
        def neg_expected_log_wealth(fractions):
            # Simulate outcomes
            total_log = 0.0
            for _ in range(self.n_simulations):
                # Generate random outcomes for each bet
                outcomes = np.random.random(n_bets) < win_probs
                returns = np.where(outcomes, win_amounts, -loss_amounts)
                portfolio_return = np.sum(fractions * returns)
                wealth = 1 + portfolio_return
                if wealth > 0:
                    total_log += np.log(wealth)
                else:
                    total_log += -100  # Heavily penalize bankruptcy
            return -total_log / self.n_simulations

        # Constraint: Total allocation ≤ max
        def total_constraint(fractions):
            return self.max_total_allocation - np.sum(fractions)

        # Initial guess: Proportional to edge
        initial_fractions = np.maximum(0, edges) / np.sum(np.maximum(0, edges)) * self.max_total_allocation * 0.5

        # Bounds: Each fraction between 0 and max_total_allocation
        bounds = Bounds(lb=np.zeros(n_bets), ub=np.ones(n_bets) * self.max_total_allocation)

        # Optimize
        try:
            result = minimize(
                neg_expected_log_wealth,
                initial_fractions,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': total_constraint},
                options={'maxiter': 100}
            )
            optimal_fractions = np.maximum(0, result.x)
        except Exception:
            # Fallback to edge-proportional
            optimal_fractions = initial_fractions

        # Apply minimum bet filter
        optimal_fractions = np.where(optimal_fractions < self.min_bet, 0, optimal_fractions)

        # Normalize if exceeds max
        if np.sum(optimal_fractions) > self.max_total_allocation:
            optimal_fractions = optimal_fractions / np.sum(optimal_fractions) * self.max_total_allocation

        dollar_amounts = optimal_fractions * capital
        expected_growth = -neg_expected_log_wealth(optimal_fractions)

        return {
            'allocations': optimal_fractions.tolist(),
            'dollar_amounts': dollar_amounts.tolist(),
            'expected_log_growth': float(expected_growth),
            'total_allocation': float(np.sum(optimal_fractions)),
            'edges': edges.tolist()
        }


# =============================================================================
# ID 701: ADAPTIVE SCALE SELECTOR (AAAI 2025)
# =============================================================================

class AdaptiveScaleSelector(BaseFormula):
    """
    ID 701: Dynamically select which timescale is optimal NOW.

    Paper: Hu et al. (2025). "Adaptive Multi-Scale Decomposition Framework
           for Time Series Forecasting." AAAI Conference.

    Concept:
        - Track performance of each timescale over recent history
        - Softmax weighting based on performance
        - Scales that are "hot" get more weight
    """

    FORMULA_ID = 701
    NAME = "AdaptiveScaleSelector"
    CATEGORY = "infinite_possibilities"

    # Standard timescales in seconds
    TIMESCALES = [1, 2, 5, 10, 30, 60]

    def __init__(self, lookback: int = 50, temperature: float = 1.0, **kwargs):
        """
        Args:
            lookback: Number of recent trades to consider per scale
            temperature: Softmax temperature (lower = more selective)
        """
        super().__init__(lookback=lookback, **kwargs)
        self.temperature = temperature

        # Track performance per timescale
        self.scale_performance = {ts: deque(maxlen=lookback) for ts in self.TIMESCALES}
        self.scale_trades = {ts: 0 for ts in self.TIMESCALES}

    def _compute(self) -> None:
        """Required by BaseFormula."""
        pass

    def update_performance(self, timescale: int, pnl: float, won: bool):
        """Record a trade outcome for a timescale."""
        if timescale in self.scale_performance:
            self.scale_performance[timescale].append((pnl, won))
            self.scale_trades[timescale] += 1

    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate adaptive weights for each timescale.

        Returns:
            {
                'weights': {1: 0.25, 2: 0.20, 5: 0.15, ...},
                'hot_scales': [1, 2],  # Top performing
                'cold_scales': [60],   # Underperforming
                'scores': {1: 1.5, 2: 1.2, ...}
            }
        """
        scores = {}

        for ts in self.TIMESCALES:
            perf = list(self.scale_performance[ts])
            if len(perf) < 3:
                # Not enough data, neutral score
                scores[ts] = 0.0
            else:
                pnls = [p[0] for p in perf]
                wins = [p[1] for p in perf]

                # Score = win_rate_excess * sqrt(n) + total_pnl_normalized
                win_rate = sum(wins) / len(wins)
                win_rate_excess = win_rate - 0.5
                confidence = np.sqrt(len(perf))

                total_pnl = sum(pnls)

                scores[ts] = win_rate_excess * confidence + total_pnl * 0.1

        # Softmax to get weights
        score_values = np.array([scores[ts] for ts in self.TIMESCALES])
        exp_scores = np.exp(score_values / self.temperature)
        weights_array = exp_scores / np.sum(exp_scores)

        weights = {ts: float(w) for ts, w in zip(self.TIMESCALES, weights_array)}

        # Identify hot and cold scales
        sorted_scales = sorted(self.TIMESCALES, key=lambda ts: scores[ts], reverse=True)
        hot_scales = [ts for ts in sorted_scales[:2] if scores[ts] > 0]
        cold_scales = [ts for ts in sorted_scales[-2:] if scores[ts] < 0]

        return {
            'weights': weights,
            'hot_scales': hot_scales,
            'cold_scales': cold_scales,
            'scores': scores,
            'total_trades': sum(self.scale_trades.values())
        }


# =============================================================================
# ID 702: BAYESIAN PROBABILITY AGGREGATOR
# =============================================================================

class BayesianProbabilityAggregator(BaseFormula):
    """
    ID 702: Combine probabilities from multiple timescales using Bayesian methods.

    Paper: Clemen & Winkler (1999). "Combining Probability Distributions
           From Experts in Risk Analysis." Risk Analysis.

    Method: Log-odds aggregation with learned weights

    Formula:
        log(p_combined / (1-p_combined)) = Σ w_i * log(p_i / (1-p_i))
    """

    FORMULA_ID = 702
    NAME = "BayesianProbabilityAggregator"
    CATEGORY = "infinite_possibilities"

    def __init__(self, prior: float = 0.5, **kwargs):
        """
        Args:
            prior: Prior probability when no signals available
        """
        super().__init__(**kwargs)
        self.prior = prior
        self.weight_history = deque(maxlen=100)

    def _compute(self) -> None:
        """Required by BaseFormula."""
        pass

    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate probabilities from multiple timescales.

        Args:
            data: {
                'probabilities': {1: 0.55, 2: 0.60, 5: 0.52, ...},  # per timescale
                'weights': {1: 0.25, 2: 0.30, ...},  # from AdaptiveScaleSelector
                'directions': {1: 1, 2: 1, 5: -1, ...}  # +1 long, -1 short
            }

        Returns:
            {
                'combined_probability': 0.58,
                'combined_direction': 1,
                'confidence': 0.75,
                'log_odds': 0.33
            }
        """
        probs = data.get('probabilities', {})
        weights = data.get('weights', {})
        directions = data.get('directions', {})

        if not probs:
            return {
                'combined_probability': self.prior,
                'combined_direction': 0,
                'confidence': 0.0,
                'log_odds': 0.0
            }

        # Separate by direction
        long_log_odds = 0.0
        long_weight_sum = 0.0
        short_log_odds = 0.0
        short_weight_sum = 0.0

        for ts, prob in probs.items():
            weight = weights.get(ts, 1.0 / len(probs))
            direction = directions.get(ts, 1)

            # Clamp probability to avoid log(0)
            prob = np.clip(prob, 0.01, 0.99)
            log_odd = np.log(prob / (1 - prob))

            if direction > 0:
                long_log_odds += weight * log_odd
                long_weight_sum += weight
            else:
                short_log_odds += weight * log_odd
                short_weight_sum += weight

        # Normalize
        if long_weight_sum > 0:
            long_log_odds /= long_weight_sum
        if short_weight_sum > 0:
            short_log_odds /= short_weight_sum

        # Convert back to probabilities
        long_prob = expit(long_log_odds) if long_weight_sum > 0 else 0.5
        short_prob = expit(short_log_odds) if short_weight_sum > 0 else 0.5

        # Decide direction based on which has stronger signal
        if long_weight_sum > short_weight_sum and long_prob > 0.5:
            combined_prob = long_prob
            combined_dir = 1
            log_odds = long_log_odds
        elif short_weight_sum > long_weight_sum and short_prob > 0.5:
            combined_prob = short_prob
            combined_dir = -1
            log_odds = short_log_odds
        elif long_prob > short_prob:
            combined_prob = long_prob
            combined_dir = 1
            log_odds = long_log_odds
        else:
            combined_prob = short_prob
            combined_dir = -1
            log_odds = short_log_odds

        # Confidence based on agreement and strength
        all_dirs = list(directions.values())
        agreement = abs(sum(all_dirs)) / len(all_dirs) if all_dirs else 0
        strength = abs(combined_prob - 0.5) * 2
        confidence = agreement * 0.5 + strength * 0.5

        return {
            'combined_probability': float(combined_prob),
            'combined_direction': int(combined_dir),
            'confidence': float(confidence),
            'log_odds': float(log_odds),
            'long_prob': float(long_prob),
            'short_prob': float(short_prob)
        }


# =============================================================================
# ID 703: HOT SCALE DETECTOR
# =============================================================================

class HotScaleDetector(BaseFormula):
    """
    ID 703: Real-time detection of which timescale is performing best NOW.

    Based on: Easley et al. - Optimal Execution Horizon

    Formula:
        HotScore_k = (hit_rate_k - 0.5) * sqrt(trades_k) * (1 + pnl_k / capital)

    Timescales with HotScore > threshold are "hot" and should get more capital.
    """

    FORMULA_ID = 703
    NAME = "HotScaleDetector"
    CATEGORY = "infinite_possibilities"

    TIMESCALES = [1, 2, 5, 10, 30, 60]

    def __init__(self, window: int = 20, hot_threshold: float = 0.5, **kwargs):
        """
        Args:
            window: Rolling window for performance calculation
            hot_threshold: Score above which a scale is "hot"
        """
        super().__init__(lookback=window, **kwargs)
        self.window = window
        self.hot_threshold = hot_threshold

        # Track per timescale
        self.trades = {ts: deque(maxlen=window) for ts in self.TIMESCALES}
        self.last_update = {ts: 0 for ts in self.TIMESCALES}

    def _compute(self) -> None:
        """Required by BaseFormula."""
        pass

    def record_trade(self, timescale: int, won: bool, pnl: float):
        """Record a trade outcome."""
        if timescale in self.trades:
            self.trades[timescale].append({'won': won, 'pnl': pnl, 'time': time.time()})
            self.last_update[timescale] = time.time()

    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate HotScore for each timescale.

        Args:
            data: {'capital': 100.0}

        Returns:
            {
                'hot_scores': {1: 1.2, 2: 0.8, ...},
                'is_hot': {1: True, 2: True, 5: False, ...},
                'recommended_scales': [1, 2, 10],
                'avoid_scales': [60]
            }
        """
        capital = data.get('capital', 100.0)
        hot_scores = {}

        for ts in self.TIMESCALES:
            trades_list = list(self.trades[ts])

            if len(trades_list) < 3:
                hot_scores[ts] = 0.0
                continue

            wins = sum(1 for t in trades_list if t['won'])
            hit_rate = wins / len(trades_list)
            total_pnl = sum(t['pnl'] for t in trades_list)

            # HotScore formula
            hit_rate_excess = hit_rate - 0.5
            confidence = np.sqrt(len(trades_list))
            pnl_factor = 1 + total_pnl / capital

            hot_scores[ts] = hit_rate_excess * confidence * pnl_factor

        # Determine which are hot
        is_hot = {ts: score > self.hot_threshold for ts, score in hot_scores.items()}

        # Recommendations
        sorted_scales = sorted(self.TIMESCALES, key=lambda ts: hot_scores[ts], reverse=True)
        recommended = [ts for ts in sorted_scales if hot_scores[ts] > 0][:3]
        avoid = [ts for ts in sorted_scales if hot_scores[ts] < -0.5]

        return {
            'hot_scores': hot_scores,
            'is_hot': is_hot,
            'recommended_scales': recommended,
            'avoid_scales': avoid,
            'hottest_scale': sorted_scales[0] if sorted_scales else 1
        }


# =============================================================================
# ID 704: SIGNAL FRESHNESS INDEX
# =============================================================================

class SignalFreshnessIndex(BaseFormula):
    """
    ID 704: Measure how fresh/stale a signal is based on age.

    A 1-second signal from 5 seconds ago is worthless.
    A 60-second signal from 5 seconds ago is still valid.

    Formula:
        Freshness = exp(-λ * age_seconds)

    Where λ depends on timescale:
        λ_1s = 1.0 (half-life = 0.69s)
        λ_5s = 0.2 (half-life = 3.5s)
        λ_60s = 0.01 (half-life = 69s)
    """

    FORMULA_ID = 704
    NAME = "SignalFreshnessIndex"
    CATEGORY = "infinite_possibilities"

    # Decay rates per timescale (higher = faster decay)
    DECAY_RATES = {
        1: 1.0,      # 1-second signals decay fast
        2: 0.5,
        5: 0.2,
        10: 0.1,
        30: 0.033,
        60: 0.017,
        120: 0.008,
        300: 0.003   # 5-minute signals decay very slowly
    }

    def __init__(self, min_freshness: float = 0.1, **kwargs):
        """
        Args:
            min_freshness: Below this threshold, signal is considered stale
        """
        super().__init__(**kwargs)
        self.min_freshness = min_freshness
        self.signal_timestamps = {}  # {(timescale, direction): timestamp}

    def _compute(self) -> None:
        """Required by BaseFormula."""
        pass

    def record_signal(self, timescale: int, direction: int, confidence: float):
        """Record when a signal was generated."""
        key = (timescale, direction)
        self.signal_timestamps[key] = {
            'time': time.time(),
            'confidence': confidence
        }

    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate freshness for all tracked signals.

        Args:
            data: {'timescale': 5, 'direction': 1} or {'all': True}

        Returns:
            {
                'freshness': 0.85,
                'is_fresh': True,
                'adjusted_confidence': 0.51,  # original * freshness
                'age_seconds': 1.2
            }
        """
        now = time.time()

        if data.get('all', False):
            # Return freshness for all signals
            all_freshness = {}
            for (ts, direction), info in self.signal_timestamps.items():
                age = now - info['time']
                decay_rate = self.DECAY_RATES.get(ts, 0.1)
                freshness = np.exp(-decay_rate * age)
                all_freshness[(ts, direction)] = {
                    'freshness': freshness,
                    'is_fresh': freshness >= self.min_freshness,
                    'adjusted_confidence': info['confidence'] * freshness,
                    'age_seconds': age
                }
            return {'all_signals': all_freshness}

        # Single signal query
        timescale = data.get('timescale', 1)
        direction = data.get('direction', 1)
        key = (timescale, direction)

        if key not in self.signal_timestamps:
            return {
                'freshness': 0.0,
                'is_fresh': False,
                'adjusted_confidence': 0.0,
                'age_seconds': float('inf')
            }

        info = self.signal_timestamps[key]
        age = now - info['time']
        decay_rate = self.DECAY_RATES.get(timescale, 0.1)
        freshness = np.exp(-decay_rate * age)

        return {
            'freshness': float(freshness),
            'is_fresh': freshness >= self.min_freshness,
            'adjusted_confidence': float(info['confidence'] * freshness),
            'age_seconds': float(age)
        }


# =============================================================================
# ID 705: CROSS-SCALE CORRELATION MONITOR
# =============================================================================

class CrossScaleCorrelationMonitor(BaseFormula):
    """
    ID 705: Monitor correlation between signals across timescales.

    When all timescales suddenly AGREE → Strong signal, increase position
    When timescales suddenly DIVERGE → Confusion, reduce position

    Formula:
        CrossScaleCorr = mean pairwise correlation of signals
        Alarm when |delta_corr| > threshold
    """

    FORMULA_ID = 705
    NAME = "CrossScaleCorrelationMonitor"
    CATEGORY = "infinite_possibilities"

    TIMESCALES = [1, 2, 5, 10, 30, 60]

    def __init__(self, lookback: int = 20, alarm_threshold: float = 0.3, **kwargs):
        """
        Args:
            lookback: History length for correlation calculation
            alarm_threshold: Change in correlation that triggers alarm
        """
        super().__init__(lookback=lookback, **kwargs)
        self.alarm_threshold = alarm_threshold

        # Store signal history per timescale
        self.signal_history = {ts: deque(maxlen=lookback) for ts in self.TIMESCALES}
        self.correlation_history = deque(maxlen=lookback)

    def _compute(self) -> None:
        """Required by BaseFormula."""
        pass

    def record_signals(self, signals: Dict[int, float]):
        """Record signals from all timescales at this moment."""
        for ts in self.TIMESCALES:
            sig = signals.get(ts, 0.0)
            self.signal_history[ts].append(sig)

    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate cross-scale correlation and detect regime changes.

        Returns:
            {
                'cross_correlation': 0.75,
                'correlation_change': 0.15,
                'all_agree': True,
                'regime_alarm': False,
                'position_multiplier': 1.2
            }
        """
        # Need enough history
        min_len = min(len(self.signal_history[ts]) for ts in self.TIMESCALES)
        if min_len < 5:
            return {
                'cross_correlation': 0.0,
                'correlation_change': 0.0,
                'all_agree': False,
                'regime_alarm': False,
                'position_multiplier': 1.0
            }

        # Build signal matrix
        signals_matrix = np.array([
            list(self.signal_history[ts])[-min_len:]
            for ts in self.TIMESCALES
        ])

        # Calculate pairwise correlations
        correlations = []
        n = len(self.TIMESCALES)
        for i in range(n):
            for j in range(i+1, n):
                corr = np.corrcoef(signals_matrix[i], signals_matrix[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        if not correlations:
            return {
                'cross_correlation': 0.0,
                'correlation_change': 0.0,
                'all_agree': False,
                'regime_alarm': False,
                'position_multiplier': 1.0
            }

        current_corr = np.mean(correlations)
        self.correlation_history.append(current_corr)

        # Calculate change
        if len(self.correlation_history) >= 2:
            prev_corr = self.correlation_history[-2]
            corr_change = current_corr - prev_corr
        else:
            corr_change = 0.0

        # Check if all agree (same sign)
        latest_signals = [list(self.signal_history[ts])[-1] for ts in self.TIMESCALES]
        signs = [np.sign(s) for s in latest_signals if s != 0]
        all_agree = len(set(signs)) <= 1 and len(signs) > 0

        # Regime alarm
        regime_alarm = abs(corr_change) > self.alarm_threshold

        # Position multiplier
        if all_agree and current_corr > 0.5:
            position_multiplier = 1.0 + current_corr * 0.5  # Up to 1.5x
        elif regime_alarm:
            position_multiplier = 0.5  # Reduce during confusion
        else:
            position_multiplier = 1.0

        return {
            'cross_correlation': float(current_corr),
            'correlation_change': float(corr_change),
            'all_agree': bool(all_agree),
            'regime_alarm': bool(regime_alarm),
            'position_multiplier': float(position_multiplier)
        }


# =============================================================================
# ID 706: INFINITE POSSIBILITIES CONTROLLER (MASTER)
# =============================================================================

class InfinitePossibilitiesController(BaseFormula):
    """
    ID 706: MASTER controller that combines ALL above formulas.

    Every second:
    1. Generate signals at ALL timescales (1s, 2s, 5s, 10s, 30s, 60s)
    2. Filter by freshness (ID 704)
    3. Check which scales are "hot" (ID 703)
    4. Aggregate probabilities via Bayesian (ID 702)
    5. Allocate capital via Whitrow Kelly (ID 700)
    6. Monitor cross-scale correlation (ID 705)
    7. Execute ALL positive-EV bets simultaneously

    With $100 capital, this captures EVERY mathematical edge.
    """

    FORMULA_ID = 706
    NAME = "InfinitePossibilitiesController"
    CATEGORY = "infinite_possibilities"

    TIMESCALES = [1, 2, 5, 10, 30, 60]

    def __init__(self, capital: float = 100.0, **kwargs):
        """
        Args:
            capital: Total capital available
        """
        super().__init__(**kwargs)
        self.capital = capital

        # Initialize all sub-formulas
        self.kelly = SimultaneousKellyWhitrow(max_total_allocation=0.8)
        self.scale_selector = AdaptiveScaleSelector(lookback=50)
        self.prob_aggregator = BayesianProbabilityAggregator()
        self.hot_detector = HotScaleDetector(window=20)
        self.freshness = SignalFreshnessIndex()
        self.correlation = CrossScaleCorrelationMonitor()

        # State
        self.last_execution = 0
        self.total_trades = 0
        self.total_pnl = 0.0

    def _compute(self) -> None:
        """Required by BaseFormula."""
        pass

    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Master calculation combining all sub-formulas.

        Args:
            data: {
                'signals': {
                    1: {'direction': 1, 'confidence': 0.55, 'edge': 0.05},
                    2: {'direction': 1, 'confidence': 0.60, 'edge': 0.10},
                    5: {'direction': -1, 'confidence': 0.52, 'edge': 0.02},
                    ...
                },
                'capital': 100.0,
                'current_price': 95000.0
            }

        Returns:
            {
                'action': 'MULTI_TRADE',
                'trades': [
                    {'timescale': 1, 'direction': 1, 'size': 10.0, 'confidence': 0.55},
                    {'timescale': 2, 'direction': 1, 'size': 15.0, 'confidence': 0.60},
                ],
                'total_allocation': 45.0,
                'expected_log_growth': 0.05,
                'cross_correlation': 0.75,
                'hot_scales': [1, 2],
                'regime_status': 'STABLE'
            }
        """
        signals = data.get('signals', {})
        capital = data.get('capital', self.capital)

        if not signals:
            return {
                'action': 'HOLD',
                'trades': [],
                'total_allocation': 0.0,
                'expected_log_growth': 0.0,
                'reason': 'NO_SIGNALS'
            }

        # Step 1: Record signals for freshness tracking
        for ts, sig in signals.items():
            self.freshness.record_signal(ts, sig.get('direction', 1), sig.get('confidence', 0.5))

        # Step 2: Check freshness and filter stale signals
        fresh_signals = {}
        for ts, sig in signals.items():
            fresh_result = self.freshness.calculate({'timescale': ts, 'direction': sig.get('direction', 1)})
            if fresh_result['is_fresh']:
                fresh_signals[ts] = {
                    **sig,
                    'adjusted_confidence': fresh_result['adjusted_confidence']
                }

        if not fresh_signals:
            return {
                'action': 'HOLD',
                'trades': [],
                'total_allocation': 0.0,
                'reason': 'ALL_SIGNALS_STALE'
            }

        # Step 3: Get hot scale weights
        scale_result = self.scale_selector.calculate({})
        scale_weights = scale_result['weights']
        hot_scales = scale_result['hot_scales']

        # Step 4: Record signals for correlation monitoring
        signal_values = {ts: sig.get('direction', 0) * sig.get('confidence', 0.5)
                        for ts, sig in fresh_signals.items()}
        self.correlation.record_signals(signal_values)

        # Step 5: Check cross-scale correlation
        corr_result = self.correlation.calculate({})
        position_multiplier = corr_result['position_multiplier']

        # Step 6: Aggregate probabilities
        probabilities = {ts: sig.get('confidence', 0.5) for ts, sig in fresh_signals.items()}
        directions = {ts: sig.get('direction', 1) for ts, sig in fresh_signals.items()}

        agg_result = self.prob_aggregator.calculate({
            'probabilities': probabilities,
            'weights': scale_weights,
            'directions': directions
        })

        # Step 7: Build opportunities for Kelly
        opportunities = []
        for ts, sig in fresh_signals.items():
            edge = sig.get('edge', 0.0)
            if edge > 0:  # Only positive edge
                # Boost hot scales
                hot_boost = 1.2 if ts in hot_scales else 1.0

                opportunities.append({
                    'win_prob': sig.get('adjusted_confidence', sig.get('confidence', 0.5)),
                    'win_amount': 1.5 * hot_boost,  # Risk/reward
                    'loss_amount': 1.0,
                    'timescale': ts,
                    'direction': sig.get('direction', 1),
                    'edge': edge
                })

        if not opportunities:
            return {
                'action': 'HOLD',
                'trades': [],
                'total_allocation': 0.0,
                'reason': 'NO_POSITIVE_EDGE'
            }

        # Step 8: Calculate Kelly allocations
        kelly_result = self.kelly.calculate({
            'opportunities': opportunities,
            'capital': capital
        })

        # Step 9: Apply position multiplier from correlation
        adjusted_allocations = [a * position_multiplier for a in kelly_result['allocations']]
        adjusted_amounts = [a * position_multiplier for a in kelly_result['dollar_amounts']]

        # Step 10: Build trade list
        trades = []
        for i, opp in enumerate(opportunities):
            if adjusted_amounts[i] > 1.0:  # Minimum $1 trade
                trades.append({
                    'timescale': opp['timescale'],
                    'direction': opp['direction'],
                    'size': adjusted_amounts[i],
                    'confidence': opp['win_prob'],
                    'edge': opp['edge'],
                    'allocation_pct': adjusted_allocations[i] * 100
                })

        # Sort by edge (best first)
        trades.sort(key=lambda t: t['edge'], reverse=True)

        # Determine regime status
        if corr_result['regime_alarm']:
            regime_status = 'UNSTABLE'
        elif corr_result['all_agree']:
            regime_status = 'STRONG_AGREEMENT'
        else:
            regime_status = 'STABLE'

        return {
            'action': 'MULTI_TRADE' if trades else 'HOLD',
            'trades': trades,
            'total_allocation': sum(t['size'] for t in trades),
            'total_allocation_pct': sum(t['allocation_pct'] for t in trades),
            'expected_log_growth': kelly_result['expected_log_growth'],
            'cross_correlation': corr_result['cross_correlation'],
            'hot_scales': hot_scales,
            'cold_scales': scale_result['cold_scales'],
            'regime_status': regime_status,
            'combined_probability': agg_result['combined_probability'],
            'combined_direction': agg_result['combined_direction'],
            'position_multiplier': position_multiplier,
            'n_opportunities': len(opportunities),
            'n_trades': len(trades)
        }

    def record_outcome(self, timescale: int, won: bool, pnl: float):
        """Record trade outcome for learning."""
        self.hot_detector.record_trade(timescale, won, pnl)
        self.scale_selector.update_performance(timescale, pnl, won)
        self.total_trades += 1
        self.total_pnl += pnl


# =============================================================================
# REGISTRATION
# =============================================================================

def register_infinite_possibilities():
    """Register all Infinite Possibilities formulas (IDs 700-706)."""
    from .base import FORMULA_REGISTRY

    formulas = [
        SimultaneousKellyWhitrow,      # 700
        AdaptiveScaleSelector,          # 701
        BayesianProbabilityAggregator,  # 702
        HotScaleDetector,               # 703
        SignalFreshnessIndex,           # 704
        CrossScaleCorrelationMonitor,   # 705
        InfinitePossibilitiesController # 706
    ]

    registered = 0
    for formula_class in formulas:
        try:
            instance = formula_class()
            FORMULA_REGISTRY[instance.FORMULA_ID] = instance
            registered += 1
        except Exception as e:
            print(f"[InfinitePossibilities] Error registering {formula_class.__name__}: {e}")

    print(f"[InfinitePossibilities] Registered {registered} formulas (IDs 700-706)")


# Auto-register if imported
if __name__ != "__main__":
    pass  # Will be registered via __init__.py
