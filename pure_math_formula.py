"""
PURE MATH TRADING FORMULA - Export for GitHub
==============================================
Mathematical edge detection and position sizing.
Based on 683 academic formulas distilled to core logic.

Author: Kevin
Repository: pump-fun-trading
"""

import numpy as np
from typing import Dict, Tuple
from collections import deque


class PureMathTradingFormula:
    """
    Core trading formula using pure mathematics.

    EDGE DETECTION:
        Compares TRUE price (calculated) vs MARKET price (observed)
        Edge% = (TRUE - MARKET) / MARKET * 100

    PROBABILITY CALIBRATION (Platt 1999):
        P(win) = 1 / (1 + exp(A*x + B))
        Where A, B are calibrated from historical data

    POSITION SIZING (Kelly 1956):
        f* = (p*b - q) / b
        Where p=win_prob, q=1-p, b=reward_ratio

    BAYESIAN AGGREGATION (Clemen 1989):
        Combined = sigmoid(sum(w * log(p/(1-p))))
        Combines multiple probability estimates
    """

    def __init__(self, capital: float = 100.0):
        self.capital = capital
        self.prices = deque(maxlen=1000)

        # Platt calibration parameters (calibrate from your data)
        self.platt_A = -1.87
        self.platt_B = -0.03

        # Kelly fraction (25% for safety)
        self.kelly_fraction = 0.25

        # Position state
        self.position = 0  # -1=SHORT, 0=FLAT, 1=LONG
        self.entry_price = 0.0
        self.position_size = 0.0

    def calculate_edge(self, true_price: float, market_price: float) -> float:
        """
        Calculate edge percentage.

        Positive = market undervalued (LONG)
        Negative = market overvalued (SHORT)
        """
        if market_price <= 0:
            return 0.0
        return (true_price - market_price) / market_price * 100

    def platt_probability(self, signal: float) -> float:
        """
        Convert signal to calibrated probability using Platt scaling.

        Formula: P = 1 / (1 + exp(A*x + B))

        Reference: Platt (1999) "Probabilistic Outputs for SVMs"
        """
        x = self.platt_A * signal + self.platt_B
        return 1.0 / (1.0 + np.exp(-x))

    def edge_to_probability(self, edge_pct: float) -> float:
        """
        Convert edge percentage to win probability.

        Edge > 0: P(up) > 0.5
        Edge < 0: P(up) < 0.5
        """
        # Clamp edge effect
        effect = min(abs(edge_pct) / 20, 0.45)

        if edge_pct > 0:
            return 0.5 + effect  # Bullish
        else:
            return 0.5 - effect  # Bearish

    def bayesian_aggregate(self, probabilities: list, weights: list = None) -> float:
        """
        Combine multiple probabilities using log-odds Bayesian aggregation.

        Formula: log_odds = sum(w * log(p / (1-p)))
                 combined = 1 / (1 + exp(-log_odds))

        Reference: Clemen (1989) "Combining forecasts"
        """
        if not probabilities:
            return 0.5

        if weights is None:
            weights = [1.0] * len(probabilities)

        # Normalize weights
        total_w = sum(weights)
        norm_weights = [w / total_w for w in weights]

        # Clamp probabilities to avoid log(0)
        probs = [max(0.01, min(0.99, p)) for p in probabilities]

        # Log-odds aggregation
        log_odds = sum(w * np.log(p / (1 - p)) for p, w in zip(probs, norm_weights))

        return 1.0 / (1.0 + np.exp(-log_odds))

    def kelly_size(self, win_prob: float, reward_ratio: float = 1.5) -> float:
        """
        Calculate optimal position size using Kelly Criterion.

        Formula: f* = (p*b - q) / b

        Where:
            p = probability of winning
            q = probability of losing (1-p)
            b = reward/risk ratio

        Reference: Kelly (1956) "A New Interpretation of Information Rate"
        """
        if win_prob <= 0 or win_prob >= 1 or reward_ratio <= 0:
            return 0.0

        p = win_prob
        q = 1 - p
        b = reward_ratio

        kelly = (p * b - q) / b

        # Apply fractional Kelly for safety
        kelly *= self.kelly_fraction

        # Clamp to reasonable range
        return max(0, min(kelly, 0.25))

    def get_signal(self, true_price: float, market_price: float) -> Dict:
        """
        Generate complete trading signal.

        Returns:
            direction: 1 (LONG), -1 (SHORT), 0 (FLAT)
            probability: P(winning trade)
            kelly: optimal position fraction
            size: dollar amount to trade
            edge: edge percentage
        """
        # Store price
        self.prices.append(market_price)

        # Calculate edge
        edge = self.calculate_edge(true_price, market_price)

        # Convert to probability
        p_edge = self.edge_to_probability(edge)

        # Add mean reversion signal if enough data
        probabilities = [p_edge]
        weights = [4.0]  # Edge gets highest weight

        if len(self.prices) > 20:
            prices_arr = np.array(list(self.prices))
            mean_price = np.mean(prices_arr)
            std_price = np.std(prices_arr)

            if std_price > 0:
                z_score = (market_price - mean_price) / std_price
                # Mean reversion: high price = lower P(up)
                p_mr = 0.5 - z_score * 0.1
                p_mr = max(0.1, min(0.9, p_mr))
                probabilities.append(p_mr)
                weights.append(1.5)

        # Bayesian combination
        combined_prob = self.bayesian_aggregate(probabilities, weights)

        # Determine direction
        if combined_prob > 0.55:
            direction = 1  # LONG
            win_prob = combined_prob
        elif combined_prob < 0.45:
            direction = -1  # SHORT
            win_prob = 1 - combined_prob
        else:
            direction = 0  # FLAT
            win_prob = 0.5

        # Kelly sizing
        kelly = self.kelly_size(win_prob) if direction != 0 else 0.0
        size = self.capital * kelly

        return {
            'direction': direction,
            'direction_str': 'LONG' if direction > 0 else ('SHORT' if direction < 0 else 'FLAT'),
            'probability': combined_prob,
            'win_prob': win_prob,
            'kelly': kelly,
            'size': size,
            'edge_pct': edge,
            'true_price': true_price,
            'market_price': market_price
        }

    def should_trade(self, signal: Dict) -> Tuple[bool, str]:
        """
        Determine if we should execute the trade.
        """
        if signal['direction'] == 0:
            return False, f"Probability {signal['probability']:.2f} near 50%"

        if signal['kelly'] < 0.02:
            return False, f"Kelly {signal['kelly']:.2%} too small"

        if abs(signal['edge_pct']) < 0.5:
            return False, f"Edge {signal['edge_pct']:.2f}% too small"

        return True, "All checks passed"


# =============================================================================
# STANDALONE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example usage
    formula = PureMathTradingFormula(capital=100.0)

    # Simulate with sample prices
    true_price = 98000   # What we calculate as TRUE value
    market_price = 87000  # What market is trading at

    signal = formula.get_signal(true_price, market_price)
    should, reason = formula.should_trade(signal)

    print("=" * 50)
    print("PURE MATH TRADING FORMULA")
    print("=" * 50)
    print(f"TRUE Price:    ${true_price:,.2f}")
    print(f"MARKET Price:  ${market_price:,.2f}")
    print(f"Edge:          {signal['edge_pct']:.2f}%")
    print(f"Direction:     {signal['direction_str']}")
    print(f"Probability:   {signal['probability']:.2%}")
    print(f"Kelly:         {signal['kelly']:.2%}")
    print(f"Position Size: ${signal['size']:.2f}")
    print(f"Should Trade:  {should} ({reason})")
    print("=" * 50)
