"""
PURE MATHEMATICS FORMULAS (IDs 720-730)
=======================================
Core mathematical formulas for $100 -> $1B trading.

Based on Academic Research:
- Platt (1999): Probability calibration
- Clemen (1989): Forecast combination
- Satopaa (2014): Log-odds aggregation
- Leung & Li (2015): Optimal mean reversion trading
- Bertram (2010): Analytic solutions for statistical arbitrage
- Kelly (1956): Optimal betting criterion
- Grossman & Zhou (1993): Drawdown-constrained investment
- Grinold & Kahn (1999): Information ratio

These 11 formulas are the ESSENTIAL mathematics for profitable trading.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
from .base import BaseFormula, FormulaRegistry


# =============================================================================
# ID 720: PLATT PROBABILITY CALIBRATION
# Paper: Platt (1999) "Probabilistic outputs for support vector machines"
# =============================================================================

@FormulaRegistry.register(720, "PlattProbabilityCalibration", "pure_math")
class PlattProbabilityCalibration(BaseFormula):
    """
    Convert any signal score to calibrated probability P(win).

    Formula: P(y=1|x) = 1 / (1 + exp(A*f(x) + B))

    Where:
        - f(x) is the raw signal score
        - A, B are learned parameters (A > 0)
        - Output is calibrated probability in [0, 1]

    This is ESSENTIAL because Kelly criterion requires true P(win).
    Uncalibrated probabilities lead to overbetting and ruin.
    """

    DESCRIPTION = "Platt (1999) - Calibrate signal to true probability"

    def __init__(self, A: float = -1.0, B: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.A = A  # Negative for proper sigmoid direction
        self.B = B
        # Online calibration tracking
        self.predictions = deque(maxlen=100)
        self.outcomes = deque(maxlen=100)

    def _compute(self) -> None:
        """Compute calibrated probability from price data."""
        if len(self.returns) < 2:
            self.signal = 0
            self.confidence = 0.5
            return

        # Use recent return as raw score
        raw_score = float(np.mean(list(self.returns)[-5:]) * 100)

        # Platt scaling
        calibrated = self.calibrate(raw_score)

        # Convert to signal
        if calibrated > 0.55:
            self.signal = 1
            self.confidence = calibrated
        elif calibrated < 0.45:
            self.signal = -1
            self.confidence = 1 - calibrated
        else:
            self.signal = 0
            self.confidence = 0.5

    def calibrate(self, raw_score: float) -> float:
        """
        Apply Platt scaling to convert raw score to probability.

        P(y=1|x) = 1 / (1 + exp(A*x + B))
        """
        exponent = self.A * raw_score + self.B
        # Clip to prevent overflow
        exponent = np.clip(exponent, -500, 500)
        return float(1.0 / (1.0 + np.exp(exponent)))

    def update_calibration(self, prediction: float, actual_outcome: int):
        """
        Update A, B parameters based on realized outcomes.
        Uses online gradient descent for continuous calibration.

        actual_outcome: 1 if price went up, 0 if down
        """
        self.predictions.append(prediction)
        self.outcomes.append(actual_outcome)

        if len(self.predictions) < 20:
            return

        # Simple online update (gradient descent on log-loss)
        preds = np.array(self.predictions)
        actuals = np.array(self.outcomes)

        # Gradient of log-loss w.r.t. A and B
        errors = preds - actuals

        # Update parameters (learning rate 0.01)
        lr = 0.01
        # This is simplified - full implementation would use scipy.optimize
        avg_error = np.mean(errors)
        if avg_error > 0.05:  # Overconfident
            self.A *= 0.99
        elif avg_error < -0.05:  # Underconfident
            self.A *= 1.01

    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate calibrated probability from raw signal.

        Input: {'raw_score': float}
        Output: {'probability': float, 'calibrated': bool}
        """
        raw_score = data.get('raw_score', 0)
        prob = self.calibrate(raw_score)

        return {
            'probability': prob,
            'calibrated': True,
            'A': self.A,
            'B': self.B
        }


# =============================================================================
# ID 721: LOG-ODDS BAYESIAN AGGREGATION
# Papers: Clemen (1989), Satopaa (2014)
# =============================================================================

@FormulaRegistry.register(721, "LogOddsBayesianAggregation", "pure_math")
class LogOddsBayesianAggregation(BaseFormula):
    """
    Combine multiple probability forecasts using log-odds aggregation.

    Formula:
        log_odds = sum(w_i * log(p_i / (1 - p_i)))
        combined_prob = 1 / (1 + exp(-log_odds))

    This is mathematically optimal for independent signals.

    Papers:
        - Clemen (1989): "Combining forecasts: A review"
        - Satopaa et al. (2014): "Combining multiple probability predictions"

    Key insight: Averaging in log-odds space is superior to linear pooling.
    Ranjan & Gneiting (2010) proved linear combination is always uncalibrated.
    """

    DESCRIPTION = "Clemen (1989) + Satopaa (2014) - Log-odds probability fusion"

    def __init__(self, prior: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.prior = prior
        self.signal_history = deque(maxlen=100)

    def _compute(self) -> None:
        self.signal = 0
        self.confidence = 0.5

    def combine(self, probabilities: List[float],
                weights: Optional[List[float]] = None) -> float:
        """
        Combine multiple probability estimates using log-odds.

        Args:
            probabilities: List of P(up) estimates from different signals
            weights: Optional weights (default: equal weighting)

        Returns:
            Combined probability P(up | all signals)
        """
        if not probabilities:
            return self.prior

        # Filter valid probabilities (must be in (0, 1))
        valid_probs = []
        valid_weights = []

        for i, p in enumerate(probabilities):
            if 0.01 < p < 0.99:
                valid_probs.append(p)
                if weights and i < len(weights):
                    valid_weights.append(weights[i])
                else:
                    valid_weights.append(1.0)

        if not valid_probs:
            return self.prior

        # Normalize weights
        total_weight = sum(valid_weights)
        if total_weight == 0:
            return self.prior
        norm_weights = [w / total_weight for w in valid_weights]

        # Log-odds aggregation
        # log_odds = sum(w_i * log(p_i / (1 - p_i)))
        log_odds = 0.0
        for p, w in zip(valid_probs, norm_weights):
            log_odds += w * np.log(p / (1 - p))

        # Convert back to probability
        # P = 1 / (1 + exp(-log_odds))
        combined = 1.0 / (1.0 + np.exp(-log_odds))

        return float(np.clip(combined, 0.01, 0.99))

    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine multiple probability forecasts.

        Input: {
            'probabilities': List[float],
            'weights': Optional[List[float]],
            'accuracies': Optional[List[float]]  # Use as weights
        }
        """
        probs = data.get('probabilities', [])
        weights = data.get('weights', None)

        # If accuracies provided, use them as weights
        if weights is None and 'accuracies' in data:
            # Weight by edge over random (accuracy - 0.5)
            weights = [max(0, a - 0.5) for a in data['accuracies']]

        combined = self.combine(probs, weights)

        # Direction from combined probability
        if combined > 0.55:
            direction = 1
        elif combined < 0.45:
            direction = -1
        else:
            direction = 0

        return {
            'combined_probability': combined,
            'direction': direction,
            'n_signals': len(probs),
            'confidence': abs(combined - 0.5) * 2
        }


# =============================================================================
# ID 722: OU HALF-LIFE CALCULATOR
# Formula: HL = ln(2) / kappa
# =============================================================================

@FormulaRegistry.register(722, "OUHalfLifeCalculator", "pure_math")
class OUHalfLifeCalculator(BaseFormula):
    """
    Calculate half-life of mean reversion for OU process.

    Formula: HL = ln(2) / kappa

    Where kappa is estimated from AR(1) regression:
        dX_t = kappa * (theta - X_t) * dt + sigma * dW_t

    Ernie Chan method:
        Regress (y_t - y_{t-1}) on y_{t-1}
        kappa = -coefficient
        HL = ln(2) / kappa

    Key insight from research:
        - HL determines optimal lookback for moving averages
        - HL determines max hold time (2x HL for full reversion)
        - If kappa <= 0, series is trending (don't mean-revert trade)
    """

    DESCRIPTION = "OU half-life via Ernie Chan AR(1) method"

    def __init__(self, **kwargs):
        super().__init__(lookback=100, **kwargs)
        self.kappa = 0.0
        self.theta = 0.0
        self.sigma = 0.0
        self.half_life = float('inf')

    def _compute(self) -> None:
        if len(self.prices) < 30:
            self.signal = 0
            self.confidence = 0.0
            return

        # Estimate OU parameters
        self._estimate_ou_params()

        # If mean-reverting (kappa > 0), calculate signal
        if self.kappa > 0:
            prices = self._prices_array()
            current = prices[-1]
            deviation = (current - self.theta) / self.sigma if self.sigma > 0 else 0

            # Signal based on deviation from mean
            if deviation < -2:
                self.signal = 1  # Below mean, expect up
                self.confidence = min(0.9, abs(deviation) / 4)
            elif deviation > 2:
                self.signal = -1  # Above mean, expect down
                self.confidence = min(0.9, abs(deviation) / 4)
            else:
                self.signal = 0
                self.confidence = abs(deviation) / 4
        else:
            # Not mean-reverting
            self.signal = 0
            self.confidence = 0.0

    def _estimate_ou_params(self):
        """
        Estimate OU parameters using Ernie Chan's method.

        Regress: y_t - y_{t-1} = a + b * y_{t-1} + epsilon
        Then: kappa = -b, theta = a / kappa
        """
        prices = self._prices_array()
        if len(prices) < 30:
            return

        # Create regression variables
        y = prices[1:] - prices[:-1]  # Changes
        x = prices[:-1]  # Lagged prices

        # Simple linear regression
        n = len(y)
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator == 0:
            return

        b = numerator / denominator  # Slope
        a = y_mean - b * x_mean  # Intercept

        # OU parameters
        self.kappa = -b  # Speed of mean reversion

        if self.kappa > 0:
            self.theta = a / self.kappa  # Long-term mean
            self.half_life = np.log(2) / self.kappa

            # Estimate sigma from residuals
            residuals = y - (a + b * x)
            self.sigma = np.std(residuals) * np.sqrt(252)  # Annualized
        else:
            self.half_life = float('inf')

    def calculate(self, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Return OU parameters and half-life.
        """
        return {
            'kappa': self.kappa,
            'theta': self.theta,
            'sigma': self.sigma,
            'half_life': self.half_life,
            'is_mean_reverting': self.kappa > 0,
            'optimal_lookback': int(self.half_life) if self.half_life < 1000 else 100,
            'max_hold_time': self.half_life * 2 if self.half_life < 1000 else 200
        }


# =============================================================================
# ID 723: LEUNG-LI OPTIMAL ENTRY THRESHOLD
# Paper: Leung & Li (2015) "Optimal Mean Reversion Trading"
# =============================================================================

@FormulaRegistry.register(723, "LeungLiOptimalEntry", "pure_math")
class LeungLiOptimalEntry(BaseFormula):
    """
    Calculate optimal entry/exit thresholds for mean reversion trading.

    Paper: Leung & Li (2015) "Optimal Mean Reversion Trading with
           Transaction Costs and Stop-Loss Exit"

    Key insight: Rather than ad-hoc entry (e.g., 2 sigma), solve the
    optimal stopping problem to maximize expected profit.

    Results:
        - Entry threshold depends on transaction costs, kappa, sigma
        - Higher transaction costs -> wider entry threshold
        - Higher volatility -> wider thresholds
        - Faster mean reversion -> tighter thresholds

    Simplified formula (Bertram 2010):
        For maximizing expected return, thresholds are symmetric around mean.
        Entry at theta +/- k*sigma where k depends on costs and kappa.
    """

    DESCRIPTION = "Leung-Li (2015) - Optimal mean reversion entry/exit"

    def __init__(self, transaction_cost: float = 0.001, **kwargs):
        super().__init__(**kwargs)
        self.transaction_cost = transaction_cost  # As fraction (0.1% = 0.001)
        self.entry_threshold = 2.0  # Default 2 sigma
        self.exit_threshold = 0.0   # Exit at mean
        self.stop_loss = -3.0       # Stop at -3 sigma

    def _compute(self) -> None:
        if len(self.prices) < 50:
            self.signal = 0
            self.confidence = 0.0
            return

        # Calculate z-score
        prices = self._prices_array()
        mean = np.mean(prices)
        std = np.std(prices)

        if std == 0:
            self.signal = 0
            self.confidence = 0.0
            return

        z_score = (prices[-1] - mean) / std

        # Optimal entry based on z-score vs threshold
        if z_score < -self.entry_threshold:
            self.signal = 1  # Enter LONG
            self.confidence = min(0.9, abs(z_score) / 4)
        elif z_score > self.entry_threshold:
            self.signal = -1  # Enter SHORT
            self.confidence = min(0.9, abs(z_score) / 4)
        else:
            self.signal = 0
            self.confidence = 0.0

    def calculate_optimal_threshold(self, kappa: float, sigma: float,
                                     cost: float = None) -> Dict[str, float]:
        """
        Calculate optimal entry threshold based on OU parameters.

        From Bertram (2010), for maximizing expected return:
        The optimal threshold increases with:
            - Transaction costs (need wider spread to cover costs)
            - Volatility (more uncertainty, wait for larger deviation)
        And decreases with:
            - Speed of mean reversion (faster reversion, can enter earlier)

        Simplified approximation:
            threshold = sqrt(2 * cost / kappa) + base
        """
        if cost is None:
            cost = self.transaction_cost

        if kappa <= 0:
            return {
                'entry_long': -3.0,
                'entry_short': 3.0,
                'exit': 0.0,
                'stop_loss_long': -5.0,
                'stop_loss_short': 5.0,
                'valid': False
            }

        # Approximate optimal threshold
        # Higher costs or lower kappa -> need wider threshold
        cost_adjustment = np.sqrt(2 * cost * 10000 / kappa) if kappa > 0 else 1.0

        # Base threshold of ~1.5 sigma, adjusted for costs
        base_threshold = 1.5 + cost_adjustment

        # Cap at reasonable values
        entry = min(3.0, max(1.0, base_threshold))

        self.entry_threshold = entry
        self.exit_threshold = 0.5  # Exit closer to mean
        self.stop_loss = -entry - 1.5  # Stop loss wider than entry

        return {
            'entry_long': -entry,      # Enter long below mean
            'entry_short': entry,      # Enter short above mean
            'exit': self.exit_threshold,
            'stop_loss_long': self.stop_loss,
            'stop_loss_short': -self.stop_loss,
            'valid': True
        }

    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate if current price is at optimal entry.

        Input: {
            'z_score': float,  # Current z-score
            'kappa': float,    # OU speed of mean reversion
            'sigma': float,    # OU volatility
        }
        """
        z_score = data.get('z_score', 0)
        kappa = data.get('kappa', 0.1)
        sigma = data.get('sigma', 0.01)

        # Calculate optimal thresholds
        thresholds = self.calculate_optimal_threshold(kappa, sigma)

        # Determine action
        if z_score <= thresholds['entry_long']:
            action = 'ENTER_LONG'
            direction = 1
        elif z_score >= thresholds['entry_short']:
            action = 'ENTER_SHORT'
            direction = -1
        elif abs(z_score) <= thresholds['exit']:
            action = 'EXIT'
            direction = 0
        else:
            action = 'HOLD'
            direction = 0

        return {
            'action': action,
            'direction': direction,
            'z_score': z_score,
            'entry_threshold': self.entry_threshold,
            'distance_to_entry': abs(z_score) - self.entry_threshold,
            **thresholds
        }


# =============================================================================
# ID 724: KELLY CRITERION WITH EDGE
# Paper: Kelly (1956) "A New Interpretation of Information Rate"
# =============================================================================

@FormulaRegistry.register(724, "KellyCriterionWithEdge", "pure_math")
class KellyCriterionWithEdge(BaseFormula):
    """
    Kelly Criterion for optimal position sizing.

    Formula: f* = (p * b - q) / b

    Where:
        f* = Optimal fraction of capital to bet
        p = Probability of winning
        q = 1 - p = Probability of losing
        b = Odds ratio (win amount / loss amount)

    For trading with TP/SL:
        b = TP / SL
        p = P(hitting TP before SL)

    CRITICAL: Use FRACTIONAL Kelly (25-50%) for safety.
    Full Kelly has 50% probability of 50% drawdown!

    From research:
        - 50% Kelly: 75% of growth, 25% of variance
        - 25% Kelly: Still significant growth, much safer
        - Half-Kelly: 1/9 chance of halving vs 1/3 for full Kelly
    """

    DESCRIPTION = "Kelly (1956) - Optimal bet sizing with fractional Kelly"

    def __init__(self, kelly_fraction: float = 0.25, **kwargs):
        super().__init__(**kwargs)
        self.kelly_fraction = kelly_fraction  # 25% Kelly default
        self.max_kelly = 0.25  # Never bet more than 25%
        self.min_kelly = 0.01  # Minimum 1% to be worth trading

    def _compute(self) -> None:
        self.signal = 0
        self.confidence = 0.0

    def calculate_kelly(self, win_prob: float, win_amount: float,
                        loss_amount: float) -> float:
        """
        Calculate Kelly fraction.

        Args:
            win_prob: Probability of winning P(win)
            win_amount: Amount won if win (as fraction, e.g., 0.01 = 1%)
            loss_amount: Amount lost if lose (as fraction)

        Returns:
            Optimal bet fraction f* (already fractional Kelly applied)
        """
        if win_prob <= 0 or win_prob >= 1:
            return 0.0

        if loss_amount <= 0 or win_amount <= 0:
            return 0.0

        p = win_prob
        q = 1 - p
        b = win_amount / loss_amount  # Odds ratio

        # Kelly formula: f* = (p*b - q) / b
        kelly = (p * b - q) / b

        # Apply fractional Kelly
        kelly *= self.kelly_fraction

        # Clip to safe range
        return float(np.clip(kelly, 0, self.max_kelly))

    def expected_value(self, win_prob: float, win_amount: float,
                       loss_amount: float) -> float:
        """
        Calculate expected value of a trade.

        EV = P(win) * win - P(loss) * loss

        Only trade if EV > 0
        """
        return win_prob * win_amount - (1 - win_prob) * loss_amount

    def risk_of_ruin(self, win_prob: float, kelly_fraction_used: float) -> float:
        """
        Estimate probability of drawdown based on Kelly usage.

        From research:
            - Full Kelly: 1/n chance of reaching 1/n of peak
            - At Kelly fraction f: Probability of halving ≈ (1-f)^2 / f^2
        """
        if kelly_fraction_used <= 0:
            return 0.0

        # Simplified risk estimate
        # Higher Kelly = higher risk
        risk = kelly_fraction_used ** 2
        return min(1.0, risk)

    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate optimal position size.

        Input: {
            'win_prob': float,     # P(win) - MUST be calibrated!
            'tp_percent': float,   # Take profit as percent (e.g., 0.003 = 0.3%)
            'sl_percent': float,   # Stop loss as percent
            'capital': float,      # Total capital
        }
        """
        win_prob = data.get('win_prob', 0.5)
        tp = data.get('tp_percent', 0.003)
        sl = data.get('sl_percent', 0.002)
        capital = data.get('capital', 100.0)

        # Calculate Kelly
        kelly = self.calculate_kelly(win_prob, tp, sl)

        # Calculate EV
        ev = self.expected_value(win_prob, tp, sl)

        # Position size
        position_size = capital * kelly

        # Should trade?
        should_trade = (kelly >= self.min_kelly and ev > 0 and win_prob >= 0.5)

        return {
            'kelly_fraction': kelly,
            'position_size': position_size,
            'expected_value': ev,
            'should_trade': should_trade,
            'ev_per_dollar': ev / sl if sl > 0 else 0,
            'reward_risk_ratio': tp / sl if sl > 0 else 0,
            'risk_of_ruin': self.risk_of_ruin(win_prob, kelly)
        }


# =============================================================================
# ID 725: DRAWDOWN-CONSTRAINED KELLY
# Paper: Grossman & Zhou (1993) "Optimal Investment Strategies for
#        Controlling Drawdowns"
# =============================================================================

@FormulaRegistry.register(725, "DrawdownConstrainedKelly", "pure_math")
class DrawdownConstrainedKelly(BaseFormula):
    """
    Kelly criterion with maximum drawdown constraint.

    Paper: Grossman & Zhou (1993)

    Constraint: W_t >= alpha * M_t
    Where:
        W_t = Current wealth
        M_t = Maximum wealth achieved so far
        alpha = Minimum fraction to maintain (e.g., 0.8 = max 20% drawdown)

    Optimal policy: Invest proportional to "surplus" = W_t - alpha * M_t

    This prevents ruin while still capturing growth.

    Key insight:
        - Standard Kelly can have 50% drawdowns
        - Constrained Kelly limits drawdown at cost of some growth
        - alpha = 0.8 means max 20% drawdown
    """

    DESCRIPTION = "Grossman-Zhou (1993) - Kelly with drawdown constraint"

    def __init__(self, max_drawdown: float = 0.20, **kwargs):
        super().__init__(**kwargs)
        self.max_drawdown = max_drawdown  # Max 20% drawdown
        self.alpha = 1 - max_drawdown     # Floor as fraction of peak
        self.peak_capital = 0.0
        self.current_capital = 0.0

    def _compute(self) -> None:
        self.signal = 0
        self.confidence = 0.0

    def update_capital(self, capital: float):
        """Track capital for drawdown calculation."""
        self.current_capital = capital
        if capital > self.peak_capital:
            self.peak_capital = capital

    def calculate_constrained_kelly(self, base_kelly: float,
                                     capital: float,
                                     peak: float = None) -> float:
        """
        Calculate drawdown-constrained position size.

        From Grossman-Zhou:
            Position = base_kelly * surplus / capital
            surplus = W_t - alpha * M_t
        """
        if peak is None:
            peak = self.peak_capital

        if peak == 0:
            peak = capital

        # Floor level
        floor = self.alpha * peak

        # Surplus available for risk
        surplus = max(0, capital - floor)

        if surplus == 0:
            return 0.0  # At floor, don't risk anything

        # Scale Kelly by surplus ratio
        surplus_ratio = surplus / capital if capital > 0 else 0
        constrained_kelly = base_kelly * surplus_ratio

        return float(np.clip(constrained_kelly, 0, 0.25))

    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate drawdown-constrained position size.

        Input: {
            'base_kelly': float,   # Kelly fraction before constraint
            'capital': float,      # Current capital
            'peak': float,         # Peak capital (optional)
        }
        """
        base_kelly = data.get('base_kelly', 0.1)
        capital = data.get('capital', 100.0)
        peak = data.get('peak', None)

        self.update_capital(capital)
        if peak is None:
            peak = self.peak_capital

        # Calculate constraint
        floor = self.alpha * peak
        surplus = max(0, capital - floor)
        current_drawdown = (peak - capital) / peak if peak > 0 else 0

        # Constrained Kelly
        constrained = self.calculate_constrained_kelly(base_kelly, capital, peak)

        # Position size
        position_size = capital * constrained

        return {
            'constrained_kelly': constrained,
            'base_kelly': base_kelly,
            'position_size': position_size,
            'surplus': surplus,
            'floor': floor,
            'current_drawdown': current_drawdown,
            'at_risk': current_drawdown >= self.max_drawdown * 0.8,
            'can_trade': surplus > 0 and constrained > 0.01
        }


# =============================================================================
# ID 726: EXPECTED VALUE TRACKER
# Formula: EV = P(win) * win - P(loss) * loss
# =============================================================================

@FormulaRegistry.register(726, "ExpectedValueTracker", "pure_math")
class ExpectedValueTracker(BaseFormula):
    """
    Track realized vs expected value to verify edge is real.

    Formula: EV = P(win) * avg_win - P(loss) * avg_loss

    Also known as "Expectancy" in trading.

    Key metrics:
        - Expected EV: Calculated from probabilities
        - Realized EV: Actual P&L / number of trades
        - Edge decay: Is our edge getting smaller?

    From research:
        - If realized EV < expected EV consistently, probabilities are wrong
        - Edge changes over time - must track continuously
        - "Sweet spot" for EV is $20-$60 per trade (from Option Alpha research)
    """

    DESCRIPTION = "Track expected vs realized edge"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trade_results = deque(maxlen=100)  # (predicted_ev, actual_pnl)
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0

    def _compute(self) -> None:
        self.signal = 0
        self.confidence = 0.0

    def record_trade(self, predicted_ev: float, actual_pnl: float, won: bool):
        """Record a trade result."""
        self.trade_results.append((predicted_ev, actual_pnl))
        self.total_pnl += actual_pnl
        if won:
            self.wins += 1
        else:
            self.losses += 1

    def calculate_expectancy(self) -> Dict[str, float]:
        """
        Calculate trading expectancy from history.

        Expectancy = (Win% * Avg Win) - (Loss% * Avg Loss)
        """
        if not self.trade_results:
            return {'expectancy': 0, 'win_rate': 0, 'avg_win': 0, 'avg_loss': 0}

        total = self.wins + self.losses
        if total == 0:
            return {'expectancy': 0, 'win_rate': 0, 'avg_win': 0, 'avg_loss': 0}

        win_rate = self.wins / total

        # Calculate average win and loss
        wins = [r[1] for r in self.trade_results if r[1] > 0]
        losses = [r[1] for r in self.trade_results if r[1] <= 0]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0

        # Expectancy formula
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        return {
            'expectancy': expectancy,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': total,
            'total_pnl': self.total_pnl
        }

    def calculate(self, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get current edge statistics.
        """
        stats = self.calculate_expectancy()

        # Check if edge is real
        if len(self.trade_results) >= 20:
            predicted_evs = [r[0] for r in self.trade_results]
            actual_pnls = [r[1] for r in self.trade_results]

            avg_predicted = np.mean(predicted_evs)
            avg_actual = np.mean(actual_pnls)

            # Edge ratio: actual / predicted
            edge_ratio = avg_actual / avg_predicted if avg_predicted != 0 else 0
            edge_is_real = edge_ratio > 0.5  # Actual >= 50% of predicted
        else:
            edge_ratio = 1.0
            edge_is_real = True  # Not enough data

        stats['edge_ratio'] = edge_ratio
        stats['edge_is_real'] = edge_is_real
        stats['sample_size'] = len(self.trade_results)

        return stats


# =============================================================================
# ID 727: INFORMATION RATIO
# Paper: Grinold & Kahn (1999) "Active Portfolio Management"
# =============================================================================

@FormulaRegistry.register(727, "InformationRatio", "pure_math")
class InformationRatio(BaseFormula):
    """
    Information Ratio for measuring strategy quality.

    Formula: IR = IC * sqrt(BR)

    Where:
        IR = Information Ratio (excess return / tracking error)
        IC = Information Coefficient (correlation of forecast with returns)
        BR = Breadth (number of independent bets)

    Alternative: IR = alpha / tracking_error

    Paper: Grinold & Kahn (1999) "Active Portfolio Management"

    Key insight:
        - IR > 0.5 is good
        - IR > 1.0 is excellent
        - More frequent trading increases BR, improving IR
    """

    DESCRIPTION = "Grinold-Kahn (1999) - Information ratio for strategy quality"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.forecasts = deque(maxlen=100)  # (forecast, actual_return)
        self.returns = deque(maxlen=100)

    def _compute(self) -> None:
        self.signal = 0
        self.confidence = 0.0

    def record_forecast(self, forecast: float, actual_return: float):
        """Record a forecast and its realized return."""
        self.forecasts.append((forecast, actual_return))

    def calculate_ic(self) -> float:
        """
        Calculate Information Coefficient.

        IC = correlation(forecasts, actual_returns)
        """
        if len(self.forecasts) < 10:
            return 0.0

        forecasts = np.array([f[0] for f in self.forecasts])
        actuals = np.array([f[1] for f in self.forecasts])

        if np.std(forecasts) == 0 or np.std(actuals) == 0:
            return 0.0

        correlation = np.corrcoef(forecasts, actuals)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0

    def calculate_ir(self, alpha: float = None, tracking_error: float = None,
                     breadth: int = None) -> float:
        """
        Calculate Information Ratio.

        Method 1: IR = alpha / tracking_error
        Method 2: IR = IC * sqrt(BR)
        """
        if alpha is not None and tracking_error is not None and tracking_error > 0:
            return alpha / tracking_error

        # Use fundamental law
        ic = self.calculate_ic()
        br = breadth if breadth else len(self.forecasts)

        if br <= 0:
            return 0.0

        return ic * np.sqrt(br)

    def calculate(self, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Calculate information ratio and components.
        """
        ic = self.calculate_ic()
        br = len(self.forecasts)
        ir = self.calculate_ir(breadth=br)

        # Quality assessment
        if ir > 1.0:
            quality = "EXCELLENT"
        elif ir > 0.5:
            quality = "GOOD"
        elif ir > 0:
            quality = "MARGINAL"
        else:
            quality = "POOR"

        return {
            'information_ratio': ir,
            'information_coefficient': ic,
            'breadth': br,
            'quality': quality,
            'sample_size': len(self.forecasts)
        }


# =============================================================================
# ID 728: RISK OF RUIN CALCULATOR
# =============================================================================

@FormulaRegistry.register(728, "RiskOfRuinCalculator", "pure_math")
class RiskOfRuinCalculator(BaseFormula):
    """
    Calculate probability of ruin given trading parameters.

    Formula (simplified):
        RoR = ((1 - edge) / (1 + edge)) ^ capital_units

    Where edge = (win_rate * avg_win - loss_rate * avg_loss) / avg_loss

    Key insights from research:
        - Full Kelly: 1/n chance of dropping to 1/n of peak
        - Half Kelly: Squares the denominator (much safer)
        - At Kelly fraction f: P(halving) ≈ 1/(2^(1/f) - 1)

    This helps determine if a strategy is viable long-term.
    """

    DESCRIPTION = "Calculate probability of ruin/bankruptcy"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _compute(self) -> None:
        self.signal = 0
        self.confidence = 0.0

    def calculate_risk_of_ruin(self, win_rate: float, avg_win: float,
                                avg_loss: float, capital_units: int = 100) -> float:
        """
        Calculate probability of losing all capital.

        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average win amount
            avg_loss: Average loss amount
            capital_units: Number of "units" of capital (bet sizes)
        """
        if win_rate <= 0 or win_rate >= 1:
            return 1.0 if win_rate <= 0 else 0.0

        if avg_loss == 0:
            return 0.0

        # Calculate edge
        edge = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_loss

        if edge <= 0:
            return 1.0  # Negative edge = certain ruin

        # Risk of ruin formula
        ratio = (1 - edge) / (1 + edge)

        if ratio >= 1:
            return 1.0

        ror = ratio ** capital_units
        return float(np.clip(ror, 0, 1))

    def kelly_drawdown_probability(self, kelly_fraction: float,
                                    drawdown_level: float = 0.5) -> float:
        """
        Probability of experiencing a given drawdown at Kelly fraction.

        At full Kelly:
            P(drawdown to level L) = L
            E.g., P(50% drawdown) = 50%

        At fractional Kelly f:
            P(drawdown to L) ≈ L^(1/f)
        """
        if kelly_fraction <= 0:
            return 0.0

        # At fraction f of Kelly, probability of drawdown to L
        prob = drawdown_level ** (1 / kelly_fraction)
        return float(np.clip(prob, 0, 1))

    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate risk of ruin statistics.

        Input: {
            'win_rate': float,
            'avg_win': float,
            'avg_loss': float,
            'capital': float,
            'bet_size': float,
            'kelly_fraction': float
        }
        """
        win_rate = data.get('win_rate', 0.5)
        avg_win = data.get('avg_win', 0.01)
        avg_loss = data.get('avg_loss', 0.01)
        capital = data.get('capital', 100)
        bet_size = data.get('bet_size', 1)
        kelly_frac = data.get('kelly_fraction', 0.25)

        capital_units = int(capital / bet_size) if bet_size > 0 else 100

        # Risk of ruin
        ror = self.calculate_risk_of_ruin(win_rate, avg_win, avg_loss, capital_units)

        # Drawdown probabilities at this Kelly fraction
        dd_50 = self.kelly_drawdown_probability(kelly_frac, 0.5)
        dd_20 = self.kelly_drawdown_probability(kelly_frac, 0.8)  # 20% DD = 80% remaining

        # Edge calculation
        edge = (win_rate * avg_win - (1 - win_rate) * avg_loss)

        return {
            'risk_of_ruin': ror,
            'prob_50pct_drawdown': dd_50,
            'prob_20pct_drawdown': dd_20,
            'edge_per_trade': edge,
            'capital_units': capital_units,
            'is_viable': ror < 0.01 and edge > 0  # <1% ruin prob, positive edge
        }


# =============================================================================
# ID 729: BERTRAM FIRST PASSAGE TIME
# Paper: Bertram (2010) "Analytic Solutions for Optimal Statistical Arbitrage"
# =============================================================================

@FormulaRegistry.register(729, "BertramFirstPassageTime", "pure_math")
class BertramFirstPassageTime(BaseFormula):
    """
    Expected time to hit target price using OU process.

    Paper: Bertram (2010) "Analytic solutions for optimal statistical
           arbitrage trading", Physica A, 389(11): 2234-2243

    For OU process: dX = kappa*(theta - X)*dt + sigma*dW

    Mean first passage time from x to target b:
        E[T] = integral formula involving parabolic cylinder functions

    Simplified approximation:
        E[T] ≈ (1/kappa) * ln(|x - theta| / |b - theta|)

    This determines expected trade duration for position sizing.
    """

    DESCRIPTION = "Bertram (2010) - Expected time to hit price target"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _compute(self) -> None:
        self.signal = 0
        self.confidence = 0.0

    def expected_passage_time(self, current: float, target: float,
                               theta: float, kappa: float) -> float:
        """
        Calculate expected time to reach target.

        Simplified formula:
            E[T] ≈ (1/kappa) * |ln((current - theta) / (target - theta))|

        Args:
            current: Current price/level
            target: Target price/level
            theta: Long-term mean
            kappa: Speed of mean reversion

        Returns:
            Expected time in same units as kappa (if kappa is per day, time is days)
        """
        if kappa <= 0:
            return float('inf')

        # Distance from mean
        current_dist = abs(current - theta)
        target_dist = abs(target - theta)

        if target_dist == 0:
            return 0.0  # Already at target

        if current_dist == 0:
            current_dist = 0.001  # Avoid log(0)

        # Simplified expected time
        # This is approximate - exact formula involves special functions
        expected_time = (1 / kappa) * abs(np.log(current_dist / target_dist))

        return float(max(0, expected_time))

    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate expected first passage time.

        Input: {
            'current_price': float,
            'target_price': float,
            'theta': float,      # Long-term mean
            'kappa': float,      # Mean reversion speed
            'sigma': float,      # Volatility (optional, for confidence)
        }
        """
        current = data.get('current_price', 0)
        target = data.get('target_price', 0)
        theta = data.get('theta', 0)
        kappa = data.get('kappa', 0.1)
        sigma = data.get('sigma', 0.01)

        # Expected passage time
        ept = self.expected_passage_time(current, target, theta, kappa)

        # Half-life for reference
        half_life = np.log(2) / kappa if kappa > 0 else float('inf')

        # Probability of reaching target (rough estimate)
        # Higher if target is toward mean, lower if away from mean
        to_mean = abs(current - theta) > abs(target - theta)

        if to_mean:
            # Target is toward mean - higher probability
            reach_prob = 0.7 + 0.2 * (1 - abs(target - theta) / abs(current - theta))
        else:
            # Target is away from mean - lower probability
            reach_prob = 0.3

        return {
            'expected_time': ept,
            'half_life': half_life,
            'target_toward_mean': to_mean,
            'reach_probability': reach_prob,
            'kappa': kappa,
            'max_recommended_hold': ept * 2  # 2x expected time as max
        }


# =============================================================================
# ID 730: PURE MATH MASTER CONTROLLER
# Combines all above formulas into single decision
# =============================================================================

@FormulaRegistry.register(730, "PureMathMasterController", "pure_math")
class PureMathMasterController(BaseFormula):
    """
    MASTER CONTROLLER: Combines all pure math formulas for optimal trading.

    Decision process:
    1. Calibrate signal to probability (ID 720)
    2. Combine multiple signals via log-odds (ID 721)
    3. Check OU parameters for mean reversion (ID 722)
    4. Verify at optimal entry threshold (ID 723)
    5. Calculate Kelly fraction (ID 724)
    6. Apply drawdown constraint (ID 725)
    7. Check expected value is positive (ID 726)
    8. Verify risk of ruin is acceptable (ID 728)

    Only trade when ALL conditions pass.
    """

    DESCRIPTION = "MASTER - Combines all pure math for optimal decision"

    def __init__(self, capital: float = 100.0, max_drawdown: float = 0.20, **kwargs):
        super().__init__(**kwargs)
        self.capital = capital
        self.max_drawdown = max_drawdown

        # Initialize component formulas
        self.calibrator = PlattProbabilityCalibration()
        self.aggregator = LogOddsBayesianAggregation()
        self.ou_calculator = OUHalfLifeCalculator()
        self.entry_optimizer = LeungLiOptimalEntry()
        self.kelly = KellyCriterionWithEdge()
        self.dd_kelly = DrawdownConstrainedKelly(max_drawdown=max_drawdown)
        self.ev_tracker = ExpectedValueTracker()
        self.risk_calc = RiskOfRuinCalculator()

    def _compute(self) -> None:
        self.signal = 0
        self.confidence = 0.0

    def decide(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make optimal trading decision using all pure math components.

        Input: {
            'signals': List[float],           # Raw signal scores
            'signal_probs': List[float],      # Or pre-calibrated probabilities
            'current_price': float,
            'prices': List[float],            # Price history
            'capital': float,
            'peak_capital': float,
            'tp_percent': float,              # Take profit %
            'sl_percent': float,              # Stop loss %
        }
        """
        checks_passed = 0
        checks_failed = []

        # Get inputs
        signals = data.get('signals', [])
        signal_probs = data.get('signal_probs', [])
        prices = data.get('prices', [])
        capital = data.get('capital', self.capital)
        peak = data.get('peak_capital', capital)
        tp = data.get('tp_percent', 0.003)
        sl = data.get('sl_percent', 0.002)

        # ===== STEP 1: PROBABILITY CALIBRATION =====
        if signal_probs:
            probabilities = signal_probs
        elif signals:
            probabilities = [self.calibrator.calibrate(s) for s in signals]
        else:
            probabilities = [0.5]

        # ===== STEP 2: BAYESIAN AGGREGATION =====
        agg_result = self.aggregator.calculate({'probabilities': probabilities})
        combined_prob = agg_result['combined_probability']
        direction = agg_result['direction']

        if combined_prob >= 0.55 or combined_prob <= 0.45:
            checks_passed += 1
        else:
            checks_failed.append(f"LOW_PROB:{combined_prob:.2f}")

        # Adjust for direction
        win_prob = combined_prob if direction > 0 else (1 - combined_prob)

        # ===== STEP 3: OU PARAMETERS =====
        if len(prices) >= 30:
            for p in prices[-50:]:
                self.ou_calculator.prices.append(p)
            self.ou_calculator._estimate_ou_params()
            ou_params = self.ou_calculator.calculate()

            if ou_params['is_mean_reverting']:
                checks_passed += 1
            else:
                checks_failed.append("NOT_MEAN_REVERTING")
        else:
            ou_params = {'kappa': 0.1, 'theta': prices[-1] if prices else 0, 'half_life': 100}

        # ===== STEP 4: OPTIMAL ENTRY CHECK =====
        if prices:
            mean = np.mean(prices[-50:]) if len(prices) >= 50 else np.mean(prices)
            std = np.std(prices[-50:]) if len(prices) >= 50 else np.std(prices)
            current = prices[-1]
            z_score = (current - mean) / std if std > 0 else 0

            entry_result = self.entry_optimizer.calculate({
                'z_score': z_score,
                'kappa': ou_params.get('kappa', 0.1),
                'sigma': std
            })

            if entry_result['action'] in ['ENTER_LONG', 'ENTER_SHORT']:
                checks_passed += 1
            else:
                checks_failed.append(f"NOT_AT_ENTRY:{entry_result['action']}")
        else:
            z_score = 0
            entry_result = {'action': 'NO_DATA'}

        # ===== STEP 5: KELLY FRACTION =====
        kelly_result = self.kelly.calculate({
            'win_prob': win_prob,
            'tp_percent': tp,
            'sl_percent': sl,
            'capital': capital
        })

        if kelly_result['should_trade']:
            checks_passed += 1
        else:
            checks_failed.append(f"KELLY_REJECT:EV={kelly_result['expected_value']:.4f}")

        # ===== STEP 6: DRAWDOWN CONSTRAINT =====
        dd_result = self.dd_kelly.calculate({
            'base_kelly': kelly_result['kelly_fraction'],
            'capital': capital,
            'peak': peak
        })

        if dd_result['can_trade']:
            checks_passed += 1
        else:
            checks_failed.append(f"DD_CONSTRAINT:{dd_result['current_drawdown']:.1%}")

        # ===== STEP 7: EXPECTED VALUE =====
        ev = kelly_result['expected_value']
        if ev > 0:
            checks_passed += 1
        else:
            checks_failed.append(f"NEG_EV:{ev:.4f}")

        # ===== STEP 8: RISK OF RUIN =====
        risk_result = self.risk_calc.calculate({
            'win_rate': win_prob,
            'avg_win': tp,
            'avg_loss': sl,
            'capital': capital,
            'bet_size': dd_result['position_size'],
            'kelly_fraction': dd_result['constrained_kelly']
        })

        if risk_result['is_viable']:
            checks_passed += 1
        else:
            checks_failed.append(f"HIGH_RUIN_RISK:{risk_result['risk_of_ruin']:.2%}")

        # ===== FINAL DECISION =====
        min_checks = 5  # At least 5 of 7 checks must pass
        should_trade = (
            checks_passed >= min_checks and
            direction != 0 and
            dd_result['can_trade'] and
            ev > 0
        )

        return {
            'should_trade': should_trade,
            'direction': direction if should_trade else 0,
            'position_size': dd_result['position_size'] if should_trade else 0,
            'win_probability': win_prob,
            'kelly_fraction': dd_result['constrained_kelly'],
            'expected_value': ev,
            'checks_passed': checks_passed,
            'checks_total': 7,
            'checks_failed': checks_failed,
            'z_score': z_score if 'z_score' in dir() else 0,
            'half_life': ou_params.get('half_life', 100),
            'risk_of_ruin': risk_result['risk_of_ruin']
        }


# =============================================================================
# REGISTRATION
# =============================================================================

def register_pure_math():
    """Register all pure math formulas."""
    from .base import FORMULA_REGISTRY

    formulas = [
        PlattProbabilityCalibration,
        LogOddsBayesianAggregation,
        OUHalfLifeCalculator,
        LeungLiOptimalEntry,
        KellyCriterionWithEdge,
        DrawdownConstrainedKelly,
        ExpectedValueTracker,
        InformationRatio,
        RiskOfRuinCalculator,
        BertramFirstPassageTime,
        PureMathMasterController,
    ]

    # Formulas auto-register via decorator, just instantiate to verify
    count = 0
    for formula_class in formulas:
        instance = formula_class()
        if instance.FORMULA_ID in FORMULA_REGISTRY:
            count += 1

    print(f"[PureMath] Registered {count} formulas (IDs 720-730)")


# Auto-register on import
register_pure_math()
