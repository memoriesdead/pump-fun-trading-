#!/usr/bin/env python3
"""
SIGNAL PERFORMANCE ANALYZER
===========================

Deep analysis of signal performance with:
1. Statistical significance testing (t-test, permutation)
2. Correlation analysis between signals
3. Regime-based performance (bull/bear/sideways)
4. Decay analysis (does alpha decay over time?)
5. Feature importance attribution

"The goal is to find signals that are robust, uncorrelated, and persistent."
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path


@dataclass
class DetailedSignalAnalysis:
    """Comprehensive signal analysis results"""
    signal_id: int
    signal_name: str

    # Basic performance
    total_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0

    # Statistical tests
    t_stat: float = 0.0
    t_pvalue: float = 1.0
    permutation_pvalue: float = 1.0
    bootstrap_ci_lower: float = 0.0
    bootstrap_ci_upper: float = 0.0
    is_significant: bool = False

    # Regime analysis
    bull_market_sharpe: float = 0.0
    bear_market_sharpe: float = 0.0
    sideways_market_sharpe: float = 0.0
    regime_consistency: float = 0.0

    # Alpha decay
    decay_half_life_hours: float = 0.0
    first_hour_sharpe: float = 0.0
    after_6h_sharpe: float = 0.0
    is_decaying: bool = False

    # Trade characteristics
    avg_hold_time_sec: float = 0.0
    avg_pnl_per_trade: float = 0.0
    pnl_skewness: float = 0.0
    pnl_kurtosis: float = 0.0
    max_consecutive_losses: int = 0
    max_consecutive_wins: int = 0

    # Time-of-day analysis
    best_hour_utc: int = 0
    worst_hour_utc: int = 0
    hour_consistency: float = 0.0

    # Final scores
    robustness_score: float = 0.0
    overall_score: float = 0.0


class SignalAnalyzer:
    """
    Comprehensive signal analysis engine.

    Analyzes signals across multiple dimensions:
    - Statistical significance
    - Regime robustness
    - Alpha decay
    - Time-of-day patterns
    - Trade characteristics
    """

    def __init__(self, n_permutations: int = 1000, n_bootstrap: int = 1000):
        self.n_permutations = n_permutations
        self.n_bootstrap = n_bootstrap

    def analyze_signal(
        self,
        trades: List[dict],
        signal_id: int,
        signal_name: str
    ) -> DetailedSignalAnalysis:
        """Run full analysis on a signal's trades"""

        if len(trades) < 10:
            return DetailedSignalAnalysis(
                signal_id=signal_id,
                signal_name=signal_name,
                total_trades=len(trades)
            )

        # Extract returns
        returns = np.array([t.get('pnl_pct', 0) for t in trades])
        pnls = np.array([t.get('pnl_sol', 0) for t in trades])
        hold_times = np.array([t.get('hold_time_sec', 0) for t in trades])
        entry_times = np.array([t.get('entry_time', 0) for t in trades])

        # Basic stats
        total_trades = len(trades)
        winning = sum(1 for p in pnls if p > 0)
        win_rate = winning / total_trades
        total_pnl = float(np.sum(pnls))
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe = (mean_return / std_return * np.sqrt(252 * 100)) if std_return > 0 else 0

        # Statistical significance
        t_stat, t_pvalue = self._t_test(returns)
        perm_pvalue = self._permutation_test(returns)
        ci_lower, ci_upper = self._bootstrap_ci(returns)

        is_significant = t_pvalue < 0.05 and perm_pvalue < 0.05

        # Regime analysis
        bull_sharpe, bear_sharpe, side_sharpe = self._regime_analysis(trades)
        regime_consistency = self._regime_consistency(bull_sharpe, bear_sharpe, side_sharpe)

        # Alpha decay
        decay_half_life, first_hour, after_6h, is_decaying = self._alpha_decay_analysis(trades)

        # Trade characteristics
        avg_hold = float(np.mean(hold_times)) if len(hold_times) > 0 else 0
        avg_pnl = float(np.mean(pnls))
        skewness = float(stats.skew(returns)) if len(returns) > 2 else 0
        kurtosis = float(stats.kurtosis(returns)) if len(returns) > 3 else 0
        max_losses, max_wins = self._consecutive_analysis(pnls)

        # Time-of-day analysis
        best_hour, worst_hour, hour_consistency = self._time_of_day_analysis(trades)

        # Robustness score (0-1)
        robustness = self._calculate_robustness(
            sharpe, regime_consistency, is_significant,
            not is_decaying, hour_consistency
        )

        # Overall score (0-1)
        overall = self._calculate_overall_score(
            sharpe, win_rate, total_pnl, robustness, is_significant, total_trades
        )

        return DetailedSignalAnalysis(
            signal_id=signal_id,
            signal_name=signal_name,
            total_trades=total_trades,
            win_rate=float(win_rate),
            total_pnl=float(total_pnl),
            sharpe_ratio=float(sharpe),
            t_stat=float(t_stat),
            t_pvalue=float(t_pvalue),
            permutation_pvalue=float(perm_pvalue),
            bootstrap_ci_lower=float(ci_lower),
            bootstrap_ci_upper=float(ci_upper),
            is_significant=is_significant,
            bull_market_sharpe=float(bull_sharpe),
            bear_market_sharpe=float(bear_sharpe),
            sideways_market_sharpe=float(side_sharpe),
            regime_consistency=float(regime_consistency),
            decay_half_life_hours=float(decay_half_life),
            first_hour_sharpe=float(first_hour),
            after_6h_sharpe=float(after_6h),
            is_decaying=is_decaying,
            avg_hold_time_sec=float(avg_hold),
            avg_pnl_per_trade=float(avg_pnl),
            pnl_skewness=float(skewness),
            pnl_kurtosis=float(kurtosis),
            max_consecutive_losses=max_losses,
            max_consecutive_wins=max_wins,
            best_hour_utc=best_hour,
            worst_hour_utc=worst_hour,
            hour_consistency=float(hour_consistency),
            robustness_score=float(robustness),
            overall_score=float(overall)
        )

    def _t_test(self, returns: np.ndarray) -> Tuple[float, float]:
        """One-sample t-test: Is mean return significantly different from 0?"""
        if len(returns) < 2:
            return 0.0, 1.0

        t_stat, p_value = stats.ttest_1samp(returns, 0)
        return float(t_stat), float(p_value)

    def _permutation_test(self, returns: np.ndarray) -> float:
        """
        Permutation test for statistical significance.
        More robust than parametric tests for non-normal distributions.
        """
        if len(returns) < 10:
            return 1.0

        observed_mean = np.mean(returns)
        count_extreme = 0

        for _ in range(self.n_permutations):
            # Randomly flip signs (null hypothesis: mean = 0)
            permuted = returns * np.random.choice([-1, 1], size=len(returns))
            if abs(np.mean(permuted)) >= abs(observed_mean):
                count_extreme += 1

        return count_extreme / self.n_permutations

    def _bootstrap_ci(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Bootstrap confidence interval for mean return"""
        if len(returns) < 10:
            return 0.0, 0.0

        means = []
        for _ in range(self.n_bootstrap):
            sample = np.random.choice(returns, size=len(returns), replace=True)
            means.append(np.mean(sample))

        alpha = 1 - confidence
        lower = np.percentile(means, alpha / 2 * 100)
        upper = np.percentile(means, (1 - alpha / 2) * 100)

        return float(lower), float(upper)

    def _regime_analysis(self, trades: List[dict]) -> Tuple[float, float, float]:
        """
        Analyze performance across market regimes.

        Simplified regime detection:
        - Bull: Token survived > 10 minutes
        - Bear: Token died < 2 minutes
        - Sideways: Everything else
        """
        bull_returns = []
        bear_returns = []
        side_returns = []

        for trade in trades:
            # Use token survival as proxy for regime
            hold_time = trade.get('hold_time_sec', 0)
            pnl_pct = trade.get('pnl_pct', 0)
            graduated = trade.get('graduated', False)

            if graduated or hold_time > 600:  # > 10 min or graduated
                bull_returns.append(pnl_pct)
            elif hold_time < 120:  # < 2 min
                bear_returns.append(pnl_pct)
            else:
                side_returns.append(pnl_pct)

        def calc_sharpe(returns):
            if len(returns) < 3:
                return 0.0
            mean = np.mean(returns)
            std = np.std(returns)
            return (mean / std * np.sqrt(252 * 100)) if std > 0 else 0

        return (
            calc_sharpe(bull_returns),
            calc_sharpe(bear_returns),
            calc_sharpe(side_returns)
        )

    def _regime_consistency(
        self,
        bull: float,
        bear: float,
        sideways: float
    ) -> float:
        """
        Calculate regime consistency score.
        Higher score = signal works across all regimes.
        """
        sharpes = [bull, bear, sideways]
        positive_count = sum(1 for s in sharpes if s > 0)

        if positive_count == 0:
            return 0.0
        if positive_count == 3:
            # All positive - check variance
            mean_sharpe = np.mean(sharpes)
            std_sharpe = np.std(sharpes)
            if mean_sharpe > 0:
                return min(1.0, mean_sharpe / (std_sharpe + 0.1))
        return positive_count / 3.0

    def _alpha_decay_analysis(
        self,
        trades: List[dict]
    ) -> Tuple[float, float, float, bool]:
        """
        Analyze how quickly alpha decays after signal.

        Returns:
            (half_life_hours, first_hour_sharpe, after_6h_sharpe, is_decaying)
        """
        # Group trades by time since token creation
        first_hour = []
        hour_1_6 = []
        after_6h = []

        for trade in trades:
            token_age = trade.get('token_age_sec', 0)
            pnl_pct = trade.get('pnl_pct', 0)

            if token_age < 3600:  # First hour
                first_hour.append(pnl_pct)
            elif token_age < 21600:  # 1-6 hours
                hour_1_6.append(pnl_pct)
            else:
                after_6h.append(pnl_pct)

        def calc_sharpe(returns):
            if len(returns) < 3:
                return 0.0
            mean = np.mean(returns)
            std = np.std(returns)
            return (mean / std * np.sqrt(252 * 100)) if std > 0 else 0

        first_sharpe = calc_sharpe(first_hour)
        after_sharpe = calc_sharpe(after_6h)

        # Estimate half-life (simplified)
        if first_sharpe > 0 and after_sharpe < first_sharpe:
            # Decay rate
            decay_ratio = after_sharpe / first_sharpe if first_sharpe != 0 else 0
            if decay_ratio > 0:
                half_life = -6 / np.log2(decay_ratio) if decay_ratio < 1 else float('inf')
            else:
                half_life = 1.0  # Very fast decay
        else:
            half_life = float('inf')  # No decay

        is_decaying = first_sharpe > 0.5 and after_sharpe < first_sharpe * 0.5

        return (
            min(half_life, 100),  # Cap at 100 hours
            first_sharpe,
            after_sharpe,
            is_decaying
        )

    def _consecutive_analysis(self, pnls: np.ndarray) -> Tuple[int, int]:
        """Find max consecutive wins and losses"""
        if len(pnls) == 0:
            return 0, 0

        max_losses = 0
        max_wins = 0
        current_losses = 0
        current_wins = 0

        for pnl in pnls:
            if pnl < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)

        return max_losses, max_wins

    def _time_of_day_analysis(
        self,
        trades: List[dict]
    ) -> Tuple[int, int, float]:
        """Analyze performance by hour of day (UTC)"""
        hour_returns = defaultdict(list)

        for trade in trades:
            entry_time = trade.get('entry_time', 0)
            hour = (entry_time // 3600000) % 24  # ms to hours
            pnl_pct = trade.get('pnl_pct', 0)
            hour_returns[hour].append(pnl_pct)

        # Calculate mean return per hour
        hour_means = {}
        for hour, returns in hour_returns.items():
            if len(returns) >= 3:
                hour_means[hour] = np.mean(returns)

        if not hour_means:
            return 0, 0, 0.0

        best_hour = max(hour_means, key=hour_means.get)
        worst_hour = min(hour_means, key=hour_means.get)

        # Consistency: how similar are returns across hours?
        if len(hour_means) >= 3:
            values = list(hour_means.values())
            mean_of_means = np.mean(values)
            std_of_means = np.std(values)
            consistency = 1.0 - min(1.0, std_of_means / (abs(mean_of_means) + 0.001))
        else:
            consistency = 0.5

        return int(best_hour), int(worst_hour), float(consistency)

    def _calculate_robustness(
        self,
        sharpe: float,
        regime_consistency: float,
        is_significant: bool,
        no_decay: bool,
        hour_consistency: float
    ) -> float:
        """Calculate robustness score (0-1)"""
        score = 0.0

        # Sharpe contribution (40%)
        score += 0.4 * min(1.0, max(0.0, sharpe / 2.0))

        # Regime consistency (25%)
        score += 0.25 * regime_consistency

        # Statistical significance (15%)
        score += 0.15 * (1.0 if is_significant else 0.0)

        # No alpha decay (10%)
        score += 0.10 * (1.0 if no_decay else 0.5)

        # Hour consistency (10%)
        score += 0.10 * hour_consistency

        return min(1.0, max(0.0, score))

    def _calculate_overall_score(
        self,
        sharpe: float,
        win_rate: float,
        total_pnl: float,
        robustness: float,
        is_significant: bool,
        n_trades: int
    ) -> float:
        """Calculate overall signal score (0-1)"""
        # Normalize components
        sharpe_score = min(1.0, max(0.0, sharpe / 3.0))
        winrate_score = max(0.0, (win_rate - 0.3) / 0.4)  # 30-70% range
        pnl_score = min(1.0, max(0.0, total_pnl / 10.0))  # 0-10 SOL range
        trade_score = min(1.0, n_trades / 100.0)  # More trades = better

        # Weighted combination
        score = (
            0.25 * sharpe_score +
            0.15 * winrate_score +
            0.15 * pnl_score +
            0.25 * robustness +
            0.10 * (1.0 if is_significant else 0.0) +
            0.10 * trade_score
        )

        return min(1.0, max(0.0, score))


class SignalCorrelationAnalyzer:
    """
    Analyze correlations between signals.

    Goal: Find uncorrelated signals for portfolio diversification.
    """

    def __init__(self):
        self.signal_returns: Dict[int, np.ndarray] = {}
        self.correlation_matrix: Optional[np.ndarray] = None

    def add_signal_returns(self, signal_id: int, returns: np.ndarray):
        """Add returns for a signal"""
        self.signal_returns[signal_id] = returns

    def compute_correlation_matrix(self) -> Dict:
        """Compute correlation matrix between all signals"""
        signal_ids = sorted(self.signal_returns.keys())

        if len(signal_ids) < 2:
            return {'error': 'Need at least 2 signals'}

        # Find common length (use min for now)
        min_len = min(len(self.signal_returns[sid]) for sid in signal_ids)

        # Build matrix
        n = len(signal_ids)
        returns_matrix = np.zeros((n, min_len))

        for i, sid in enumerate(signal_ids):
            returns_matrix[i] = self.signal_returns[sid][:min_len]

        # Compute correlation
        self.correlation_matrix = np.corrcoef(returns_matrix)

        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                corr = self.correlation_matrix[i, j]
                if abs(corr) > 0.7:
                    high_corr_pairs.append({
                        'signal_1': signal_ids[i],
                        'signal_2': signal_ids[j],
                        'correlation': float(corr)
                    })

        # Find uncorrelated signals (for portfolio)
        uncorrelated = []
        for i, sid in enumerate(signal_ids):
            avg_abs_corr = np.mean(np.abs(self.correlation_matrix[i]))
            if avg_abs_corr < 0.3:
                uncorrelated.append({
                    'signal_id': sid,
                    'avg_abs_correlation': float(avg_abs_corr)
                })

        return {
            'signal_ids': signal_ids,
            'correlation_matrix': self.correlation_matrix.tolist(),
            'high_correlation_pairs': high_corr_pairs,
            'uncorrelated_signals': uncorrelated,
            'avg_abs_correlation': float(np.mean(np.abs(self.correlation_matrix)))
        }

    def find_best_uncorrelated_set(
        self,
        n_signals: int = 10,
        min_sharpe: float = 0.5
    ) -> List[int]:
        """
        Find best set of uncorrelated signals.

        Uses greedy selection to maximize portfolio Sharpe.
        """
        if self.correlation_matrix is None:
            self.compute_correlation_matrix()

        signal_ids = sorted(self.signal_returns.keys())
        selected = []
        remaining = set(signal_ids)

        # Greedy selection
        while len(selected) < n_signals and remaining:
            best_signal = None
            best_score = -float('inf')

            for sid in remaining:
                # Score = low correlation with selected + high standalone
                idx = signal_ids.index(sid)

                if selected:
                    selected_indices = [signal_ids.index(s) for s in selected]
                    avg_corr = np.mean([
                        abs(self.correlation_matrix[idx, si])
                        for si in selected_indices
                    ])
                else:
                    avg_corr = 0.0

                # Simple score: prefer low correlation
                score = 1.0 - avg_corr

                if score > best_score:
                    best_score = score
                    best_signal = sid

            if best_signal is not None:
                selected.append(best_signal)
                remaining.remove(best_signal)

        return selected


def run_full_analysis(
    trades_by_signal: Dict[int, List[dict]],
    signal_names: Dict[int, str],
    output_path: Optional[Path] = None
) -> Dict:
    """
    Run complete analysis on all signals.

    Returns:
        Comprehensive analysis results
    """
    analyzer = SignalAnalyzer()
    correlation_analyzer = SignalCorrelationAnalyzer()

    results = {}

    print("=" * 70)
    print("  SIGNAL PERFORMANCE ANALYSIS")
    print("=" * 70)

    for signal_id, trades in trades_by_signal.items():
        signal_name = signal_names.get(signal_id, f"Signal_{signal_id}")
        print(f"  Analyzing [{signal_id}] {signal_name}...")

        # Run analysis
        analysis = analyzer.analyze_signal(trades, signal_id, signal_name)
        results[signal_id] = {
            'analysis': analysis.__dict__,
            'n_trades': len(trades)
        }

        # Add to correlation analyzer
        returns = np.array([t.get('pnl_pct', 0) for t in trades])
        if len(returns) >= 10:
            correlation_analyzer.add_signal_returns(signal_id, returns)

    # Correlation analysis
    print("\n  Computing signal correlations...")
    correlation_results = correlation_analyzer.compute_correlation_matrix()

    # Find best uncorrelated set
    best_portfolio = correlation_analyzer.find_best_uncorrelated_set(n_signals=10)

    # Summary
    significant_count = sum(
        1 for r in results.values()
        if r['analysis'].get('is_significant', False)
    )

    top_signals = sorted(
        results.items(),
        key=lambda x: x[1]['analysis'].get('overall_score', 0),
        reverse=True
    )[:20]

    output = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'summary': {
            'total_signals_analyzed': len(results),
            'significant_signals': significant_count,
            'avg_sharpe': float(np.mean([
                r['analysis'].get('sharpe_ratio', 0) for r in results.values()
            ])),
            'avg_win_rate': float(np.mean([
                r['analysis'].get('win_rate', 0) for r in results.values()
            ])),
        },
        'top_20_signals': [
            {
                'signal_id': sid,
                'signal_name': results[sid]['analysis']['signal_name'],
                'overall_score': results[sid]['analysis']['overall_score'],
                'sharpe': results[sid]['analysis']['sharpe_ratio'],
                'win_rate': results[sid]['analysis']['win_rate'],
                'robustness': results[sid]['analysis']['robustness_score'],
                'is_significant': results[sid]['analysis']['is_significant']
            }
            for sid, _ in top_signals
        ],
        'best_portfolio_signals': best_portfolio,
        'correlation_analysis': correlation_results,
        'full_results': results
    }

    # Save if path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("  TOP 10 SIGNALS BY OVERALL SCORE")
    print("=" * 70)

    for sid, _ in top_signals[:10]:
        a = results[sid]['analysis']
        sig = "***" if a['is_significant'] else "   "
        print(f"  {sig} [{sid}] {a['signal_name']}")
        print(f"      Score: {a['overall_score']:.3f} | Sharpe: {a['sharpe_ratio']:.2f} | "
              f"Win: {a['win_rate']:.1%} | Robust: {a['robustness_score']:.2f}")

    return output


if __name__ == "__main__":
    # Example usage
    print("Signal Analyzer - Run from simulation orchestrator")
