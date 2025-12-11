"""
Token Filters - Filter tokens based on trading criteria.

Usage:
    from trading.core import TokenFilter, Token, FilterConfig

    filter = TokenFilter(FilterConfig())
    if filter.passes(token):
        print(f"{token.symbol} passed all filters!")

    # Get detailed rejection reasons
    passed, reasons = filter.check_detailed(token)
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional
from .config import FilterConfig
from .token import Token
from .friction import FRICTION


@dataclass
class FilterResult:
    """Result of filtering a token."""
    passed: bool
    reasons: List[str]
    expected_edge: float
    friction_cost: float
    net_edge: float

    @property
    def is_tradeable(self) -> bool:
        """Check if token has positive expected edge after friction."""
        return self.passed and self.net_edge > 0


class TokenFilter:
    """
    Filter tokens based on mathematical edge criteria.

    All filters are designed to maximize expected value:
    - High momentum = continuation probability
    - High buy ratio = demand pressure
    - Sufficient liquidity = lower friction
    - Sufficient volume = real interest
    """

    def __init__(self, config: Optional[FilterConfig] = None):
        self.config = config or FilterConfig()

    def passes(self, token: Token) -> bool:
        """
        Quick check if token passes all filters.

        Returns:
            True if token passes all filters
        """
        return self.check_detailed(token).passed

    def check_detailed(self, token: Token) -> FilterResult:
        """
        Detailed filter check with rejection reasons.

        Returns:
            FilterResult with pass/fail status and reasons
        """
        reasons = []
        c = self.config

        # 1. Liquidity check (critical for friction)
        if token.liquidity_usd < c.min_liquidity:
            reasons.append(f"liquidity ${token.liquidity_usd:,.0f} < ${c.min_liquidity:,.0f}")
        if token.liquidity_usd > c.max_liquidity:
            reasons.append(f"liquidity ${token.liquidity_usd:,.0f} > ${c.max_liquidity:,.0f}")

        # 2. Volume check (ensures real trading activity)
        if token.volume_5m < c.min_volume_5m:
            reasons.append(f"vol_5m ${token.volume_5m:,.0f} < ${c.min_volume_5m:,.0f}")
        if token.volume_1h < c.min_volume_1h:
            reasons.append(f"vol_1h ${token.volume_1h:,.0f} < ${c.min_volume_1h:,.0f}")

        # 3. Volatility check (need movement to profit)
        volatility = abs(token.change_5m)
        if volatility < c.min_volatility_5m:
            reasons.append(f"volatility {volatility:.1f}% < {c.min_volatility_5m:.1f}%")
        if volatility > c.max_volatility_5m:
            reasons.append(f"volatility {volatility:.1f}% > {c.max_volatility_5m}% (too risky)")

        # 4. Momentum check (THE critical filter)
        if token.momentum < c.min_momentum:
            reasons.append(f"momentum {token.momentum:+.1f}% < {c.min_momentum:+.1f}%")
        if token.momentum > c.max_momentum:
            reasons.append(f"momentum {token.momentum:+.1f}% > {c.max_momentum}% (bubble)")

        # 5. Transaction count check
        if token.txns_5m < c.min_txns_5m:
            reasons.append(f"txns {token.txns_5m} < {c.min_txns_5m}")

        # 6. Buy ratio check (demand pressure)
        if token.buy_ratio < c.min_buy_ratio:
            reasons.append(f"buy_ratio {token.buy_ratio*100:.0f}% < {c.min_buy_ratio*100:.0f}%")

        # 7. Age check
        if token.age_seconds < c.min_age:
            reasons.append(f"too_new {token.age_seconds:.0f}s < {c.min_age}s")
        if token.age_seconds > c.max_age:
            reasons.append(f"too_old {token.age_seconds/3600:.0f}h > {c.max_age/3600:.0f}h")

        # Calculate expected edge
        expected_edge = self._calculate_expected_edge(token)

        # Calculate friction cost
        friction = FRICTION.compute_total_cost(
            trade_size_usd=10.0,  # Assume $10 trade
            liquidity_usd=token.liquidity_usd,
            is_pumpfun=token.is_pumpfun,
        )
        friction_cost = friction['total_cost_pct']

        # Net edge after friction
        net_edge = expected_edge - friction_cost

        return FilterResult(
            passed=len(reasons) == 0,
            reasons=reasons,
            expected_edge=expected_edge,
            friction_cost=friction_cost,
            net_edge=net_edge,
        )

    def _calculate_expected_edge(self, token: Token) -> float:
        """
        Calculate expected edge from token metrics.

        Model: momentum * buy_pressure * continuation_factor

        Returns:
            Expected edge as decimal (0.15 = 15%)
        """
        # Base edge from momentum (already in percentage)
        base_edge = token.momentum / 100

        # Adjust for buy pressure
        buy_pressure = token.buy_ratio
        adjusted_edge = base_edge * buy_pressure

        # Continuation factor (momentum tends to continue in first 60s)
        continuation = 0.4  # 40% of momentum typically continues

        return adjusted_edge * continuation

    def filter_batch(self, tokens: List[Token]) -> List[Tuple[Token, FilterResult]]:
        """
        Filter a batch of tokens and return results sorted by edge.

        Returns:
            List of (token, result) tuples sorted by net_edge descending
        """
        results = []
        for token in tokens:
            result = self.check_detailed(token)
            results.append((token, result))

        # Sort by net edge (highest first)
        results.sort(key=lambda x: x[1].net_edge, reverse=True)
        return results

    def get_best_opportunities(
        self,
        tokens: List[Token],
        max_results: int = 5,
        min_edge: float = 0.05,
    ) -> List[Tuple[Token, FilterResult]]:
        """
        Get the best trading opportunities from a batch.

        Args:
            tokens: List of tokens to filter
            max_results: Maximum number of results
            min_edge: Minimum net edge required (0.05 = 5%)

        Returns:
            List of (token, result) tuples for best opportunities
        """
        all_results = self.filter_batch(tokens)

        # Filter to only passing tokens with sufficient edge
        opportunities = [
            (token, result) for token, result in all_results
            if result.passed and result.net_edge >= min_edge
        ]

        return opportunities[:max_results]

    def analyze_rejections(self, tokens: List[Token]) -> dict:
        """
        Analyze why tokens are being rejected.

        Returns:
            Dictionary with rejection statistics
        """
        total = len(tokens)
        if total == 0:
            return {'total': 0, 'passed': 0, 'rejection_reasons': {}}

        rejection_counts = {}
        passed = 0

        for token in tokens:
            result = self.check_detailed(token)
            if result.passed:
                passed += 1
            else:
                for reason in result.reasons:
                    # Extract reason type (first word)
                    reason_type = reason.split()[0]
                    rejection_counts[reason_type] = rejection_counts.get(reason_type, 0) + 1

        # Sort by count
        sorted_reasons = dict(sorted(
            rejection_counts.items(),
            key=lambda x: x[1],
            reverse=True
        ))

        return {
            'total': total,
            'passed': passed,
            'pass_rate': passed / total if total > 0 else 0,
            'rejection_reasons': sorted_reasons,
        }
