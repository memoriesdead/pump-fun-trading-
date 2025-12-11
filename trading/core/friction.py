"""
Friction Model - Realistic trading costs for Solana DEXes.

Models all real-world costs:
- Solana transaction fees
- DEX swap fees (Raydium, PumpFun)
- Slippage based on trade size and liquidity
- Price impact from AMM constant product formula
- Transaction failure and retry costs

Usage:
    from trading.core import FRICTION

    cost = FRICTION.compute_total_cost(
        trade_size_usd=10.0,
        liquidity_usd=100000,
        is_pumpfun=False,
        sol_price=230.0
    )
    print(f"Round-trip cost: {cost['total_cost_pct']*100:.2f}%")
"""
from dataclasses import dataclass
import random
from typing import Dict, Tuple


@dataclass
class FrictionModel:
    """
    Realistic trading friction for Solana/Raydium/PumpFun.

    All values derived from empirical Solana mainnet data:
    - https://docs.jup.ag/docs/fees
    - https://solana.com/docs/core/fees
    """

    # Transaction fees (fixed costs in SOL)
    # Note: Jito tips are OPTIONAL and only needed for MEV protection
    # For basic trading, just base + priority is sufficient
    solana_base_fee_sol: float = 0.000005
    solana_priority_fee_sol: float = 0.00005  # Reduced from 0.0001 - can be lower for non-urgent
    jito_tip_sol: float = 0.0  # Optional - set to 0 for basic trading

    # DEX fees (percentage)
    jupiter_fee_pct: float = 0.0025         # 0.25%
    raydium_fee_pct: float = 0.0025         # 0.25%
    pumpfun_fee_pct: float = 0.01           # 1%

    # Slippage model
    base_slippage_pct: float = 0.005        # 0.5% base
    slippage_per_1k_usd: float = 0.001      # +0.1% per $1k
    max_slippage_pct: float = 0.03          # 3% cap

    # Transaction failure
    tx_failure_rate: float = 0.15           # 15%
    retry_cost_sol: float = 0.00015
    max_retries: int = 3

    def compute_total_cost(
        self,
        trade_size_usd: float,
        liquidity_usd: float,
        is_pumpfun: bool = False,
        sol_price: float = 230.0,
    ) -> Dict:
        """
        Compute total friction cost for a round-trip trade.

        Returns dict with:
        - total_cost_pct: Total cost as percentage of trade
        - breakdown: Individual cost components
        - min_profit_pct: Minimum profit needed to break even
        """
        if trade_size_usd <= 0:
            return {
                'total_cost_pct': 0,
                'min_profit_pct': 0,
                'breakdown': {},
                'total_cost_usd': 0,
            }

        # 1. Fixed transaction fees (entry + exit)
        fixed_fees_sol = 2 * (
            self.solana_base_fee_sol +
            self.solana_priority_fee_sol +
            self.jito_tip_sol
        )
        fixed_fees_pct = (fixed_fees_sol * sol_price) / trade_size_usd

        # 2. DEX fees (entry + exit)
        dex_fee_pct = 2 * (self.pumpfun_fee_pct if is_pumpfun else self.raydium_fee_pct)

        # 3. Slippage (entry + exit)
        liquidity_ratio = trade_size_usd / liquidity_usd if liquidity_usd > 0 else 1.0
        dynamic_slip = self.base_slippage_pct + (trade_size_usd / 1000) * self.slippage_per_1k_usd
        dynamic_slip = min(dynamic_slip, self.max_slippage_pct)
        slippage_pct = 2 * dynamic_slip * (1 + liquidity_ratio)
        slippage_pct = min(slippage_pct, 2 * self.max_slippage_pct)

        # 4. Price impact (AMM constant product)
        price_impact_pct = 2 * (trade_size_usd / (2 * liquidity_usd)) if liquidity_usd > 0 else 0.02
        price_impact_pct = min(price_impact_pct, 0.05)

        # 5. Expected retry cost
        expected_retries = self.tx_failure_rate * (1 + self.tx_failure_rate)
        retry_cost_sol = expected_retries * self.retry_cost_sol * 2
        retry_cost_pct = (retry_cost_sol * sol_price) / trade_size_usd

        total_cost_pct = (
            fixed_fees_pct +
            dex_fee_pct +
            slippage_pct +
            price_impact_pct +
            retry_cost_pct
        )

        return {
            'total_cost_pct': total_cost_pct,
            'min_profit_pct': total_cost_pct,
            'breakdown': {
                'fixed_fees_pct': fixed_fees_pct,
                'dex_fees_pct': dex_fee_pct,
                'slippage_pct': slippage_pct,
                'price_impact_pct': price_impact_pct,
                'retry_cost_pct': retry_cost_pct,
            },
            'total_cost_usd': total_cost_pct * trade_size_usd,
        }

    def apply_entry_friction(
        self,
        intended_price: float,
        trade_size_usd: float,
        liquidity_usd: float,
        is_pumpfun: bool = False,
    ) -> Tuple[float, float, float]:
        """
        Apply realistic friction to entry price.

        Returns:
        - actual_price: Price after slippage (higher = worse for us)
        - tokens_received: Actual tokens after fees
        - friction_cost_usd: Total friction paid
        """
        if trade_size_usd <= 0 or intended_price <= 0:
            return intended_price, 0, 0

        # Dynamic slippage
        liquidity_ratio = trade_size_usd / liquidity_usd if liquidity_usd > 0 else 0.5
        slip = self.base_slippage_pct + (trade_size_usd / 1000) * self.slippage_per_1k_usd
        slip = min(slip, self.max_slippage_pct)
        slip *= (1 + liquidity_ratio * 0.5)

        # Price impact
        impact = (trade_size_usd / (2 * liquidity_usd)) if liquidity_usd > 0 else 0.01
        impact = min(impact, 0.025)

        # Actual entry price is HIGHER (worse when buying)
        total_slip = slip + impact
        actual_price = intended_price * (1 + total_slip)

        # DEX fee reduces tokens received
        fee_pct = self.pumpfun_fee_pct if is_pumpfun else self.raydium_fee_pct
        effective_usd = trade_size_usd * (1 - fee_pct)
        tokens_received = effective_usd / actual_price if actual_price > 0 else 0

        # Friction cost
        ideal_tokens = trade_size_usd / intended_price if intended_price > 0 else 0
        friction_cost_usd = (ideal_tokens - tokens_received) * intended_price if ideal_tokens > 0 else 0

        return actual_price, tokens_received, max(0, friction_cost_usd)

    def apply_exit_friction(
        self,
        intended_price: float,
        tokens_to_sell: float,
        liquidity_usd: float,
        is_pumpfun: bool = False,
        sol_price: float = 230.0,
    ) -> Tuple[float, float, float]:
        """
        Apply realistic friction to exit price.

        Returns:
        - actual_price: Price after slippage (lower = worse for us)
        - usd_received: Actual USD after fees
        - friction_cost_usd: Total friction paid
        """
        if tokens_to_sell <= 0 or intended_price <= 0:
            return intended_price, 0, 0

        trade_size_usd = tokens_to_sell * intended_price

        # Dynamic slippage
        liquidity_ratio = trade_size_usd / liquidity_usd if liquidity_usd > 0 else 0.5
        slip = self.base_slippage_pct + (trade_size_usd / 1000) * self.slippage_per_1k_usd
        slip = min(slip, self.max_slippage_pct)
        slip *= (1 + liquidity_ratio * 0.5)

        # Price impact
        impact = (trade_size_usd / (2 * liquidity_usd)) if liquidity_usd > 0 else 0.01
        impact = min(impact, 0.025)

        # Actual exit price is LOWER (worse when selling)
        total_slip = slip + impact
        actual_price = intended_price * (1 - total_slip)

        # DEX fee reduces USD received
        fee_pct = self.pumpfun_fee_pct if is_pumpfun else self.raydium_fee_pct
        gross_usd = tokens_to_sell * actual_price
        usd_received = gross_usd * (1 - fee_pct)

        # Transaction fees
        tx_fees_usd = (self.solana_base_fee_sol + self.solana_priority_fee_sol + self.jito_tip_sol) * sol_price
        usd_received -= tx_fees_usd

        # Friction cost
        ideal_usd = tokens_to_sell * intended_price
        friction_cost_usd = ideal_usd - usd_received

        return actual_price, max(0, usd_received), max(0, friction_cost_usd)

    def simulate_tx_success(self) -> Tuple[bool, int, float]:
        """
        Simulate transaction success/failure.

        Returns:
        - success: Whether transaction succeeded
        - retries: Number of retries needed
        - extra_cost_sol: Additional cost from retries
        """
        retries = 0
        success = False
        extra_cost = 0.0

        for _ in range(self.max_retries + 1):
            if random.random() > self.tx_failure_rate:
                success = True
                break
            retries += 1
            extra_cost += self.retry_cost_sol

        return success, retries, extra_cost


# Global instance
FRICTION = FrictionModel()
