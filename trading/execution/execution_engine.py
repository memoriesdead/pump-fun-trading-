#!/usr/bin/env python3
"""
RENTECH EXECUTION ENGINE
========================

Smart order routing + MEV protection for pump.fun trading on Solana.

Key Features:
1. Smart order routing (Raydium, Orca, pump.fun bonding curve)
2. MEV protection (private mempools, Jito bundles)
3. Slippage optimization
4. Gas optimization (priority fees)
5. Execution analytics

Based on:
- Optimal execution research (Almgren-Chriss model)
- MEV protection strategies (Flashbots concepts)
- Market microstructure theory
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
from collections import defaultdict
import json
import hashlib


# ============================================================
# CONFIGURATION
# ============================================================

# Pump.fun Constants
PUMP_PROGRAM_ID = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
PUMP_FEE_BPS = 100  # 1% pump.fun fee

# Bonding Curve Parameters (y = x^2 / 1B)
BONDING_CURVE_DIVISOR = 1_000_000_000
GRADUATION_SOL = 85.0  # Graduates to Raydium at ~85 SOL

# Slippage Settings
DEFAULT_SLIPPAGE_BPS = 100  # 1% default slippage
MAX_SLIPPAGE_BPS = 500      # 5% max allowed
MIN_SLIPPAGE_BPS = 10       # 0.1% minimum

# Priority Fee Settings (in microlamports)
BASE_PRIORITY_FEE = 10_000       # 0.01 SOL
HIGH_PRIORITY_FEE = 100_000     # 0.1 SOL for urgent
ULTRA_PRIORITY_FEE = 500_000    # 0.5 SOL for critical

# MEV Protection
JITO_TIP_LAMPORTS = 1_000_000   # 0.001 SOL default Jito tip
PRIVATE_MEMPOOL_TIMEOUT = 5.0    # seconds

# Retry Settings
MAX_RETRIES = 3
RETRY_DELAY = 1.0


# ============================================================
# DATA STRUCTURES
# ============================================================

class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"       # Time-weighted average price
    ICEBERG = "iceberg"  # Split into chunks


class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


class ExecutionVenue(Enum):
    """Execution venues"""
    PUMP_BONDING = "pump_bonding"    # Pump.fun bonding curve
    RAYDIUM = "raydium"              # Raydium AMM
    ORCA = "orca"                    # Orca Whirlpool
    JUPITER = "jupiter"             # Jupiter aggregator


class MEVProtection(Enum):
    """MEV protection levels"""
    NONE = "none"           # Standard mempool
    JITO_BUNDLE = "jito"    # Jito bundle for tips
    PRIVATE_RPC = "private" # Private RPC endpoint


@dataclass
class Order:
    """Trading order"""
    id: str
    mint: str
    side: OrderSide
    amount_sol: float
    order_type: OrderType = OrderType.MARKET
    slippage_bps: int = DEFAULT_SLIPPAGE_BPS
    priority_fee: int = BASE_PRIORITY_FEE
    mev_protection: MEVProtection = MEVProtection.JITO_BUNDLE
    venue_preference: Optional[ExecutionVenue] = None
    max_price_impact_bps: int = 200  # 2% max impact
    created_at: float = field(default_factory=time.time)


@dataclass
class Quote:
    """Price quote from venue"""
    venue: ExecutionVenue
    input_amount: float
    output_amount: float
    price_impact_bps: float
    fee_bps: float
    route: List[str]  # Token path
    valid_until: float


@dataclass
class ExecutionResult:
    """Result of order execution"""
    order_id: str
    success: bool
    signature: Optional[str]
    venue: ExecutionVenue
    executed_amount: float
    average_price: float
    slippage_realized: float
    fees_paid_sol: float
    priority_fee_paid: float
    latency_ms: float
    error: Optional[str] = None


@dataclass
class ExecutionStats:
    """Aggregated execution statistics"""
    total_orders: int = 0
    successful_orders: int = 0
    failed_orders: int = 0
    total_volume_sol: float = 0.0
    total_fees_sol: float = 0.0
    avg_slippage_bps: float = 0.0
    avg_latency_ms: float = 0.0
    venue_distribution: Dict[str, int] = field(default_factory=dict)


# ============================================================
# BONDING CURVE MATH
# ============================================================

class PumpBondingCurve:
    """
    Pump.fun bonding curve mathematics.

    The bonding curve is: price = tokens_sold^2 / 1B
    This means early buyers get better prices.
    """

    @staticmethod
    def calculate_buy_price(
        current_supply: int,
        buy_amount_tokens: int
    ) -> float:
        """
        Calculate SOL needed to buy tokens.

        Integrates: price = x^2 / 1B from current to current+amount
        Result: SOL = (1/3B) * (new^3 - old^3)
        """
        old_supply = current_supply
        new_supply = current_supply + buy_amount_tokens

        # Integration of x^2/1B from old to new
        sol_cost = (new_supply**3 - old_supply**3) / (3 * BONDING_CURVE_DIVISOR)

        return sol_cost / 1e9  # Convert to SOL

    @staticmethod
    def calculate_sell_price(
        current_supply: int,
        sell_amount_tokens: int
    ) -> float:
        """
        Calculate SOL received for selling tokens.

        Same formula but in reverse.
        """
        old_supply = current_supply
        new_supply = current_supply - sell_amount_tokens

        if new_supply < 0:
            new_supply = 0

        sol_received = (old_supply**3 - new_supply**3) / (3 * BONDING_CURVE_DIVISOR)

        return sol_received / 1e9

    @staticmethod
    def calculate_price_impact(
        current_supply: int,
        amount_tokens: int,
        is_buy: bool
    ) -> float:
        """Calculate price impact in basis points"""
        if current_supply == 0:
            return 10000  # 100% impact for first trade

        current_price = (current_supply**2) / BONDING_CURVE_DIVISOR

        if is_buy:
            new_supply = current_supply + amount_tokens
        else:
            new_supply = max(0, current_supply - amount_tokens)

        new_price = (new_supply**2) / BONDING_CURVE_DIVISOR

        impact = abs(new_price - current_price) / current_price

        return impact * 10000  # Convert to bps

    @staticmethod
    def estimate_graduation_progress(sol_in_curve: float) -> float:
        """Estimate progress toward graduation (0-100%)"""
        return min(100.0, (sol_in_curve / GRADUATION_SOL) * 100)


# ============================================================
# SMART ORDER ROUTER
# ============================================================

class SmartOrderRouter:
    """
    Routes orders to optimal venues.

    Considers:
    - Price impact
    - Fees
    - Liquidity depth
    - Latency
    - MEV protection
    """

    def __init__(self):
        self.venue_stats: Dict[ExecutionVenue, Dict] = defaultdict(
            lambda: {"fills": 0, "avg_slippage": 0}
        )
        self.bonding_curve = PumpBondingCurve()

    async def get_best_route(
        self,
        order: Order,
        token_state: Optional[Dict] = None
    ) -> Tuple[ExecutionVenue, Quote]:
        """
        Find the best execution route for an order.

        Returns: (venue, quote)
        """
        # Get quotes from all venues
        quotes = await self._get_all_quotes(order, token_state)

        if not quotes:
            raise ValueError("No quotes available for order")

        # Score each quote
        scored = []
        for venue, quote in quotes.items():
            score = self._score_quote(quote, order)
            scored.append((score, venue, quote))

        # Sort by score (higher is better)
        scored.sort(reverse=True, key=lambda x: x[0])

        best_venue, best_quote = scored[0][1], scored[0][2]

        return best_venue, best_quote

    async def _get_all_quotes(
        self,
        order: Order,
        token_state: Optional[Dict]
    ) -> Dict[ExecutionVenue, Quote]:
        """Get quotes from all available venues"""
        quotes = {}

        # Check if token is still on bonding curve
        is_graduated = token_state and token_state.get("graduated", False)

        if not is_graduated:
            # Pump.fun bonding curve quote
            quote = self._get_bonding_curve_quote(order, token_state)
            if quote:
                quotes[ExecutionVenue.PUMP_BONDING] = quote
        else:
            # Token graduated - check DEXes
            # In production, these would be actual RPC calls
            raydium_quote = await self._get_raydium_quote(order)
            if raydium_quote:
                quotes[ExecutionVenue.RAYDIUM] = raydium_quote

            jupiter_quote = await self._get_jupiter_quote(order)
            if jupiter_quote:
                quotes[ExecutionVenue.JUPITER] = jupiter_quote

        return quotes

    def _get_bonding_curve_quote(
        self,
        order: Order,
        token_state: Optional[Dict]
    ) -> Optional[Quote]:
        """Calculate quote from pump.fun bonding curve"""
        if not token_state:
            return None

        current_supply = token_state.get("token_supply", 0)
        sol_in_curve = token_state.get("sol_reserves", 0)

        # Estimate tokens for SOL amount
        # This is a simplified calculation
        if order.side == OrderSide.BUY:
            # Estimate tokens received for SOL
            estimated_tokens = self._estimate_tokens_for_sol(
                order.amount_sol, current_supply
            )
            output = estimated_tokens
            impact_bps = self.bonding_curve.calculate_price_impact(
                current_supply, estimated_tokens, True
            )
        else:
            # Selling tokens for SOL
            estimated_sol = self.bonding_curve.calculate_sell_price(
                current_supply, int(order.amount_sol * 1e9)  # Assume amount is in tokens
            )
            output = estimated_sol
            impact_bps = self.bonding_curve.calculate_price_impact(
                current_supply, int(order.amount_sol * 1e9), False
            )

        return Quote(
            venue=ExecutionVenue.PUMP_BONDING,
            input_amount=order.amount_sol,
            output_amount=output,
            price_impact_bps=impact_bps,
            fee_bps=PUMP_FEE_BPS,
            route=[order.mint],
            valid_until=time.time() + 30  # Valid for 30 seconds
        )

    def _estimate_tokens_for_sol(
        self,
        sol_amount: float,
        current_supply: int
    ) -> float:
        """Estimate tokens received for given SOL amount"""
        # Solve: sol = (new^3 - old^3) / 3B for new
        # new^3 = sol * 3B + old^3
        sol_lamports = sol_amount * 1e9
        old_cubed = current_supply ** 3
        new_cubed = sol_lamports * 3 * BONDING_CURVE_DIVISOR + old_cubed
        new_supply = new_cubed ** (1/3)
        return new_supply - current_supply

    async def _get_raydium_quote(self, order: Order) -> Optional[Quote]:
        """Get quote from Raydium (placeholder)"""
        # In production, this would query Raydium's API
        return None

    async def _get_jupiter_quote(self, order: Order) -> Optional[Quote]:
        """Get quote from Jupiter (placeholder)"""
        # In production, this would query Jupiter's API
        return None

    def _score_quote(self, quote: Quote, order: Order) -> float:
        """
        Score a quote for ranking.

        Higher score = better quote.
        """
        score = 0.0

        # Price impact penalty (major factor)
        if quote.price_impact_bps <= order.max_price_impact_bps:
            score += 100 - (quote.price_impact_bps / 10)
        else:
            score -= 50  # Significant penalty for exceeding max impact

        # Fee bonus (lower is better)
        score += max(0, 20 - quote.fee_bps / 10)

        # Output amount bonus (more is better)
        score += quote.output_amount * 0.01

        # Venue preference bonus
        if order.venue_preference == quote.venue:
            score += 10

        return score


# ============================================================
# MEV PROTECTOR
# ============================================================

class MEVProtector:
    """
    MEV protection for Solana transactions.

    Strategies:
    1. Jito bundles - Pay for inclusion guarantee
    2. Private RPCs - Avoid public mempool
    3. Order timing - Random delays
    """

    def __init__(
        self,
        jito_tip: int = JITO_TIP_LAMPORTS,
        private_rpc: Optional[str] = None
    ):
        self.jito_tip = jito_tip
        self.private_rpc = private_rpc
        self.stats = {
            "bundles_sent": 0,
            "private_txs": 0,
            "standard_txs": 0
        }

    async def protect_transaction(
        self,
        transaction: bytes,
        protection_level: MEVProtection
    ) -> Tuple[bool, Optional[str]]:
        """
        Send transaction with MEV protection.

        Returns: (success, signature)
        """
        if protection_level == MEVProtection.JITO_BUNDLE:
            return await self._send_jito_bundle(transaction)
        elif protection_level == MEVProtection.PRIVATE_RPC:
            return await self._send_private(transaction)
        else:
            return await self._send_standard(transaction)

    async def _send_jito_bundle(
        self,
        transaction: bytes
    ) -> Tuple[bool, Optional[str]]:
        """Send via Jito bundle with tip"""
        self.stats["bundles_sent"] += 1

        # In production:
        # 1. Create bundle with tip transaction
        # 2. Submit to Jito block engine
        # 3. Wait for confirmation

        # Placeholder
        await asyncio.sleep(0.1)
        signature = f"jito_{hashlib.sha256(transaction).hexdigest()[:16]}"

        return True, signature

    async def _send_private(
        self,
        transaction: bytes
    ) -> Tuple[bool, Optional[str]]:
        """Send via private RPC endpoint"""
        self.stats["private_txs"] += 1

        if not self.private_rpc:
            return await self._send_standard(transaction)

        # In production: Send to private RPC
        await asyncio.sleep(0.1)
        signature = f"priv_{hashlib.sha256(transaction).hexdigest()[:16]}"

        return True, signature

    async def _send_standard(
        self,
        transaction: bytes
    ) -> Tuple[bool, Optional[str]]:
        """Send via standard RPC (exposed to MEV)"""
        self.stats["standard_txs"] += 1

        # In production: Send to public RPC
        await asyncio.sleep(0.1)
        signature = f"std_{hashlib.sha256(transaction).hexdigest()[:16]}"

        return True, signature

    def calculate_optimal_tip(
        self,
        order_value_sol: float,
        urgency: str = "normal"
    ) -> int:
        """
        Calculate optimal Jito tip based on order value.

        Higher value orders warrant higher tips for better placement.
        """
        base = JITO_TIP_LAMPORTS

        # Scale with order value
        value_multiplier = min(10, order_value_sol)

        # Urgency multiplier
        urgency_mult = {
            "low": 0.5,
            "normal": 1.0,
            "high": 2.0,
            "critical": 5.0
        }.get(urgency, 1.0)

        tip = int(base * value_multiplier * urgency_mult)

        return tip


# ============================================================
# EXECUTION ENGINE
# ============================================================

class ExecutionEngine:
    """
    Main execution engine.

    Orchestrates order routing, execution, and MEV protection.
    """

    def __init__(
        self,
        rpc_endpoint: str = "https://api.mainnet-beta.solana.com",
        jito_enabled: bool = True,
        private_rpc: Optional[str] = None
    ):
        self.rpc_endpoint = rpc_endpoint
        self.router = SmartOrderRouter()
        self.mev = MEVProtector(
            private_rpc=private_rpc
        )

        self.stats = ExecutionStats()
        self.pending_orders: Dict[str, Order] = {}
        self.execution_history: List[ExecutionResult] = []

    async def execute_order(
        self,
        order: Order,
        token_state: Optional[Dict] = None
    ) -> ExecutionResult:
        """
        Execute a trading order.

        Full flow:
        1. Get best route
        2. Build transaction
        3. Apply MEV protection
        4. Send and confirm
        5. Record analytics
        """
        start_time = time.time()

        try:
            # 1. Get optimal route
            venue, quote = await self.router.get_best_route(order, token_state)

            # 2. Validate quote
            if quote.price_impact_bps > order.max_price_impact_bps:
                return ExecutionResult(
                    order_id=order.id,
                    success=False,
                    signature=None,
                    venue=venue,
                    executed_amount=0,
                    average_price=0,
                    slippage_realized=0,
                    fees_paid_sol=0,
                    priority_fee_paid=0,
                    latency_ms=(time.time() - start_time) * 1000,
                    error=f"Price impact {quote.price_impact_bps} bps exceeds max {order.max_price_impact_bps}"
                )

            # 3. Build transaction (placeholder)
            transaction = self._build_transaction(order, quote)

            # 4. Execute with MEV protection
            success, signature = await self.mev.protect_transaction(
                transaction,
                order.mev_protection
            )

            # 5. Calculate realized slippage
            expected_price = quote.input_amount / quote.output_amount if quote.output_amount > 0 else 0
            realized_slippage = quote.price_impact_bps  # Simplified

            # 6. Calculate fees
            fees_sol = order.amount_sol * (quote.fee_bps / 10000)

            result = ExecutionResult(
                order_id=order.id,
                success=success,
                signature=signature,
                venue=venue,
                executed_amount=order.amount_sol,
                average_price=expected_price,
                slippage_realized=realized_slippage,
                fees_paid_sol=fees_sol,
                priority_fee_paid=order.priority_fee / 1e9,  # Convert to SOL
                latency_ms=(time.time() - start_time) * 1000
            )

            # Update stats
            self._update_stats(result)

            return result

        except Exception as e:
            return ExecutionResult(
                order_id=order.id,
                success=False,
                signature=None,
                venue=ExecutionVenue.PUMP_BONDING,
                executed_amount=0,
                average_price=0,
                slippage_realized=0,
                fees_paid_sol=0,
                priority_fee_paid=0,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )

    def _build_transaction(self, order: Order, quote: Quote) -> bytes:
        """Build Solana transaction (placeholder)"""
        # In production, this would create actual Solana transaction
        tx_data = {
            "order_id": order.id,
            "mint": order.mint,
            "side": order.side.value,
            "amount": order.amount_sol,
            "venue": quote.venue.value
        }
        return json.dumps(tx_data).encode()

    def _update_stats(self, result: ExecutionResult):
        """Update execution statistics"""
        self.stats.total_orders += 1

        if result.success:
            self.stats.successful_orders += 1
            self.stats.total_volume_sol += result.executed_amount
            self.stats.total_fees_sol += result.fees_paid_sol

            # Update running averages
            n = self.stats.successful_orders
            self.stats.avg_slippage_bps = (
                (self.stats.avg_slippage_bps * (n-1) + result.slippage_realized) / n
            )
            self.stats.avg_latency_ms = (
                (self.stats.avg_latency_ms * (n-1) + result.latency_ms) / n
            )

            # Venue distribution
            venue_name = result.venue.value
            self.stats.venue_distribution[venue_name] = \
                self.stats.venue_distribution.get(venue_name, 0) + 1
        else:
            self.stats.failed_orders += 1

        self.execution_history.append(result)

    async def execute_twap(
        self,
        order: Order,
        duration_seconds: int,
        num_slices: int = 10
    ) -> List[ExecutionResult]:
        """
        Execute order using TWAP strategy.

        Splits order into slices over time to minimize impact.
        """
        slice_amount = order.amount_sol / num_slices
        interval = duration_seconds / num_slices

        results = []

        for i in range(num_slices):
            slice_order = Order(
                id=f"{order.id}_slice_{i}",
                mint=order.mint,
                side=order.side,
                amount_sol=slice_amount,
                order_type=OrderType.MARKET,
                slippage_bps=order.slippage_bps,
                priority_fee=order.priority_fee,
                mev_protection=order.mev_protection
            )

            result = await self.execute_order(slice_order)
            results.append(result)

            if i < num_slices - 1:
                await asyncio.sleep(interval)

        return results

    async def execute_iceberg(
        self,
        order: Order,
        visible_size_pct: float = 0.2
    ) -> List[ExecutionResult]:
        """
        Execute order using iceberg strategy.

        Only shows part of order at a time.
        """
        visible_amount = order.amount_sol * visible_size_pct
        remaining = order.amount_sol
        results = []
        slice_num = 0

        while remaining > 0:
            slice_amount = min(visible_amount, remaining)

            slice_order = Order(
                id=f"{order.id}_ice_{slice_num}",
                mint=order.mint,
                side=order.side,
                amount_sol=slice_amount,
                order_type=OrderType.MARKET,
                slippage_bps=order.slippage_bps,
                priority_fee=order.priority_fee,
                mev_protection=order.mev_protection
            )

            result = await self.execute_order(slice_order)
            results.append(result)

            if result.success:
                remaining -= slice_amount
            else:
                break  # Stop on failure

            slice_num += 1

            # Random delay to appear natural
            await asyncio.sleep(0.5 + 0.5 * (hash(order.id) % 10) / 10)

        return results

    def get_stats(self) -> ExecutionStats:
        """Get execution statistics"""
        return self.stats

    def reset_stats(self):
        """Reset statistics"""
        self.stats = ExecutionStats()
        self.execution_history = []


# ============================================================
# EXECUTION ANALYTICS
# ============================================================

class ExecutionAnalytics:
    """
    Analyzes execution quality.

    Metrics:
    - Implementation shortfall
    - Slippage analysis
    - Venue performance
    - Cost analysis
    """

    def __init__(self, engine: ExecutionEngine):
        self.engine = engine

    def calculate_implementation_shortfall(
        self,
        results: List[ExecutionResult],
        arrival_price: float
    ) -> float:
        """
        Calculate implementation shortfall.

        IS = (Execution Price - Arrival Price) / Arrival Price
        """
        if not results or arrival_price <= 0:
            return 0.0

        total_value = sum(r.executed_amount for r in results if r.success)
        total_paid = sum(
            r.executed_amount * r.average_price
            for r in results if r.success
        )

        if total_value <= 0:
            return 0.0

        avg_execution_price = total_paid / total_value

        return (avg_execution_price - arrival_price) / arrival_price

    def analyze_slippage(self) -> Dict:
        """Analyze slippage across all executions"""
        history = self.engine.execution_history

        if not history:
            return {"error": "No execution history"}

        slippages = [r.slippage_realized for r in history if r.success]

        if not slippages:
            return {"error": "No successful executions"}

        import statistics

        return {
            "mean_slippage_bps": statistics.mean(slippages),
            "median_slippage_bps": statistics.median(slippages),
            "std_slippage_bps": statistics.stdev(slippages) if len(slippages) > 1 else 0,
            "max_slippage_bps": max(slippages),
            "min_slippage_bps": min(slippages),
            "total_executions": len(slippages)
        }

    def venue_performance(self) -> Dict[str, Dict]:
        """Analyze performance by venue"""
        history = self.engine.execution_history
        venue_data = defaultdict(list)

        for result in history:
            if result.success:
                venue_data[result.venue.value].append(result)

        analysis = {}
        for venue, results in venue_data.items():
            slippages = [r.slippage_realized for r in results]
            latencies = [r.latency_ms for r in results]

            import statistics

            analysis[venue] = {
                "executions": len(results),
                "avg_slippage_bps": statistics.mean(slippages),
                "avg_latency_ms": statistics.mean(latencies),
                "success_rate": 1.0,  # Only successful in venue_data
                "total_volume_sol": sum(r.executed_amount for r in results)
            }

        return analysis

    def cost_breakdown(self) -> Dict:
        """Break down execution costs"""
        stats = self.engine.get_stats()

        if stats.total_volume_sol <= 0:
            return {"error": "No volume"}

        return {
            "total_volume_sol": stats.total_volume_sol,
            "total_fees_sol": stats.total_fees_sol,
            "fee_pct": (stats.total_fees_sol / stats.total_volume_sol) * 100,
            "avg_slippage_cost_bps": stats.avg_slippage_bps,
            "total_orders": stats.total_orders,
            "success_rate": stats.successful_orders / max(1, stats.total_orders)
        }


# ============================================================
# MAIN RUNNER
# ============================================================

async def example_usage():
    """Example of execution engine usage"""
    print("=" * 60)
    print("  RENTECH EXECUTION ENGINE")
    print("=" * 60)
    print()

    # Create engine
    engine = ExecutionEngine()
    analytics = ExecutionAnalytics(engine)

    # Simulate token state
    token_state = {
        "mint": "pump_token_123",
        "token_supply": 500_000_000,  # 500M tokens
        "sol_reserves": 10.0,  # 10 SOL in curve
        "graduated": False
    }

    # Create test orders
    orders = [
        Order(
            id="order_1",
            mint="pump_token_123",
            side=OrderSide.BUY,
            amount_sol=0.5,
            slippage_bps=150,
            mev_protection=MEVProtection.JITO_BUNDLE
        ),
        Order(
            id="order_2",
            mint="pump_token_123",
            side=OrderSide.BUY,
            amount_sol=1.0,
            slippage_bps=200,
            mev_protection=MEVProtection.PRIVATE_RPC
        ),
        Order(
            id="order_3",
            mint="pump_token_123",
            side=OrderSide.SELL,
            amount_sol=0.3,
            mev_protection=MEVProtection.NONE
        )
    ]

    # Execute orders
    print("Executing orders:")
    print("-" * 60)

    for order in orders:
        result = await engine.execute_order(order, token_state)
        print(f"  Order {order.id}:")
        print(f"    Success: {result.success}")
        print(f"    Venue: {result.venue.value}")
        print(f"    Latency: {result.latency_ms:.1f}ms")
        if result.signature:
            print(f"    Signature: {result.signature}")
        print()

    # Show stats
    stats = engine.get_stats()
    print("Execution Statistics:")
    print("-" * 60)
    print(f"  Total orders: {stats.total_orders}")
    print(f"  Successful: {stats.successful_orders}")
    print(f"  Failed: {stats.failed_orders}")
    print(f"  Total volume: {stats.total_volume_sol:.2f} SOL")
    print(f"  Avg slippage: {stats.avg_slippage_bps:.1f} bps")
    print(f"  Avg latency: {stats.avg_latency_ms:.1f}ms")
    print()

    # Cost breakdown
    print("Cost Breakdown:")
    print("-" * 60)
    costs = analytics.cost_breakdown()
    for key, value in costs.items():
        print(f"  {key}: {value}")
    print()

    # Bonding curve example
    print("Bonding Curve Analysis:")
    print("-" * 60)
    bc = PumpBondingCurve()
    current = 500_000_000  # 500M tokens
    print(f"  Current supply: {current:,} tokens")
    print(f"  Graduation progress: {bc.estimate_graduation_progress(10.0):.1f}%")
    print(f"  Price impact (buy 10M): {bc.calculate_price_impact(current, 10_000_000, True):.0f} bps")
    print(f"  Price impact (sell 10M): {bc.calculate_price_impact(current, 10_000_000, False):.0f} bps")


if __name__ == "__main__":
    asyncio.run(example_usage())
