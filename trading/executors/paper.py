"""
Paper Executor - Simulated Order Execution
==========================================

Simulates order execution with realistic slippage model.
MUST produce results that closely match real executor for parity.

RenTech Principle: Paper executor behavior must be calibrated
to match real execution as closely as possible.
"""

import asyncio
import random
import time
import uuid
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from ..models import Order, ExecutionResult, OrderSide


@dataclass
class SlippageStats:
    """Track slippage statistics for calibration"""
    total_trades: int = 0
    total_slippage: float = 0.0
    max_slippage: float = 0.0
    min_slippage: float = 1.0

    @property
    def avg_slippage(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.total_slippage / self.total_trades


class SlippageModel:
    """
    Realistic slippage model based on pump.fun bonding curve mechanics.

    Slippage depends on:
    1. Order size relative to liquidity
    2. Bonding curve position (early = more slippage)
    3. Network congestion
    4. Random market impact

    Calibrated from historical pump.fun data.
    """

    def __init__(
        self,
        base_slippage: float = 0.005,      # 0.5% base
        impact_factor: float = 0.05,        # 5% per 100% of liquidity
        congestion_factor: float = 0.01,    # 1% additional during congestion
        randomness: float = 0.002,          # +/- 0.2% noise
    ):
        self.base_slippage = base_slippage
        self.impact_factor = impact_factor
        self.congestion_factor = congestion_factor
        self.randomness = randomness
        self.stats = SlippageStats()

    def estimate(
        self,
        order_size_sol: float,
        liquidity_sol: float = 10.0,
        bonding_progress: float = 0.5,
        is_congested: bool = False,
    ) -> float:
        """
        Estimate slippage percentage for an order.

        Args:
            order_size_sol: Order size in SOL
            liquidity_sol: Current liquidity in SOL
            bonding_progress: Bonding curve progress (0-1)
            is_congested: Whether network is congested

        Returns:
            Estimated slippage as decimal (0.01 = 1%)
        """
        # Base slippage (network + spread)
        slippage = self.base_slippage

        # Impact slippage based on order size vs liquidity
        # Bonding curve is quadratic, so impact increases with size
        liquidity_ratio = order_size_sol / max(liquidity_sol, 0.1)
        impact = liquidity_ratio * self.impact_factor

        # Early bonding curve has less liquidity = more slippage
        # At 10% progress, multiply impact by 2x
        # At 50% progress, multiply impact by 1x
        # At 90% progress, multiply impact by 0.5x
        progress_multiplier = 1.5 - bonding_progress
        impact *= progress_multiplier

        slippage += impact

        # Congestion premium
        if is_congested:
            slippage += self.congestion_factor

        # Random market conditions
        noise = random.gauss(0, self.randomness)
        slippage += noise

        # Clamp to reasonable bounds
        slippage = max(0.001, min(0.15, slippage))

        # Track statistics
        self.stats.total_trades += 1
        self.stats.total_slippage += slippage
        self.stats.max_slippage = max(self.stats.max_slippage, slippage)
        self.stats.min_slippage = min(self.stats.min_slippage, slippage)

        return slippage

    def calibrate(self, real_slippage_data: List[float]):
        """
        Calibrate model parameters from real slippage data.

        Args:
            real_slippage_data: List of actual slippage values from real trades
        """
        if not real_slippage_data:
            return

        avg_real = sum(real_slippage_data) / len(real_slippage_data)

        # Adjust base slippage to match real average
        adjustment = avg_real - self.stats.avg_slippage
        self.base_slippage += adjustment * 0.5  # Partial adjustment

        print(f"Calibrated slippage model: base={self.base_slippage:.4f}")


class PaperExecutor:
    """
    Paper execution that simulates Solana fills.

    Used to validate strategy before real money.
    MUST match real executor behavior for RenTech parity.
    """

    def __init__(
        self,
        slippage_model: Optional[SlippageModel] = None,
        latency_ms: float = 100.0,          # Simulated latency
        failure_rate: float = 0.02,          # 2% random failure rate
    ):
        self.slippage_model = slippage_model or SlippageModel()
        self.latency_ms = latency_ms
        self.failure_rate = failure_rate
        self.execution_log: List[Dict[str, Any]] = []
        self.disabled = False  # Can disable execution for shadow trading

    async def execute(self, order: Order) -> ExecutionResult:
        """
        Simulate order execution.

        Args:
            order: Order to execute

        Returns:
            ExecutionResult with simulated fill details
        """
        start_time = time.time()

        # Check if disabled (for shadow trading)
        if self.disabled:
            return ExecutionResult(
                success=False,
                error="Executor disabled",
                paper_mode=True,
            )

        # Simulate network latency
        await asyncio.sleep(self.latency_ms / 1000)

        # Random failure simulation
        if random.random() < self.failure_rate:
            return ExecutionResult(
                success=False,
                error="Simulated transaction failure",
                paper_mode=True,
                latency_ms=(time.time() - start_time) * 1000,
            )

        # Calculate slippage
        slippage = self.slippage_model.estimate(
            order_size_sol=order.amount_sol,
            liquidity_sol=order.context.get('liquidity', 10.0),
            bonding_progress=order.context.get('bonding_progress', 0.5),
            is_congested=order.context.get('is_congested', False),
        )

        # Apply slippage to fill price
        if order.side == OrderSide.BUY:
            fill_price = order.expected_price * (1 + slippage)
            tokens = order.amount_sol / fill_price
            sol_amount = order.amount_sol
        else:
            fill_price = order.expected_price * (1 - slippage)
            tokens = order.amount_tokens
            sol_amount = tokens * fill_price

        # Generate paper signature
        tx_signature = f"PAPER_{uuid.uuid4().hex[:16]}"

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Build result
        result = ExecutionResult(
            success=True,
            fill_price=fill_price,
            slippage=slippage,
            tokens=tokens,
            sol_amount=sol_amount,
            timestamp=time.time(),
            tx_signature=tx_signature,
            latency_ms=latency_ms,
            paper_mode=True,
        )

        # Log execution
        self.execution_log.append({
            'order': order.to_dict(),
            'result': result.to_dict(),
            'timestamp': time.time(),
        })

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_log:
            return {'trades': 0}

        successful = [e for e in self.execution_log if e['result']['success']]

        return {
            'total_trades': len(self.execution_log),
            'successful_trades': len(successful),
            'success_rate': len(successful) / len(self.execution_log),
            'avg_slippage': self.slippage_model.stats.avg_slippage,
            'max_slippage': self.slippage_model.stats.max_slippage,
            'avg_latency_ms': sum(e['result']['latency_ms'] for e in self.execution_log) / len(self.execution_log),
        }

    def reset(self):
        """Reset execution log and stats"""
        self.execution_log = []
        self.slippage_model.stats = SlippageStats()


# Singleton for easy access
_default_paper_executor: Optional[PaperExecutor] = None


def get_paper_executor() -> PaperExecutor:
    """Get or create default paper executor"""
    global _default_paper_executor
    if _default_paper_executor is None:
        _default_paper_executor = PaperExecutor()
    return _default_paper_executor
