"""
Trading Models - Shared Data Structures
=======================================

All data structures used by TradingEngine.
Paper and Real modes use IDENTICAL structures.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import time
import uuid


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


@dataclass
class Order:
    """Order to execute (identical for paper/real)"""
    mint: str
    side: OrderSide
    amount_sol: float = 0.0
    amount_tokens: float = 0.0
    expected_price: float = 0.0
    order_type: OrderType = OrderType.MARKET
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    order_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    def to_dict(self) -> dict:
        return {
            'order_id': self.order_id,
            'mint': self.mint,
            'side': self.side.value,
            'amount_sol': self.amount_sol,
            'amount_tokens': self.amount_tokens,
            'expected_price': self.expected_price,
            'order_type': self.order_type.value,
            'timestamp': self.timestamp,
        }


@dataclass
class ExecutionResult:
    """Result of order execution (paper or real)"""
    success: bool
    fill_price: float = 0.0
    slippage: float = 0.0
    tokens: float = 0.0
    sol_amount: float = 0.0
    timestamp: float = field(default_factory=time.time)
    tx_signature: str = ""
    latency_ms: float = 0.0
    paper_mode: bool = True
    error: Optional[str] = None
    realized_pnl: float = 0.0

    def to_dict(self) -> dict:
        return {
            'success': self.success,
            'fill_price': self.fill_price,
            'slippage': self.slippage,
            'tokens': self.tokens,
            'sol_amount': self.sol_amount,
            'timestamp': self.timestamp,
            'tx_signature': self.tx_signature,
            'latency_ms': self.latency_ms,
            'paper_mode': self.paper_mode,
            'error': self.error,
            'realized_pnl': self.realized_pnl,
        }


@dataclass
class Position:
    """Open position (identical tracking for paper/real)"""
    mint: str
    entry_price: float
    entry_time: float
    tokens: float
    sol_invested: float
    entry_signals: Dict[int, float] = field(default_factory=dict)
    peak_pnl: float = 0.0
    partial_exits: int = 0
    tx_signature: str = ""
    paper_mode: bool = True

    # Friction tracking (RenTech reality)
    actual_entry_price: float = 0.0     # Price after slippage
    entry_friction_usd: float = 0.0     # Friction paid on entry
    actual_tokens: float = 0.0          # Tokens received after fees
    liquidity_at_entry: float = 0.0     # For slippage calculation on exit

    @property
    def current_pnl(self) -> float:
        """Requires current price to calculate - set externally"""
        return getattr(self, '_current_pnl', 0.0)

    @current_pnl.setter
    def current_pnl(self, value: float):
        self._current_pnl = value
        if value > self.peak_pnl:
            self.peak_pnl = value

    def to_dict(self) -> dict:
        return {
            'mint': self.mint,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time,
            'tokens': self.tokens,
            'sol_invested': self.sol_invested,
            'peak_pnl': self.peak_pnl,
            'partial_exits': self.partial_exits,
            'paper_mode': self.paper_mode,
        }


@dataclass
class EntryDecision:
    """Decision on whether to enter a position"""
    should_enter: bool
    reason: str
    quality_score: int = 0
    signals: Dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'should_enter': self.should_enter,
            'reason': self.reason,
            'quality_score': self.quality_score,
        }


@dataclass
class ExitDecision:
    """Decision on whether to exit a position"""
    should_exit: bool
    reason: Optional[str]
    current_pnl: float

    def to_dict(self) -> dict:
        return {
            'should_exit': self.should_exit,
            'reason': self.reason,
            'current_pnl': self.current_pnl,
        }


@dataclass
class TradeDecision:
    """Complete trade decision (entry or exit)"""
    action: str  # 'BUY', 'SELL', 'SKIP'
    mint: str = ""
    size: float = 0.0
    pnl: float = 0.0
    reason: str = ""
    signals: Dict[int, float] = field(default_factory=dict)
    result: Optional[ExecutionResult] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            'action': self.action,
            'mint': self.mint,
            'size': self.size,
            'pnl': self.pnl,
            'reason': self.reason,
            'timestamp': self.timestamp,
            'result': self.result.to_dict() if self.result else None,
        }


@dataclass
class ClosedTrade:
    """Completed trade record"""
    mint: str
    entry_price: float
    exit_price: float
    entry_time: float
    exit_time: float
    tokens: float
    sol_invested: float
    sol_received: float
    pnl_sol: float
    pnl_pct: float
    exit_reason: str
    entry_signals: Dict[int, float] = field(default_factory=dict)
    paper_mode: bool = True

    def to_dict(self) -> dict:
        return {
            'mint': self.mint,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'entry_time': self.entry_time,
            'exit_time': self.exit_time,
            'sol_invested': self.sol_invested,
            'sol_received': self.sol_received,
            'pnl_sol': self.pnl_sol,
            'pnl_pct': self.pnl_pct,
            'exit_reason': self.exit_reason,
            'paper_mode': self.paper_mode,
        }


@dataclass
class TokenData:
    """Token data for signal computation"""
    mint: str
    price: float
    timestamp: float
    trades: List[Dict[str, Any]] = field(default_factory=list)
    bonding_curve_progress: float = 0.0
    liquidity_sol: float = 0.0
    market_cap_sol: float = 0.0
    creator_address: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'mint': self.mint,
            'price': self.price,
            'timestamp': self.timestamp,
            'trade_count': len(self.trades),
            'bonding_curve_progress': self.bonding_curve_progress,
            'liquidity_sol': self.liquidity_sol,
            'market_cap_sol': self.market_cap_sol,
        }
