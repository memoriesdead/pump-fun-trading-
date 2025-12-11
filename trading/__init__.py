"""
Trading Module - RenTech 1:1 Parity Architecture
================================================

Single codebase for BOTH paper and real trading.
The ONLY difference is the executor (mock vs Solana).

Usage:
    from trading import TradingEngine, TradingConfig

    # Paper trading
    engine = TradingEngine(paper_mode=True, capital=100.0)

    # Real trading (ONLY after parity validation passes)
    engine = TradingEngine(paper_mode=False, capital=100.0)

RenTech Methodology:
1. Paper and real use IDENTICAL logic
2. Any divergence = bug in code
3. Validate parity before deploying capital
"""

# Core engines
from .engine import TradingEngine
from .unified_engine import UnifiedEngine, TokenSource

# Configuration
from .config import (
    TradingConfig,
    DEFAULT_CONFIG,
    AGGRESSIVE_CONFIG,
    CONSERVATIVE_CONFIG,
)

# Data models
from .models import (
    Order,
    OrderSide,
    OrderType,
    ExecutionResult,
    Position,
    ClosedTrade,
    EntryDecision,
    ExitDecision,
    TradeDecision,
    TokenData,
)

# Executors
from .executors import PaperExecutor, SlippageModel
from .executors.real import RealExecutor

# Orchestrator
from .orchestrator import LiveTradingOrchestrator

# Signals
from .signals.signal_engine import SignalEngine, get_signal_engine


__all__ = [
    # Core
    'TradingEngine',
    'UnifiedEngine',
    'TokenSource',

    # Config
    'TradingConfig',
    'DEFAULT_CONFIG',
    'AGGRESSIVE_CONFIG',
    'CONSERVATIVE_CONFIG',

    # Models
    'Order',
    'OrderSide',
    'OrderType',
    'ExecutionResult',
    'Position',
    'ClosedTrade',
    'EntryDecision',
    'ExitDecision',
    'TradeDecision',
    'TokenData',

    # Executors
    'PaperExecutor',
    'SlippageModel',
    'RealExecutor',

    # Orchestrator
    'LiveTradingOrchestrator',

    # Signals
    'SignalEngine',
    'get_signal_engine',
]
