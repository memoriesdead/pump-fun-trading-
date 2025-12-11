"""
RENTECH EXECUTION MODULE
========================

Direct trading execution on pump.fun bonding curve with MEV protection.

Components:
- PumpfunTrader: Buy/sell execution
- SolanaTransactionBuilder: Full transaction building
- LiveScalpingEngine: Signal-to-execution pipeline
- ExecutionEngine: Smart order routing + MEV protection
- SmartOrderRouter: Optimal venue selection (pump bonding, Raydium, Jupiter)
- MEVProtector: Jito bundles + private RPC protection

Usage:
    from trading.execution import ExecutionEngine, create_execution_engine

    engine = await create_execution_engine(
        keypair_path="~/.config/solana/pumpfun.json",
        mev_protection=MEVProtection.JITO_BUNDLE
    )
    result = await engine.execute_order(order)
"""

from .pumpfun_trader import (
    PumpfunTrader,
    PumpfunBondingCurve,
    PumpfunScalper,
    BondingCurveState,
    PUMPFUN_PROGRAM_ID,
    BUY_DISCRIMINATOR,
    SELL_DISCRIMINATOR,
)

from .solana_tx_builder import (
    SolanaTransactionBuilder,
    PumpfunInstructionBuilder,
    decode_pumpfun_instruction,
    derive_bonding_curve_pda,
    derive_user_ata,
)

from .live_scalper import (
    LiveScalpingEngine,
    LiveTokenAggregator,
    RiskManager,
    TokenState,
    Position,
)

from .execution_engine import (
    ExecutionEngine,
    SmartOrderRouter,
    MEVProtector,
    PumpBondingCurve,
    ExecutionAnalytics,
    Order,
    Quote,
    ExecutionResult,
    ExecutionStats,
    OrderType,
    OrderSide,
    ExecutionVenue,
    MEVProtection,
)


__all__ = [
    # Core trader
    'PumpfunTrader',
    'PumpfunBondingCurve',
    'PumpfunScalper',
    'BondingCurveState',

    # Transaction building
    'SolanaTransactionBuilder',
    'PumpfunInstructionBuilder',
    'decode_pumpfun_instruction',
    'derive_bonding_curve_pda',
    'derive_user_ata',

    # Live execution
    'LiveScalpingEngine',
    'LiveTokenAggregator',
    'RiskManager',
    'TokenState',
    'Position',

    # Constants
    'PUMPFUN_PROGRAM_ID',
    'BUY_DISCRIMINATOR',
    'SELL_DISCRIMINATOR',

    # Execution Engine (Smart Order Routing + MEV Protection)
    'ExecutionEngine',
    'SmartOrderRouter',
    'MEVProtector',
    'PumpBondingCurve',
    'ExecutionAnalytics',
    'Order',
    'Quote',
    'ExecutionResult',
    'ExecutionStats',
    'OrderType',
    'OrderSide',
    'ExecutionVenue',
    'MEVProtection',
]
