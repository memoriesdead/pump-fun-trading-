"""
RENTECH PORTFOLIO MODULE
========================

Portfolio optimization + risk management for pump.fun trading.

Components:
- portfolio_optimizer: Kelly criterion + correlation-based sizing
- risk_manager: Portfolio-level risk controls
"""

from .portfolio_optimizer import (
    PortfolioOptimizer,
    KellyCriterion,
    SignalCorrelation,
    RegimeDetector,
    RiskManager,
    PositionSizing,
    PortfolioState,
    OptimalWeights,
    SignalStats,
    MarketRegime,
    create_portfolio_optimizer,
)

__all__ = [
    'PortfolioOptimizer',
    'KellyCriterion',
    'SignalCorrelation',
    'RegimeDetector',
    'RiskManager',
    'PositionSizing',
    'PortfolioState',
    'OptimalWeights',
    'SignalStats',
    'MarketRegime',
    'create_portfolio_optimizer',
]
