"""
RENTECH SIMULATION MODULE
=========================

Monte Carlo simulation + Walk-forward optimization for pump.fun signals.

Components:
- monte_carlo_engine: Full simulation orchestrator
- signal_analyzer: Detailed signal performance analysis
- walk_forward: Walk-forward optimization framework
"""

from .monte_carlo_engine import (
    SimulationOrchestrator,
    MonteCarloSimulator,
    WalkForwardOptimizer,
    SignalBacktester,
    TradeReplayEngine,
    Trade,
    TokenState,
    Position,
    ClosedTrade,
    SignalResult,
    SignalPerformance,
    WalkForwardResult,
    load_all_signals,
)

from .signal_analyzer import (
    SignalAnalyzer,
    SignalCorrelationAnalyzer,
    DetailedSignalAnalysis,
    run_full_analysis,
)

__all__ = [
    # Monte Carlo Engine
    'SimulationOrchestrator',
    'MonteCarloSimulator',
    'WalkForwardOptimizer',
    'SignalBacktester',
    'TradeReplayEngine',
    'Trade',
    'TokenState',
    'Position',
    'ClosedTrade',
    'SignalResult',
    'SignalPerformance',
    'WalkForwardResult',
    'load_all_signals',
    # Signal Analyzer
    'SignalAnalyzer',
    'SignalCorrelationAnalyzer',
    'DetailedSignalAnalysis',
    'run_full_analysis',
]
