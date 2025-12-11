"""
Distributed Trading Module
==========================

Multi-VPS distributed trading architecture for pump.fun scalping.
Hybrid approach: pump.fun API for discovery + Solana blockchain for execution.

Components:
- HybridScalableEngine: Pump.fun WebSocket discovery + Solana execution (RECOMMENDED)
- UnbreakableEngine: Pure Solana (no pump.fun API, slower discovery)
- SolanaAnalyzer: Direct on-chain analysis (no API)
- DiscoveryNode/ExecutionNode: Legacy multi-VPS components

Architecture:
    [Pump.fun WS via Tor] --> [Token Discovery] --> [Solana Analysis] --> [Execute]

    Tor rotation provides IP diversity for pump.fun connections
    All execution goes direct to Solana (can't be rate limited)

Usage:
    # RECOMMENDED: Hybrid engine with Tor rotation
    python -m trading.distributed.hybrid_scalable_engine --rpc https://api.mainnet-beta.solana.com --tor-start 9050 --tor-count 10

    # Pure Solana engine (no pump.fun API, slower)
    python -m trading.distributed.unbreakable_engine --keypair ~/.config/solana/pumpfun.json
"""

# Core components (always available)
from .solana_analyzer import (
    SolanaAnalyzer,
    BondingCurveState,
    TokenMetrics,
    TokenWatcher,
)
from .unbreakable_engine import UnbreakableEngine, WatchedToken
from .hybrid_scalable_engine import HybridScalableEngine, DiscoveredToken

# Legacy components (optional - may have broken dependencies)
DiscoveryNode = None
ExecutionNode = None
TokenSignal = None
Position = None

try:
    from .discovery_node import DiscoveryNode, TokenSignal
except ImportError:
    pass

try:
    from .execution_node import ExecutionNode, Position
except ImportError:
    pass

__all__ = [
    # Primary engines
    'HybridScalableEngine',
    'DiscoveredToken',
    'UnbreakableEngine',
    'WatchedToken',

    # Direct Solana components (no API)
    'SolanaAnalyzer',
    'BondingCurveState',
    'TokenMetrics',
    'TokenWatcher',

    # Legacy multi-VPS components (optional)
    'DiscoveryNode',
    'ExecutionNode',
    'TokenSignal',
    'Position',
]
