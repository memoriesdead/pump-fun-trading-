"""
Executors Module
================

Paper and real execution implementations.
Both must have identical interfaces for RenTech 1:1 parity.

Usage:
    # Paper trading (simulation)
    from trading.executors import PaperExecutor
    executor = PaperExecutor()

    # Real trading (Solana)
    from trading.executors import RealExecutor
    executor = RealExecutor(keypair_path="...", rpc_url="...")
"""

from .paper import PaperExecutor, SlippageModel, get_paper_executor
from .real import RealExecutor

__all__ = [
    # Paper
    'PaperExecutor',
    'SlippageModel',
    'get_paper_executor',
    # Real
    'RealExecutor',
]
