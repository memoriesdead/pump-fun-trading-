"""
UNIFIED TRADING ENGINE
======================

Single codebase for ALL trading:
- Pump.fun bonding curve tokens
- Raydium graduated tokens
- Paper and real modes

RenTech methodology: ONE engine, ONE set of formulas.
The ONLY differences are data sources and executors.

Usage:
    python -m trading.unified_engine --mode paper --capital 100
"""

import asyncio
import aiohttp
import time
import argparse
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto

import numpy as np

from .adaptive_config import AdaptiveFormulaConfig, AdaptiveParameters
from .models import Position
from .signals.signal_engine import SignalEngine, get_signal_engine

import sys
import os

# Force UTF-8 output on Windows via environment variable
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def safe_str(s: str, max_len: int = 20) -> str:
    """Safely convert string to ASCII-safe representation"""
    if not s:
        return "???"
    # Remove non-ASCII characters
    safe = ''.join(c if ord(c) < 128 else '?' for c in str(s))
    return safe[:max_len] if len(safe) > max_len else safe


# ============================================================
# ENUMS AND CONSTANTS
# ============================================================

class TokenSource(Enum):
    """Where the token is trading"""
    PUMPFUN = auto()      # On pump.fun bonding curve
    RAYDIUM = auto()      # Graduated to Raydium


class TradeAction(Enum):
    """Trading actions"""
    SKIP = auto()
    BUY = auto()
    SELL = auto()


# API endpoints
PUMPPORTAL_WS = "wss://pumpportal.fun/api/data"

# Dexscreener endpoints
DEXSCREENER_SEARCH = "https://api.dexscreener.com/latest/dex/search"
DEXSCREENER_TOKENS = "https://api.dexscreener.com/latest/dex/tokens"
DEXSCREENER_PAIRS = "https://api.dexscreener.com/latest/dex/pairs/solana"

# Birdeye API (comprehensive Solana token data)
BIRDEYE_TOKEN_LIST = "https://public-api.birdeye.so/defi/tokenlist"
BIRDEYE_PRICE = "https://public-api.birdeye.so/defi/price"
BIRDEYE_TRADES = "https://public-api.birdeye.so/defi/txs/token"

# Jupiter API (Solana aggregator - sees ALL tokens)
JUPITER_TOKENS = "https://token.jup.ag/all"
JUPITER_PRICE = "https://price.jup.ag/v4/price"

# Raydium direct API
RAYDIUM_PAIRS = "https://api.raydium.io/v2/main/pairs"
RAYDIUM_POOLS = "https://api.raydium.io/v2/ammV3/ammPools"

# GeckoTerminal (free, comprehensive)
GECKOTERMINAL_TRENDING = "https://api.geckoterminal.com/api/v2/networks/solana/trending_pools"
GECKOTERMINAL_NEW = "https://api.geckoterminal.com/api/v2/networks/solana/new_pools"
GECKOTERMINAL_POOLS = "https://api.geckoterminal.com/api/v2/networks/solana/dexes/raydium/pools"

# Hostinger VPS for proxy rotation (bypasses rate limits)
VPS_HOST = "217.196.50.212"
VPS_USER = "root"
VPS_PROXY = f"socks5://{VPS_HOST}:1080"  # SOCKS5 proxy on VPS

# ==========================================================================
# PURE MATHEMATICAL FILTERS - No keyword searches (RenTech methodology)
# ==========================================================================
# Filter tokens by quantitative metrics ONLY:
# - Volatility (price variance over time)
# - Momentum (directional price movement)
# - Volume/Liquidity ratios
# - Trade intensity (Hawkes process)
# - Buy/sell pressure imbalance
# No amateur keyword filtering like "pump", "meme", etc.
# ==========================================================================

# ==========================================================================
# RAPID SCALPING CONFIG - Goal: 10,000 trades
# ==========================================================================
# Simple math: If pattern shows 100% certainty of 0.75%+ gain, BUY & SELL
# No complex analysis - pure momentum + volume confirmation
# ==========================================================================

# ==========================================================================
# MICRO-ACCOUNT FRICTION-AWARE FILTERS ($10 account)
# ==========================================================================
# REALITY CHECK for small accounts:
#   - $10 trade on $1k liquidity = 6.89% friction = $0.69 GONE per trade
#   - $10 trade on $500k liquidity = 1.8% friction = $0.18 per trade
#   - MUST trade HIGH LIQUIDITY to survive
#
# From edge_measurement.py (ID 331):
#   - Only trade when: expected_profit > 2 × total_friction
#   - With $10, can only afford 2-3 losses before account is dead
#   - WAIT for 10%+ moves to maximize win size vs friction
# ==========================================================================

MATH_FILTERS = {
    # =======================================================================
    # EXTREME SELECTIVITY - Only trade when edge is overwhelming
    # Paper trading insight: Only 50%+ momentum tokens delivered 10%+ moves
    # With ~8-9% round-trip friction, need 10%+ gross move to profit
    # =======================================================================

    # Volume filters - need active trading
    'min_volume_5m': 500,           # $500+ volume in 5 min (active market)
    'min_volume_1h': 2000,          # $2k+ hourly volume (sustained interest)
    'min_liquidity': 25000,         # $25k+ liquidity (reduce slippage)
    'max_liquidity': 100_000_000,   # No upper limit

    # Volatility - need BIG movement to overcome friction
    'min_volatility_5m': 10.0,      # At least 10% move potential
    'max_volatility_5m': 500.0,     # Allow very volatile tokens

    # Momentum - MUST BE EXTREME (50%+) to have edge after friction
    'min_momentum': 50.0,           # 50%+ momentum - EXTREME SELECTIVITY
    'max_momentum': 1000.0,         # Allow parabolic moves

    # Trade intensity - need active buying
    'min_txns_5m': 5,               # At least 5 trades in 5m
    'min_buy_ratio': 0.60,          # 60% buys - clear buyer dominance

    # Age - avoid rugs but catch momentum
    'min_age': 30,                  # At least 30 seconds old
    'max_age': 86400 * 30,          # Max 30 days old
}

# ==========================================================================
# MICRO-ACCOUNT TRADING CONFIG ($10 starting capital)
# ==========================================================================
# THE BRUTAL MATH OF $10:
#   - Can only afford 2-3 losses before account is dead
#   - Each loss at 4% = $0.40 gone
#   - Need 10%+ wins to grow meaningfully
#   - PATIENCE is the only edge - wait for perfect setups
#
# From Kelly Criterion (ID 321):
#   - With micro account, bet small % per trade (10-20%)
#   - This means $1-2 per trade on $10 account
#
# From transaction_costs.py (ID 319):
#   - On $100k+ liquidity: friction = 1.7%
#   - Need 3.4%+ profit JUST to break even
#   - Target 10%+ moves for meaningful gains
# ==========================================================================

SCALP_CONFIG = {
    # =========================================================================
    # EXTREME MOMENTUM CONFIG - Ride the wave
    # Only enter on 50%+ momentum tokens - let them run
    # =========================================================================
    'target_profit_pct': 0.15,      # 15% target - need big wins to overcome friction
    'max_loss_pct': 0.05,           # 5% max loss - slightly wider for volatile tokens
    'max_hold_ms': 60000,           # 60 second max hold - capture momentum burst
    'min_hold_ms': 15000,           # 15 second minimum - let move develop
    'confidence_threshold': 0.85,   # 85% confidence (be very selective)
    'trades_per_token': 1,          # Only 1 trade per token - don't chase
    'min_expected_edge': 0.12,      # 12% minimum expected move (friction = 8-9%)
}


# ==========================================================================
# REALISTIC FRICTION MODEL - RenTech Methodology
# ==========================================================================
# Paper trading MUST include real-world costs to prove mathematical edge.
# If we can't profit AFTER friction, we can't profit in reality.
# ==========================================================================

@dataclass
class FrictionModel:
    """
    Realistic trading friction for Solana/Raydium/PumpFun.

    All values derived from empirical Solana mainnet data:
    - https://docs.jup.ag/docs/fees
    - https://solana.com/docs/core/fees
    - Actual slippage observed on pump.fun tokens
    """

    # === TRANSACTION FEES (fixed costs) ===
    solana_base_fee_sol: float = 0.000005      # Base transaction fee
    solana_priority_fee_sol: float = 0.0001    # Priority fee for fast execution
    jito_tip_sol: float = 0.0005               # Jito bundle tip (MEV protection)

    # === DEX FEES (percentage of trade) ===
    jupiter_fee_pct: float = 0.0025            # 0.25% Jupiter platform fee
    raydium_fee_pct: float = 0.0025            # 0.25% Raydium swap fee
    pumpfun_fee_pct: float = 0.01              # 1% pump.fun fee

    # === SLIPPAGE MODEL (dynamic based on liquidity) ===
    base_slippage_pct: float = 0.005           # 0.5% base slippage
    slippage_per_1k_usd: float = 0.001         # +0.1% per $1000 trade size
    max_slippage_pct: float = 0.03             # 3% max slippage cap

    # === EXECUTION FAILURE ===
    tx_failure_rate: float = 0.15              # 15% of transactions fail
    retry_cost_sol: float = 0.00015            # Cost per retry attempt
    max_retries: int = 3                       # Max retry attempts

    # === PRICE IMPACT (for larger orders) ===
    # AMM price impact: ΔP = trade_size / (liquidity * 2)
    # This follows xy=k constant product formula

    def compute_total_cost(
        self,
        trade_size_usd: float,
        liquidity_usd: float,
        is_pumpfun: bool = False,
        sol_price: float = 230.0,
    ) -> dict:
        """
        Compute total friction cost for a round-trip trade.

        Returns dict with:
        - total_cost_pct: Total cost as percentage of trade
        - breakdown: Individual cost components
        - min_profit_pct: Minimum profit needed to break even
        """
        trade_size_sol = trade_size_usd / sol_price

        # 1. Fixed transaction fees (entry + exit = 2 transactions)
        fixed_fees_sol = 2 * (
            self.solana_base_fee_sol +
            self.solana_priority_fee_sol +
            self.jito_tip_sol
        )
        fixed_fees_pct = (fixed_fees_sol * sol_price) / trade_size_usd if trade_size_usd > 0 else 0

        # 2. DEX fees (entry + exit)
        if is_pumpfun:
            dex_fee_pct = 2 * self.pumpfun_fee_pct  # 2% round-trip
        else:
            dex_fee_pct = 2 * self.raydium_fee_pct  # 0.5% round-trip

        # 3. Slippage (entry + exit)
        # More slippage on larger trades relative to liquidity
        liquidity_ratio = trade_size_usd / liquidity_usd if liquidity_usd > 0 else 1.0
        dynamic_slip = self.base_slippage_pct + (trade_size_usd / 1000) * self.slippage_per_1k_usd
        dynamic_slip = min(dynamic_slip, self.max_slippage_pct)
        # Slippage increases with liquidity ratio
        slippage_pct = 2 * dynamic_slip * (1 + liquidity_ratio)
        slippage_pct = min(slippage_pct, 2 * self.max_slippage_pct)

        # 4. Price impact (AMM constant product)
        # ΔP/P = trade_size / (2 * liquidity)
        price_impact_pct = 2 * (trade_size_usd / (2 * liquidity_usd)) if liquidity_usd > 0 else 0.02
        price_impact_pct = min(price_impact_pct, 0.05)  # Cap at 5%

        # 5. Expected retry cost
        expected_retries = self.tx_failure_rate * (1 + self.tx_failure_rate)  # ~0.17 average retries
        retry_cost_sol = expected_retries * self.retry_cost_sol * 2  # Both entry and exit
        retry_cost_pct = (retry_cost_sol * sol_price) / trade_size_usd if trade_size_usd > 0 else 0

        # Total
        total_cost_pct = (
            fixed_fees_pct +
            dex_fee_pct +
            slippage_pct +
            price_impact_pct +
            retry_cost_pct
        )

        return {
            'total_cost_pct': total_cost_pct,
            'min_profit_pct': total_cost_pct,  # Need this much profit to break even
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
    ) -> tuple:
        """
        Apply realistic friction to entry price.

        Returns:
        - actual_price: Price after slippage (worse for us)
        - tokens_received: Actual tokens after fees
        - friction_cost_usd: Total friction paid
        """
        if trade_size_usd <= 0 or intended_price <= 0:
            return intended_price, 0, 0

        # Calculate dynamic slippage
        liquidity_ratio = trade_size_usd / liquidity_usd if liquidity_usd > 0 else 0.5
        slip = self.base_slippage_pct + (trade_size_usd / 1000) * self.slippage_per_1k_usd
        slip = min(slip, self.max_slippage_pct)
        slip *= (1 + liquidity_ratio * 0.5)  # Worse slippage on low liquidity

        # Price impact
        impact = (trade_size_usd / (2 * liquidity_usd)) if liquidity_usd > 0 else 0.01
        impact = min(impact, 0.025)

        # Actual entry price is HIGHER (worse for us when buying)
        total_slip = slip + impact
        actual_price = intended_price * (1 + total_slip)

        # DEX fee reduces tokens received
        fee_pct = self.pumpfun_fee_pct if is_pumpfun else self.raydium_fee_pct
        effective_usd = trade_size_usd * (1 - fee_pct)
        tokens_received = effective_usd / actual_price if actual_price > 0 else 0

        # Calculate friction cost
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
    ) -> tuple:
        """
        Apply realistic friction to exit price.

        Returns:
        - actual_price: Price after slippage (worse for us)
        - usd_received: Actual USD after fees
        - friction_cost_usd: Total friction paid
        """
        if tokens_to_sell <= 0 or intended_price <= 0:
            return intended_price, 0, 0

        trade_size_usd = tokens_to_sell * intended_price

        # Calculate dynamic slippage
        liquidity_ratio = trade_size_usd / liquidity_usd if liquidity_usd > 0 else 0.5
        slip = self.base_slippage_pct + (trade_size_usd / 1000) * self.slippage_per_1k_usd
        slip = min(slip, self.max_slippage_pct)
        slip *= (1 + liquidity_ratio * 0.5)

        # Price impact
        impact = (trade_size_usd / (2 * liquidity_usd)) if liquidity_usd > 0 else 0.01
        impact = min(impact, 0.025)

        # Actual exit price is LOWER (worse for us when selling)
        total_slip = slip + impact
        actual_price = intended_price * (1 - total_slip)

        # DEX fee reduces USD received
        fee_pct = self.pumpfun_fee_pct if is_pumpfun else self.raydium_fee_pct
        gross_usd = tokens_to_sell * actual_price
        usd_received = gross_usd * (1 - fee_pct)

        # Transaction fees
        tx_fees_usd = (self.solana_base_fee_sol + self.solana_priority_fee_sol + self.jito_tip_sol) * sol_price
        usd_received -= tx_fees_usd

        # Calculate friction cost
        ideal_usd = tokens_to_sell * intended_price
        friction_cost_usd = ideal_usd - usd_received

        return actual_price, max(0, usd_received), max(0, friction_cost_usd)

    def simulate_tx_success(self) -> tuple:
        """
        Simulate transaction success/failure.

        Returns:
        - success: Whether transaction succeeded
        - retries: Number of retries needed
        - extra_cost_sol: Additional cost from retries
        """
        import random

        retries = 0
        success = False
        extra_cost = 0.0

        for attempt in range(self.max_retries + 1):
            if random.random() > self.tx_failure_rate:
                success = True
                break
            retries += 1
            extra_cost += self.retry_cost_sol

        return success, retries, extra_cost


# Global friction model instance
FRICTION = FrictionModel()


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Token:
    """Unified token representation"""
    mint: str
    symbol: str
    name: str
    source: TokenSource

    # Price
    price_usd: float = 0.0
    price_sol: float = 0.0

    # Liquidity
    liquidity_usd: float = 0.0
    liquidity_sol: float = 0.0

    # Bonding curve (pump.fun only)
    bonding_progress: float = 0.0

    # Volume
    volume_5m: float = 0.0
    volume_1h: float = 0.0
    volume_24h: float = 0.0

    # Price changes
    change_5m: float = 0.0
    change_1h: float = 0.0

    # Trades
    buys_5m: int = 0
    sells_5m: int = 0

    # Extra data for signal computation
    pair_address: str = ""
    recent_trades: List[Dict] = field(default_factory=list)

    # Timestamps
    created_at: float = 0.0
    last_update: float = 0.0

    def score(self) -> float:
        """Unified scoring for any token"""
        s = 0.0

        # Volume score
        if self.volume_5m > 0:
            s += min(10, self.volume_5m / 1000)

        # Momentum score
        if self.change_5m > 0:
            s += min(5, self.change_5m / 2)

        # Buy pressure
        total = self.buys_5m + self.sells_5m
        if total > 0:
            buy_ratio = self.buys_5m / total
            s += (buy_ratio - 0.5) * 6

        # Liquidity
        if self.liquidity_usd > 50000:
            s += 3
        elif self.liquidity_usd > 20000:
            s += 2
        elif self.liquidity_usd > 10000:
            s += 1

        # Bonding progress bonus (near graduation)
        if self.source == TokenSource.PUMPFUN:
            if 0.7 <= self.bonding_progress < 0.85:
                s += 5  # Near graduation bonus
            elif self.bonding_progress >= 0.85:
                s += 3  # About to graduate

        return s


@dataclass
class TradeResult:
    """Result of a trade decision"""
    action: TradeAction
    mint: str
    reason: str
    size_sol: float = 0.0
    price: float = 0.0
    pnl_pct: float = 0.0
    pnl_usd: float = 0.0
    signals: Dict[int, float] = field(default_factory=dict)


@dataclass
class EngineStats:
    """Comprehensive trading statistics"""
    start_time: float = field(default_factory=time.time)
    tokens_scanned: int = 0
    tokens_qualified: int = 0
    unique_tokens_seen: int = 0
    entries: int = 0
    exits: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl_usd: float = 0.0
    total_pnl_sol: float = 0.0

    # Volume tracking
    total_volume_traded_usd: float = 0.0
    total_volume_traded_sol: float = 0.0

    # Best/worst trades
    best_trade_pnl: float = 0.0
    best_trade_token: str = ""
    worst_trade_pnl: float = 0.0
    worst_trade_token: str = ""

    # Timing stats
    total_hold_time_ms: float = 0.0
    fastest_exit_ms: float = float('inf')
    slowest_exit_ms: float = 0.0

    # API stats
    api_calls: int = 0
    api_errors: int = 0

    # ========== FRICTION TRACKING (RenTech Reality Check) ==========
    # These stats show the REAL cost of trading - paper must match real
    total_friction_usd: float = 0.0           # Total friction paid
    total_slippage_usd: float = 0.0           # Slippage portion
    total_fees_usd: float = 0.0               # Fees portion
    total_price_impact_usd: float = 0.0       # Price impact portion
    failed_txns: int = 0                      # Simulated failed transactions
    retry_costs_usd: float = 0.0              # Cost of retries
    gross_pnl_usd: float = 0.0                # PnL before friction
    net_pnl_usd: float = 0.0                  # PnL after friction (this is reality)

    @property
    def uptime(self) -> str:
        secs = int(time.time() - self.start_time)
        h, r = divmod(secs, 3600)
        m, s = divmod(r, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    @property
    def avg_hold_time_ms(self) -> float:
        return self.total_hold_time_ms / self.exits if self.exits > 0 else 0.0

    @property
    def friction_pct(self) -> float:
        """Friction as percentage of volume traded"""
        return (self.total_friction_usd / self.total_volume_traded_usd * 100) if self.total_volume_traded_usd > 0 else 0.0

    @property
    def edge_after_friction(self) -> float:
        """True edge = (Net PnL / Volume) - this is what RenTech cares about"""
        return (self.net_pnl_usd / self.total_volume_traded_usd * 100) if self.total_volume_traded_usd > 0 else 0.0


# ============================================================
# UNIFIED TRADING ENGINE
# ============================================================

class UnifiedEngine:
    """
    Single trading engine for all token types and modes.

    Formula-driven: All decisions computed by mathematical formulas.
    Unified: Same logic for pump.fun and Raydium tokens.
    Modular: Easy to extend with new data sources.
    """

    def __init__(
        self,
        paper_mode: bool = True,
        capital: float = 100.0,
        max_positions: int = 3,
        sources: List[TokenSource] = None,
    ):
        """
        Initialize unified engine.

        Args:
            paper_mode: True for paper trading
            capital: Starting capital in SOL
            max_positions: Max concurrent positions
            sources: Token sources to trade (default: all)
        """
        self.paper_mode = paper_mode
        self.capital = capital
        self.initial_capital = capital
        self.max_positions = 10  # Increased for rapid trading
        self.sources = sources or [TokenSource.RAYDIUM, TokenSource.PUMPFUN]
        self.sol_price = 230.0  # USD/SOL

        # Core components
        self.signal_engine = get_signal_engine()

        # State
        self.running = False
        self.stats = EngineStats()
        self.tokens: Dict[str, Token] = {}
        self.positions: Dict[str, Position] = {}
        self.position_configs: Dict[str, AdaptiveFormulaConfig] = {}

        # FRICTION-AWARE MODE - Based on formula analysis (IDs 317-319, 331)
        # Key insight: FEWER high-quality trades beat MANY low-quality trades
        self.scalp_mode = False  # Disable rapid scalping - doesn't beat friction
        self.target_profit = SCALP_CONFIG['target_profit_pct']  # 8% (2x friction)
        self.max_loss = SCALP_CONFIG['max_loss_pct']  # 4% (matches friction)
        self.max_hold_ms = SCALP_CONFIG['max_hold_ms']  # 2 minutes (bigger moves)
        self.min_hold_ms = SCALP_CONFIG['min_hold_ms']  # 5 seconds minimum
        self.min_confidence = SCALP_CONFIG['confidence_threshold']  # 75%
        self.min_expected_edge = SCALP_CONFIG['min_expected_edge']  # 6% min edge

        # STRICT thresholds - only trade when edge > 2x friction
        self.max_rug_score = 0.3  # Stricter rug avoidance
        self.min_quality_score = 5  # Higher quality bar
        self.min_score = 8.0  # Much higher score requirement
        self.min_liquidity = MATH_FILTERS['min_liquidity']  # Use filter value ($20k)

        # Trade tracking
        self.trade_count_per_token: Dict[str, int] = {}
        self.last_trade_time: Dict[str, float] = {}

        # Loss tracking - avoid re-entering losing positions
        self.token_losses: Dict[str, int] = {}  # Count of losses per token
        self.token_cooldown: Dict[str, float] = {}  # Cooldown timer after loss
        self.loss_cooldown_sec = 30  # 30 second cooldown after loss on a token

    async def start(self):
        """Start the unified trading engine"""
        self.running = True
        self.stats = EngineStats()

        mode = 'PAPER' if self.paper_mode else 'REAL'
        sources_str = ', '.join(s.name for s in self.sources)

        print(f"""
+======================================================================+
|                    UNIFIED TRADING ENGINE                             |
+======================================================================+
|  Mode:       {mode:10}                                             |
|  Capital:    {self.capital:>8.2f} SOL (${self.capital * self.sol_price:,.0f})                      |
|  Sources:    {sources_str:40}          |
|  Max Pos:    {self.max_positions:3}                                                |
|  Time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):19}                      |
+======================================================================+
        """)

        # Start all loops
        tasks = [
            self._position_monitor(),
            self._status_loop(),
        ]

        # Add source-specific scanners
        if TokenSource.RAYDIUM in self.sources:
            tasks.append(self._raydium_scanner())
        if TokenSource.PUMPFUN in self.sources:
            tasks.append(self._pumpfun_scanner())

        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop(self):
        """Stop the engine"""
        self.running = False
        self._print_report()

    # ============================================================
    # DATA SOURCE SCANNERS
    # ============================================================

    async def _raydium_scanner(self):
        """Scan Raydium for graduated pump.fun tokens"""
        logger.info("Starting Raydium scanner...")

        while self.running:
            try:
                tokens = await self._fetch_raydium_tokens()

                for token in tokens:
                    self.tokens[token.mint] = token

                    if await self._should_enter(token):
                        await self._enter_position(token)

            except Exception as e:
                logger.error(f"Raydium scan error: {e}")

            await asyncio.sleep(3)  # Faster scanning for rapid trades

    async def _pumpfun_scanner(self):
        """Scan pump.fun for new tokens"""
        logger.info("Starting PumpFun scanner...")

        while self.running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(PUMPPORTAL_WS) as ws:
                        # Subscribe to new tokens
                        await ws.send_json({
                            "method": "subscribeNewToken"
                        })

                        async for msg in ws:
                            if not self.running:
                                break

                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = msg.json()
                                token = self._parse_pumpfun_token(data)
                                if token:
                                    self.tokens[token.mint] = token

                                    if await self._should_enter(token):
                                        await self._enter_position(token)

            except Exception as e:
                logger.error(f"PumpFun scan error: {e}")
                await asyncio.sleep(5)

    async def _fetch_raydium_tokens(self) -> List[Token]:
        """
        Fetch ALL Raydium tokens using PURE MATHEMATICAL filtering.

        NO keyword searches - only quantitative filters:
        - Volume, liquidity, volatility, momentum, trade intensity

        Data sources (parallel fetch for speed):
        1. GeckoTerminal - new pools endpoint
        2. GeckoTerminal - trending pools
        3. Dexscreener - new pairs
        """
        all_tokens = {}  # Dedupe by mint

        async with aiohttp.ClientSession() as session:
            # Parallel fetch from multiple sources - PURE MATH, NO KEYWORD SEARCH
            tasks = [
                self._fetch_geckoterminal_new(session),
                self._fetch_geckoterminal_trending(session),
                self._fetch_geckoterminal_raydium_pools(session),
                self._fetch_dexscreener_boosted(session),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, list):
                    for token in result:
                        if token.mint and token.mint not in all_tokens:
                            # Apply PURE MATH filters
                            if self._passes_math_filters(token):
                                all_tokens[token.mint] = token

        tokens = list(all_tokens.values())
        self.stats.tokens_scanned = len(tokens)

        if tokens:
            logger.info(f"[MATH] Found {len(tokens)} tokens passing mathematical filters")

        return tokens

    def _passes_math_filters(self, token: Token) -> bool:
        """
        Pure mathematical filtering - RenTech methodology.
        NO keyword matching, only quantitative metrics.
        """
        f = MATH_FILTERS

        # Volume check
        if token.volume_5m < f['min_volume_5m']:
            return False
        if token.volume_1h < f['min_volume_1h']:
            return False

        # Liquidity check
        if token.liquidity_usd < f['min_liquidity']:
            return False
        if token.liquidity_usd > f['max_liquidity']:
            return False

        # Volatility check (absolute value)
        abs_change = abs(token.change_5m)
        if abs_change < f['min_volatility_5m']:
            return False
        if abs_change > f['max_volatility_5m']:
            return False

        # Momentum bounds
        if token.change_5m < f['min_momentum']:
            return False
        if token.change_5m > f['max_momentum']:
            return False

        # Trade intensity
        total_txns = token.buys_5m + token.sells_5m
        if total_txns < f['min_txns_5m']:
            return False

        # Buy pressure
        if total_txns > 0:
            buy_ratio = token.buys_5m / total_txns
            if buy_ratio < f['min_buy_ratio']:
                return False

        return True

    async def _fetch_geckoterminal_new(self, session: aiohttp.ClientSession) -> List[Token]:
        """Fetch new pools from GeckoTerminal"""
        tokens = []
        try:
            async with session.get(
                GECKOTERMINAL_NEW,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for pool in data.get('data', []):
                        token = self._parse_geckoterminal_pool(pool)
                        if token:
                            tokens.append(token)
        except Exception as e:
            logger.debug(f"GeckoTerminal new error: {e}")
        return tokens

    async def _fetch_geckoterminal_trending(self, session: aiohttp.ClientSession) -> List[Token]:
        """Fetch trending pools from GeckoTerminal"""
        tokens = []
        try:
            async with session.get(
                GECKOTERMINAL_TRENDING,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for pool in data.get('data', []):
                        token = self._parse_geckoterminal_pool(pool)
                        if token:
                            tokens.append(token)
        except Exception as e:
            logger.debug(f"GeckoTerminal trending error: {e}")
        return tokens

    async def _fetch_geckoterminal_raydium_pools(self, session: aiohttp.ClientSession) -> List[Token]:
        """Fetch Raydium pools directly from GeckoTerminal"""
        tokens = []
        try:
            # Fetch multiple pages
            for page in range(1, 4):  # 3 pages = ~300 pools
                url = f"{GECKOTERMINAL_POOLS}?page={page}"
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for pool in data.get('data', []):
                            token = self._parse_geckoterminal_pool(pool)
                            if token:
                                tokens.append(token)
                await asyncio.sleep(0.3)  # Rate limit
        except Exception as e:
            logger.debug(f"GeckoTerminal Raydium error: {e}")
        return tokens

    def _parse_geckoterminal_pool(self, pool: Dict) -> Optional[Token]:
        """Parse GeckoTerminal pool data into Token"""
        try:
            attrs = pool.get('attributes', {})
            relationships = pool.get('relationships', {})

            # Get base token address from relationships (format: "solana_ADDRESS")
            base_token = relationships.get('base_token', {}).get('data', {})
            token_id = base_token.get('id', '')
            mint = token_id.replace('solana_', '') if token_id.startswith('solana_') else token_id

            if not mint:
                return None

            # Get DEX type
            dex_data = relationships.get('dex', {}).get('data', {})
            dex_id = dex_data.get('id', '')

            # Parse numeric values safely
            def safe_float(val, default=0.0):
                try:
                    return float(val) if val else default
                except:
                    return default

            def safe_int(val, default=0):
                try:
                    return int(val) if val else default
                except:
                    return default

            price_change = attrs.get('price_change_percentage', {})
            volume = attrs.get('volume_usd', {})
            txns = attrs.get('transactions', {})
            m5_txns = txns.get('m5', {})

            # Extract symbol from pool name (e.g., "MIM / SOL" -> "MIM")
            name = attrs.get('name', '')
            symbol = name.split(' / ')[0] if ' / ' in name else name.split('/')[0] if '/' in name else name[:10]

            # Determine source based on dex
            source = TokenSource.RAYDIUM if 'raydium' in dex_id.lower() else TokenSource.PUMPFUN

            return Token(
                mint=mint,
                symbol=symbol.strip() if symbol else '???',
                name=name,
                source=source,
                pair_address=attrs.get('address', ''),
                price_usd=safe_float(attrs.get('base_token_price_usd')),
                liquidity_usd=safe_float(attrs.get('reserve_in_usd')),
                volume_5m=safe_float(volume.get('m5')),
                volume_1h=safe_float(volume.get('h1')),
                change_5m=safe_float(price_change.get('m5')),
                change_1h=safe_float(price_change.get('h1')),
                buys_5m=safe_int(m5_txns.get('buys')),
                sells_5m=safe_int(m5_txns.get('sells')),
                last_update=time.time(),
            )
        except Exception as e:
            return None

    async def _fetch_dexscreener_boosted(self, session: aiohttp.ClientSession) -> List[Token]:
        """Fetch boosted/trending tokens from Dexscreener"""
        tokens = []
        try:
            # Use boosted tokens endpoint (active tokens)
            async with session.get(
                "https://api.dexscreener.com/token-boosts/latest/v1",
                timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()

                    for item in data:
                        if item.get('chainId') != 'solana':
                            continue

                        mint = item.get('tokenAddress', '')
                        if not mint:
                            continue

                        # Fetch full token data
                        try:
                            async with session.get(
                                f"{DEXSCREENER_TOKENS}/{mint}",
                                timeout=aiohttp.ClientTimeout(total=5)
                            ) as tresp:
                                if tresp.status == 200:
                                    tdata = await tresp.json()
                                    pairs = tdata.get('pairs', [])
                                    # Find Raydium pair
                                    for pair in pairs:
                                        if pair.get('dexId', '').lower() in ['raydium', 'raydium-clmm']:
                                            token = Token(
                                                mint=mint,
                                                symbol=pair.get('baseToken', {}).get('symbol', '???'),
                                                name=pair.get('baseToken', {}).get('name', ''),
                                                source=TokenSource.RAYDIUM,
                                                pair_address=pair.get('pairAddress', ''),
                                                price_usd=float(pair.get('priceUsd', 0) or 0),
                                                liquidity_usd=float(pair.get('liquidity', {}).get('usd', 0) or 0),
                                                volume_5m=float(pair.get('volume', {}).get('m5', 0) or 0),
                                                volume_1h=float(pair.get('volume', {}).get('h1', 0) or 0),
                                                change_5m=float(pair.get('priceChange', {}).get('m5', 0) or 0),
                                                change_1h=float(pair.get('priceChange', {}).get('h1', 0) or 0),
                                                buys_5m=int(pair.get('txns', {}).get('m5', {}).get('buys', 0) or 0),
                                                sells_5m=int(pair.get('txns', {}).get('m5', {}).get('sells', 0) or 0),
                                                last_update=time.time(),
                                            )
                                            tokens.append(token)
                                            break
                        except:
                            pass
                        await asyncio.sleep(0.1)  # Rate limit
        except Exception as e:
            logger.debug(f"Dexscreener boosted error: {e}")
        return tokens

    def _parse_pumpfun_token(self, data: Dict) -> Optional[Token]:
        """Parse pump.fun WebSocket data into Token"""
        try:
            mint = data.get('mint', '')
            if not mint:
                return None

            return Token(
                mint=mint,
                symbol=data.get('symbol', '???'),
                name=data.get('name', ''),
                source=TokenSource.PUMPFUN,
                price_sol=float(data.get('price', 0) or 0),
                price_usd=float(data.get('price', 0) or 0) * self.sol_price,
                liquidity_sol=float(data.get('vSolInBondingCurve', 0) or 0),
                liquidity_usd=float(data.get('vSolInBondingCurve', 0) or 0) * self.sol_price,
                bonding_progress=float(data.get('bondingCurveProgress', 0) or 0),
                last_update=time.time(),
            )
        except:
            return None

    # ============================================================
    # RAPID SCALP PATTERN DETECTION
    # ============================================================

    def _detect_scalp_pattern(self, token: Token) -> Tuple[bool, float, str]:
        """
        Simple math pattern detection for 0.75% guaranteed profit.

        Returns: (should_trade, confidence, reason)

        Patterns we look for:
        1. Momentum burst: 5m change > 5% with buy pressure > 60%
        2. Volume spike: 5m volume > 2x average with positive momentum
        3. Buy pressure: Buys > 2x sells in last 5 minutes
        4. Price acceleration: Positive change with high txn count
        """
        confidence = 0.0
        reasons = []

        # FILTER: Skip tokens with negative momentum (they're dumping)
        if token.change_5m < -1.0:
            return False, 0.0, "negative_momentum"

        # FILTER: Skip tokens with sell pressure
        total_txns = token.buys_5m + token.sells_5m
        if total_txns > 0:
            buy_ratio = token.buys_5m / total_txns
            if buy_ratio < 0.4:  # More sells than buys
                return False, 0.0, "sell_pressure"

        # Pattern 1: Strong momentum with buy pressure
        if token.change_5m > 5.0:
            confidence += 0.3
            reasons.append("momentum")

            if total_txns > 0:
                buy_ratio = token.buys_5m / total_txns
                if buy_ratio > 0.6:
                    confidence += 0.25
                    reasons.append("buy_pressure")

        # Pattern 2: Volume spike with positive price
        if token.volume_5m > 1000 and token.change_5m > 0:
            confidence += 0.2
            reasons.append("volume")

        # Pattern 3: Strong buy pressure (2:1 or better)
        if token.buys_5m > token.sells_5m * 2 and token.buys_5m >= 5:
            confidence += 0.3
            reasons.append("buy_dominance")

        # Pattern 4: High transaction count with positive movement
        total_txns = token.buys_5m + token.sells_5m
        if total_txns >= 20 and token.change_5m > 2.0:
            confidence += 0.2
            reasons.append("activity")

        # Pattern 5: Fresh pump (very recent positive movement)
        if token.change_5m > 10.0 and token.buys_5m > token.sells_5m:
            confidence += 0.25
            reasons.append("fresh_pump")

        # Cap at 1.0
        confidence = min(1.0, confidence)

        should_trade = confidence >= self.min_confidence
        reason = "+".join(reasons) if reasons else "none"

        return should_trade, confidence, reason

    # ============================================================
    # ENTRY DECISION (RAPID SCALP MODE)
    # ============================================================

    async def _should_enter(self, token: Token) -> bool:
        """
        FRICTION-AWARE ENTRY - Based on formula IDs 317-319, 331

        Key insight from edge_measurement.py (ID 331):
        - Only trade when: expected_profit > 2 × total_friction
        - This ensures positive edge AFTER all real-world costs

        From transaction_costs.py (ID 319):
        - Calculate ACTUAL friction for this specific trade
        - Compare to expected profit from momentum signals
        """
        # Basic filters
        if len(self.positions) >= self.max_positions:
            return False

        if token.mint in self.positions:
            return False

        # CRITICAL: Minimum liquidity for low slippage (from formula analysis)
        if token.liquidity_usd < self.min_liquidity:
            return False

        # Rate limit per token (max 3 trades/min - from SCALP_CONFIG)
        now = time.time()
        last_trade = self.last_trade_time.get(token.mint, 0)
        if now - last_trade < 20:  # 20 seconds between trades (was 0.6s)
            return False

        # Cooldown after loss - avoid re-entering losing positions
        cooldown_until = self.token_cooldown.get(token.mint, 0)
        if now < cooldown_until:
            return False

        # Skip tokens with too many losses
        losses = self.token_losses.get(token.mint, 0)
        if losses >= 2:  # Max 2 losses per token (was 3)
            return False

        # ========== FRICTION-AWARE EDGE CHECK (ID 331) ==========
        # Calculate expected friction for this trade
        trade_size_usd = 0.5 * self.sol_price  # Typical trade size
        is_pumpfun = token.source == TokenSource.PUMPFUN

        friction_info = FRICTION.compute_total_cost(
            trade_size_usd=trade_size_usd,
            liquidity_usd=token.liquidity_usd,
            is_pumpfun=is_pumpfun,
            sol_price=self.sol_price,
        )

        total_friction_pct = friction_info['total_cost_pct'] * 100  # Convert to %

        # Expected profit = current momentum (change_5m is already %)
        expected_profit_pct = abs(token.change_5m) if token.change_5m > 0 else 0

        # CRITICAL CHECK: expected_profit must be > 2x friction
        # This is the RenTech methodology - prove mathematical edge BEFORE trading
        required_profit = total_friction_pct * 2  # 2x safety margin

        if expected_profit_pct < required_profit:
            # Edge is negative after friction - SKIP
            return False

        # ========== QUALITY FILTERS ==========
        # Score-based entry with high bar
        score = token.score()
        if score < self.min_score:
            return False

        # Check pattern for additional confirmation
        should_trade, confidence, reason = self._detect_scalp_pattern(token)

        # Require EITHER high confidence pattern OR very high score
        if should_trade and confidence >= self.min_confidence:
            self.stats.tokens_qualified += 1
            logger.info(f"[EDGE+] {token.symbol}: Expected {expected_profit_pct:.1f}% > Required {required_profit:.1f}% (friction={total_friction_pct:.1f}%)")
            return True

        if score >= self.min_score + 5:  # Score 13+ bypasses pattern check
            self.stats.tokens_qualified += 1
            logger.info(f"[SCORE+] {token.symbol}: Score {score:.1f}, Edge {expected_profit_pct:.1f}% > {required_profit:.1f}%")
            return True

        return False

    # ============================================================
    # POSITION MANAGEMENT
    # ============================================================

    async def _enter_position(self, token: Token):
        """Enter a position with formula-driven sizing and REALISTIC FRICTION"""
        # Create adaptive config for this position
        adaptive = AdaptiveFormulaConfig()

        # Initialize with price variance
        for i in range(20):
            var = (token.change_5m or 1) / 100
            price = token.price_usd * (1 + var * (i - 10) / 10)
            adaptive.update(
                price=price,
                volume=token.volume_5m / 20 if token.volume_5m else 10,
                timestamp=time.time() * 1000 - (20 - i) * 500,
            )

        params = adaptive.compute_all()

        # Formula-driven position size
        # For micro accounts ($10 = 0.043 SOL), allow full capital per trade
        size_sol = min(
            self.capital * params.position_size_pct,
            self.capital * 0.50,  # Max 50% per position (allows 2 concurrent trades)
            2.0,  # Max 2 SOL
        )
        # Minimum trade: 0.01 SOL (~$2.30) to support micro accounts
        size_sol = max(0.01, size_sol)

        if size_sol > self.capital:
            return

        size_usd = size_sol * self.sol_price
        is_pumpfun = token.source == TokenSource.PUMPFUN

        # ========== APPLY REALISTIC FRICTION (RenTech Reality) ==========
        # Simulate transaction success/failure
        tx_success, retries, retry_cost_sol = FRICTION.simulate_tx_success()

        if not tx_success:
            # Transaction failed after max retries - this happens 15% of time
            self.stats.failed_txns += 1
            self.stats.retry_costs_usd += retry_cost_sol * self.sol_price
            logger.warning(f"[FRICTION] TX FAILED for {token.symbol} after {retries} retries")
            return

        # Apply entry friction (slippage, fees, price impact)
        actual_entry_price, actual_tokens, entry_friction_usd = FRICTION.apply_entry_friction(
            intended_price=token.price_usd,
            trade_size_usd=size_usd,
            liquidity_usd=token.liquidity_usd,
            is_pumpfun=is_pumpfun,
        )

        # Add retry costs to friction
        entry_friction_usd += retry_cost_sol * self.sol_price

        # Track friction in stats
        self.stats.total_friction_usd += entry_friction_usd
        self.stats.retry_costs_usd += retry_cost_sol * self.sol_price

        # Create position with friction-adjusted values
        position = Position(
            mint=token.mint,
            entry_price=token.price_usd,           # Intended price (for display)
            actual_entry_price=actual_entry_price,  # Actual price after slippage
            entry_time=time.time(),
            tokens=size_usd / token.price_usd if token.price_usd > 0 else 0,  # Intended tokens
            actual_tokens=actual_tokens,            # Actual tokens received
            sol_invested=size_sol,
            entry_signals={},
            paper_mode=self.paper_mode,
            entry_friction_usd=entry_friction_usd,
            liquidity_at_entry=token.liquidity_usd,
        )

        self.positions[token.mint] = position
        self.position_configs[token.mint] = adaptive
        self.capital -= size_sol
        self.stats.entries += 1
        self.stats.total_volume_traded_sol += size_sol
        self.stats.total_volume_traded_usd += size_usd

        # Track trade timing
        self.last_trade_time[token.mint] = time.time()
        self.trade_count_per_token[token.mint] = self.trade_count_per_token.get(token.mint, 0) + 1

        # Get pattern info
        _, confidence, pattern = self._detect_scalp_pattern(token)
        score = token.score()

        # Calculate friction percentage
        friction_pct = (entry_friction_usd / size_usd * 100) if size_usd > 0 else 0

        print(f"""
======================================================================
  BUY | {datetime.now().strftime('%H:%M:%S')} | {safe_str(token.symbol, 12)} [{token.source.name}]
======================================================================
  Mint:        {token.mint[:16]}...
  Size:        {size_sol:.4f} SOL (${size_usd:.2f})
  Price:       ${token.price_usd:.8f} -> ${actual_entry_price:.8f} (after slip)
  Tokens:      {actual_tokens:,.0f} (after fees)
  -------------------------------------------
  FRICTION:    ${entry_friction_usd:.4f} ({friction_pct:.2f}%)
  Slippage:    {((actual_entry_price - token.price_usd) / token.price_usd * 100):.2f}%
  Retries:     {retries}
  -------------------------------------------
  Score:       {score:.1f}
  Liquidity:   ${token.liquidity_usd:,.0f}
  Vol/5m:      ${token.volume_5m:,.0f}
  Change/5m:   {token.change_5m:+.1f}%
  Buys/Sells:  {token.buys_5m}/{token.sells_5m} (5m)
  -------------------------------------------
  FORMULA PARAMS:
  Hold Time:   {params.optimal_hold_time_ms:.0f} ms
  Stop Loss:   {params.stop_loss_pct*100:.1f}%
  Take Profit: {params.take_profit_pct*100:.1f}%
  Regime:      {params.regime}
======================================================================
""")

    async def _exit_position(self, mint: str, token: Token, position: Position, reason: str):
        """Exit a position with REALISTIC FRICTION"""
        is_pumpfun = token.source == TokenSource.PUMPFUN
        hold_ms = (time.time() - position.entry_time) * 1000

        # ========== APPLY EXIT FRICTION (RenTech Reality) ==========
        # Simulate transaction success/failure for exit
        tx_success, retries, retry_cost_sol = FRICTION.simulate_tx_success()

        if not tx_success:
            # Exit TX failed - this is bad, we're stuck in position
            # In reality, we'd retry more aggressively for exits
            self.stats.failed_txns += 1
            self.stats.retry_costs_usd += retry_cost_sol * self.sol_price
            logger.warning(f"[FRICTION] EXIT TX FAILED for {token.symbol} - will retry next cycle")
            return  # Stay in position, will try again

        # Use actual tokens from position (after entry friction)
        actual_tokens = position.actual_tokens if position.actual_tokens > 0 else position.tokens
        liquidity = position.liquidity_at_entry if position.liquidity_at_entry > 0 else token.liquidity_usd

        # Apply exit friction (slippage, fees, price impact)
        actual_exit_price, usd_received, exit_friction_usd = FRICTION.apply_exit_friction(
            intended_price=token.price_usd,
            tokens_to_sell=actual_tokens,
            liquidity_usd=liquidity,
            is_pumpfun=is_pumpfun,
            sol_price=self.sol_price,
        )

        # Add retry costs
        exit_friction_usd += retry_cost_sol * self.sol_price

        # Track friction
        self.stats.total_friction_usd += exit_friction_usd
        total_friction = position.entry_friction_usd + exit_friction_usd

        # Calculate GROSS PnL (before friction) - what we WOULD have made
        intended_value = actual_tokens * token.price_usd
        entry_cost = position.sol_invested * self.sol_price
        gross_pnl_usd = intended_value - entry_cost

        # Calculate NET PnL (after friction) - what we ACTUALLY made
        net_pnl_usd = usd_received - entry_cost

        # SOL amounts
        sol_received = usd_received / self.sol_price
        pnl_sol = sol_received - position.sol_invested

        # Track both gross and net
        self.stats.gross_pnl_usd += gross_pnl_usd
        self.stats.net_pnl_usd += net_pnl_usd
        self.stats.total_pnl_usd += net_pnl_usd  # Use NET for main PnL
        self.stats.total_pnl_sol += pnl_sol

        # Update capital with ACTUAL received (after friction)
        self.capital += sol_received
        self.stats.exits += 1
        self.stats.total_volume_traded_sol += position.sol_invested
        self.stats.total_volume_traded_usd += position.sol_invested * self.sol_price

        # Track timing
        self.stats.total_hold_time_ms += hold_ms
        if hold_ms < self.stats.fastest_exit_ms:
            self.stats.fastest_exit_ms = hold_ms
        if hold_ms > self.stats.slowest_exit_ms:
            self.stats.slowest_exit_ms = hold_ms

        # Track best/worst (using NET PnL - reality)
        if net_pnl_usd > self.stats.best_trade_pnl:
            self.stats.best_trade_pnl = net_pnl_usd
            self.stats.best_trade_token = token.symbol
        if net_pnl_usd < self.stats.worst_trade_pnl:
            self.stats.worst_trade_pnl = net_pnl_usd
            self.stats.worst_trade_token = token.symbol

        # Win/loss based on NET PnL (after friction)
        if net_pnl_usd >= 0:
            self.stats.wins += 1
        else:
            self.stats.losses += 1
            # Track loss and set cooldown for this token
            self.token_losses[mint] = self.token_losses.get(mint, 0) + 1
            self.token_cooldown[mint] = time.time() + self.loss_cooldown_sec

        del self.positions[mint]
        if mint in self.position_configs:
            del self.position_configs[mint]

        # Colors for display
        c = '\033[32m' if net_pnl_usd >= 0 else '\033[31m'
        y = '\033[33m'  # Yellow for friction
        r = '\033[0m'

        # Calculate net PnL percentage
        net_pnl_pct = (net_pnl_usd / entry_cost) if entry_cost > 0 else 0
        gross_pnl_pct = (gross_pnl_usd / entry_cost) if entry_cost > 0 else 0

        print(f"""
======================================================================
  SELL | {datetime.now().strftime('%H:%M:%S')} | {safe_str(token.symbol, 12)} [{token.source.name}]
======================================================================
  GROSS PnL:  ${gross_pnl_usd:+.2f} ({gross_pnl_pct:+.1%}) [before friction]
  {y}FRICTION:   -${total_friction:.4f} (entry: ${position.entry_friction_usd:.4f} + exit: ${exit_friction_usd:.4f}){r}
  {c}NET PnL:    ${net_pnl_usd:+.2f} ({net_pnl_pct:+.1%}) [REALITY]{r}
  -------------------------------------------
  Hold Time:  {hold_ms:.0f} ms ({hold_ms/1000:.2f}s)
  Reason:     {reason}
  -------------------------------------------
  Entry:      ${position.entry_price:.8f} -> ${position.actual_entry_price:.8f}
  Exit:       ${token.price_usd:.8f} -> ${actual_exit_price:.8f}
  Slippage:   Entry +{((position.actual_entry_price - position.entry_price) / position.entry_price * 100):.2f}% / Exit -{((token.price_usd - actual_exit_price) / token.price_usd * 100):.2f}%
  -------------------------------------------
  Capital:    {self.capital:.4f} SOL (${self.capital * self.sol_price:,.2f})
======================================================================
""")

    async def _position_monitor(self):
        """Monitor and exit positions based on formula-driven timing"""
        while self.running:
            try:
                now_ms = time.time() * 1000

                for mint, position in list(self.positions.items()):
                    token = self.tokens.get(mint)
                    if not token:
                        continue

                    # Update price - REAL DATA EVERY 100ms for rapid scalping
                    if time.time() - token.last_update > 0.1:
                        await self._update_token_price(token)

                    if token.price_usd <= 0:
                        continue

                    # Get adaptive config
                    adaptive = self.position_configs.get(mint)
                    if not adaptive:
                        continue

                    # Update with new price
                    adaptive.update(
                        price=token.price_usd,
                        volume=token.volume_5m / 60 if token.volume_5m else 1,
                        timestamp=now_ms,
                    )

                    # Check exit
                    should_exit, reason, _ = adaptive.should_exit(
                        entry_price=position.entry_price,
                        entry_time_ms=position.entry_time * 1000,
                        current_price=token.price_usd,
                        current_time_ms=now_ms,
                    )

                    if should_exit:
                        await self._exit_position(mint, token, position, reason)

            except Exception as e:
                logger.error(f"Monitor error: {e}")

            await asyncio.sleep(0.1)  # 100ms for rapid exits

    async def _update_token_price(self, token: Token):
        """Update token price from appropriate source"""
        if token.source == TokenSource.RAYDIUM:
            await self._update_raydium_price(token)
        elif token.source == TokenSource.PUMPFUN:
            await self._update_pumpfun_price(token)

    async def _update_pumpfun_price(self, token: Token):
        """Update PumpFun token price from Dexscreener (they track pump.fun tokens)"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{DEXSCREENER_TOKENS}/{token.mint}",
                    timeout=aiohttp.ClientTimeout(total=3)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        pairs = data.get('pairs', [])
                        if pairs:
                            p = pairs[0]
                            token.price_usd = float(p.get('priceUsd', token.price_usd) or token.price_usd)
                            token.volume_5m = float(p.get('volume', {}).get('m5', 0) or 0)
                            token.change_5m = float(p.get('priceChange', {}).get('m5', 0) or 0)
                            token.last_update = time.time()
        except:
            pass

    async def _update_raydium_price(self, token: Token):
        """Update Raydium token price - use Jupiter for speed, fallback to Dexscreener"""
        try:
            async with aiohttp.ClientSession() as session:
                # Try Jupiter first - faster updates
                try:
                    async with session.get(
                        f"{JUPITER_PRICE}?ids={token.mint}",
                        timeout=aiohttp.ClientTimeout(total=2)
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            price_data = data.get('data', {}).get(token.mint, {})
                            if price_data.get('price'):
                                token.price_usd = float(price_data['price'])
                                token.last_update = time.time()
                                return
                except:
                    pass

                # Fallback to Dexscreener
                async with session.get(
                    f"{DEXSCREENER_TOKENS}/{token.mint}",
                    timeout=aiohttp.ClientTimeout(total=3)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        pairs = data.get('pairs', [])
                        if pairs:
                            p = pairs[0]
                            token.price_usd = float(p.get('priceUsd', token.price_usd) or token.price_usd)
                            token.volume_5m = float(p.get('volume', {}).get('m5', 0) or 0)
                            token.change_5m = float(p.get('priceChange', {}).get('m5', 0) or 0)
                            token.last_update = time.time()
        except:
            pass

    # ============================================================
    # STATUS AND REPORTING
    # ============================================================

    async def _status_loop(self):
        """Print periodic status"""
        while self.running:
            await asyncio.sleep(15)
            if not self.running:
                break

            c = '\033[32m' if self.stats.total_pnl_usd >= 0 else '\033[31m'
            r = '\033[0m'

            logger.info(
                f"STATUS | {self.stats.uptime} | "
                f"Scanned: {self.stats.tokens_scanned} | "
                f"Qualified: {self.stats.tokens_qualified} | "
                f"In/Out: {self.stats.entries}/{self.stats.exits} | "
                f"W/L: {self.stats.wins}/{self.stats.losses} | "
                f"{c}PnL: ${self.stats.total_pnl_usd:+.2f}{r} | "
                f"Open: {len(self.positions)}"
            )

            # Show open positions
            for mint, pos in self.positions.items():
                tok = self.tokens.get(mint)
                if not tok:
                    continue
                pnl = (tok.price_usd - pos.entry_price) / pos.entry_price if pos.entry_price else 0
                hold = (time.time() - pos.entry_time) * 1000
                pc = '\033[32m' if pnl >= 0 else '\033[31m'
                print(f"  {safe_str(tok.symbol, 10)}: {pc}{pnl:+.1%}{r} | {hold:.0f}ms | ${tok.price_usd:.6f}")

    def _print_report(self):
        """Print comprehensive final session report with FRICTION ANALYSIS"""
        c = '\033[32m' if self.stats.net_pnl_usd >= 0 else '\033[31m'
        r = '\033[0m'
        g = '\033[32m'
        rd = '\033[31m'
        y = '\033[33m'

        # Handle edge cases
        fastest = self.stats.fastest_exit_ms if self.stats.fastest_exit_ms != float('inf') else 0
        avg_hold = self.stats.avg_hold_time_ms

        # Calculate edge
        edge_after_friction = self.stats.edge_after_friction
        edge_color = g if edge_after_friction > 0 else rd

        # RenTech decision: Is edge profitable after friction?
        profitable_after_friction = self.stats.net_pnl_usd > 0
        decision = f"{g}[OK] PROFITABLE - READY FOR REAL TRADING{r}" if profitable_after_friction else f"{rd}[X] NOT PROFITABLE - DO NOT TRADE REAL{r}"

        print(f"""
+============================================================================+
|                         SESSION COMPLETE                                    |
|                    (WITH REALISTIC FRICTION MODEL)                          |
+============================================================================+
|                          SCANNING STATS                                     |
|  Runtime:              {self.stats.uptime:>10}                                      |
|  Tokens Scanned:       {self.stats.tokens_scanned:>8}                                        |
|  Tokens Qualified:     {self.stats.tokens_qualified:>8}                                        |
|  Unique Tokens:        {len(self.tokens):>8}                                        |
+----------------------------------------------------------------------------+
|                          TRADING STATS                                      |
|  Entries:              {self.stats.entries:>8}                                        |
|  Exits:                {self.stats.exits:>8}                                        |
|  Wins:                 {g}{self.stats.wins:>8}{r}                                        |
|  Losses:               {rd}{self.stats.losses:>8}{r}                                        |
|  Win Rate:             {self.stats.win_rate:>7.1%}                                        |
|  Failed TXs:           {rd}{self.stats.failed_txns:>8}{r}                                        |
+----------------------------------------------------------------------------+
|                          VOLUME TRADED                                      |
|  Total Volume (SOL):   {self.stats.total_volume_traded_sol:>10.4f}                                  |
|  Total Volume (USD):   ${self.stats.total_volume_traded_usd:>12,.2f}                              |
+----------------------------------------------------------------------------+
|                          TIMING STATS                                       |
|  Avg Hold Time:        {avg_hold:>10.0f} ms                                  |
|  Fastest Exit:         {fastest:>10.0f} ms                                  |
|  Slowest Exit:         {self.stats.slowest_exit_ms:>10.0f} ms                                  |
+----------------------------------------------------------------------------+
|                          BEST/WORST TRADES (NET)                            |
|  Best Trade:           {g}${self.stats.best_trade_pnl:>+9.2f}{r} ({self.stats.best_trade_token or 'N/A':>10})          |
|  Worst Trade:          {rd}${self.stats.worst_trade_pnl:>+9.2f}{r} ({self.stats.worst_trade_token or 'N/A':>10})          |
+----------------------------------------------------------------------------+
|                   {y}FRICTION ANALYSIS (RenTech Reality){r}                       |
|  Total Friction:       {y}${self.stats.total_friction_usd:>12.4f}{r}                              |
|  Friction % of Vol:    {y}{self.stats.friction_pct:>11.2f}%{r}                              |
|  Retry Costs:          {y}${self.stats.retry_costs_usd:>12.4f}{r}                              |
+----------------------------------------------------------------------------+
|                          PROFITABILITY ANALYSIS                             |
|  GROSS PnL (no fees):  {g if self.stats.gross_pnl_usd >= 0 else rd}${self.stats.gross_pnl_usd:>+12.2f}{r}                              |
|  - Friction:           {y}-${self.stats.total_friction_usd:>11.4f}{r}                              |
|  {c}NET PnL (REALITY):    ${self.stats.net_pnl_usd:>+12.2f}{r}                              |
|  ---                                                                        |
|  {edge_color}Edge After Friction:   {edge_after_friction:>+11.3f}%{r}                              |
+----------------------------------------------------------------------------+
|                          FINAL RESULTS                                      |
|  Initial Capital:      {self.initial_capital:>12.4f} SOL                              |
|  Final Capital:        {self.capital:>12.4f} SOL                              |
|  {c}Net Return:            {((self.capital - self.initial_capital) / self.initial_capital * 100):>+11.2f}%{r}                              |
+============================================================================+
|  {decision}                     |
+============================================================================+
""")


# ============================================================
# MAIN ENTRY POINT
# ============================================================

async def run(args):
    """Run the unified engine"""
    # Parse sources
    sources = []
    if 'raydium' in args.sources.lower():
        sources.append(TokenSource.RAYDIUM)
    if 'pumpfun' in args.sources.lower() or 'pump' in args.sources.lower():
        sources.append(TokenSource.PUMPFUN)
    if not sources:
        sources = [TokenSource.RAYDIUM]  # Default to Raydium

    engine = UnifiedEngine(
        paper_mode=(args.mode == 'paper'),
        capital=args.capital,
        max_positions=args.max_positions,
        sources=sources,
    )

    try:
        await engine.start()
    except KeyboardInterrupt:
        await engine.stop()


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description='Unified Trading Engine')
    parser.add_argument('--mode', choices=['paper', 'real'], default='paper',
                       help='Trading mode')
    parser.add_argument('--capital', type=float, default=100.0,
                       help='Starting capital in SOL')
    parser.add_argument('--max-positions', type=int, default=3,
                       help='Max concurrent positions')
    parser.add_argument('--sources', type=str, default='raydium',
                       help='Token sources (raydium,pumpfun)')

    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
