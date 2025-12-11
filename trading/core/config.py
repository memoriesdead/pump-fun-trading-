"""
Configuration - All trading parameters in one place.

Usage:
    from trading.core import Config, FilterConfig, ScalpConfig

    # Access default config
    config = Config()
    print(config.filters.min_momentum)

    # Override for different account sizes
    config = Config.for_account_size(10)  # $10 account
"""
from dataclasses import dataclass, field
from typing import Dict, Any


# API Endpoints
ENDPOINTS = {
    # GeckoTerminal (free, reliable)
    'GECKOTERMINAL_TRENDING': 'https://api.geckoterminal.com/api/v2/networks/solana/trending_pools',
    'GECKOTERMINAL_NEW': 'https://api.geckoterminal.com/api/v2/networks/solana/new_pools',
    'GECKOTERMINAL_POOLS': 'https://api.geckoterminal.com/api/v2/networks/solana/dexes/raydium/pools',

    # DexScreener
    'DEXSCREENER_SEARCH': 'https://api.dexscreener.com/latest/dex/search',
    'DEXSCREENER_TOKENS': 'https://api.dexscreener.com/latest/dex/tokens',

    # Pump.fun WebSocket
    'PUMPPORTAL_WS': 'wss://pumpportal.fun/api/data',

    # Jupiter
    'JUPITER_PRICE': 'https://price.jup.ag/v4/price',
}


@dataclass
class FilterConfig:
    """Token filter configuration."""

    # Volume thresholds
    min_volume_5m: float = 500.0        # $500+ in 5 min
    min_volume_1h: float = 2000.0       # $2k+ hourly

    # Liquidity
    min_liquidity: float = 25000.0      # $25k+ for lower slippage
    max_liquidity: float = 100_000_000.0

    # Volatility (need big moves to overcome friction)
    min_volatility_5m: float = 10.0     # 10%+ move potential
    max_volatility_5m: float = 500.0

    # Momentum (THE critical filter)
    min_momentum: float = 50.0          # 50%+ momentum
    max_momentum: float = 1000.0

    # Trade activity
    min_txns_5m: int = 5
    min_buy_ratio: float = 0.60         # 60% buys

    # Age
    min_age: int = 30                   # 30 seconds
    max_age: int = 86400 * 30           # 30 days

    def to_dict(self) -> Dict[str, Any]:
        return {
            'min_volume_5m': self.min_volume_5m,
            'min_volume_1h': self.min_volume_1h,
            'min_liquidity': self.min_liquidity,
            'max_liquidity': self.max_liquidity,
            'min_volatility_5m': self.min_volatility_5m,
            'max_volatility_5m': self.max_volatility_5m,
            'min_momentum': self.min_momentum,
            'max_momentum': self.max_momentum,
            'min_txns_5m': self.min_txns_5m,
            'min_buy_ratio': self.min_buy_ratio,
            'min_age': self.min_age,
            'max_age': self.max_age,
        }


@dataclass
class ScalpConfig:
    """Scalping/trading configuration."""

    target_profit_pct: float = 0.15     # 15% target
    max_loss_pct: float = 0.05          # 5% stop loss
    max_hold_ms: int = 60000            # 60 second max
    min_hold_ms: int = 15000            # 15 second min
    confidence_threshold: float = 0.85  # 85% confidence
    trades_per_token: int = 1           # One trade per token
    min_expected_edge: float = 0.12     # 12% min expected move
    max_open_positions: int = 3         # Max concurrent positions
    max_position_pct: float = 0.25      # Max 25% of capital per position
    scan_interval_ms: int = 5000        # 5 second scan interval

    def to_dict(self) -> Dict[str, Any]:
        return {
            'target_profit_pct': self.target_profit_pct,
            'max_loss_pct': self.max_loss_pct,
            'max_hold_ms': self.max_hold_ms,
            'min_hold_ms': self.min_hold_ms,
            'confidence_threshold': self.confidence_threshold,
            'trades_per_token': self.trades_per_token,
            'min_expected_edge': self.min_expected_edge,
            'max_open_positions': self.max_open_positions,
            'max_position_pct': self.max_position_pct,
            'scan_interval_ms': self.scan_interval_ms,
        }


@dataclass
class Config:
    """Master configuration."""

    # Trading mode
    paper_mode: bool = True
    capital_sol: float = 0.043          # ~$10 at $230/SOL

    # Position sizing (Kelly Criterion derived)
    position_size_pct: float = 0.20     # 20% per trade
    max_positions: int = 3

    # Sub-configs
    filters: FilterConfig = field(default_factory=FilterConfig)
    scalp: ScalpConfig = field(default_factory=ScalpConfig)

    # SOL price (for USD calculations)
    sol_price_usd: float = 230.0

    @classmethod
    def for_account_size(cls, usd: float, sol_price: float = 230.0) -> 'Config':
        """Create config optimized for account size."""
        config = cls()
        config.capital_sol = usd / sol_price
        config.sol_price_usd = sol_price

        # Micro accounts need extreme selectivity
        if usd <= 10:
            config.filters.min_momentum = 50.0
            config.filters.min_liquidity = 25000.0
            config.scalp.target_profit_pct = 0.15
            config.scalp.min_expected_edge = 0.12
            config.position_size_pct = 0.25  # Bet more per trade

        elif usd <= 100:
            config.filters.min_momentum = 30.0
            config.filters.min_liquidity = 10000.0
            config.scalp.target_profit_pct = 0.10
            config.scalp.min_expected_edge = 0.08

        else:  # $100+
            config.filters.min_momentum = 15.0
            config.filters.min_liquidity = 5000.0
            config.scalp.target_profit_pct = 0.08
            config.scalp.min_expected_edge = 0.05

        return config

    @property
    def capital_usd(self) -> float:
        return self.capital_sol * self.sol_price_usd

    @property
    def position_size_sol(self) -> float:
        return self.capital_sol * self.position_size_pct

    @property
    def position_size_usd(self) -> float:
        return self.position_size_sol * self.sol_price_usd

    def to_dict(self) -> Dict[str, Any]:
        return {
            'paper_mode': self.paper_mode,
            'capital_sol': self.capital_sol,
            'capital_usd': self.capital_usd,
            'position_size_pct': self.position_size_pct,
            'max_positions': self.max_positions,
            'sol_price_usd': self.sol_price_usd,
            'filters': self.filters.to_dict(),
            'scalp': self.scalp.to_dict(),
        }
