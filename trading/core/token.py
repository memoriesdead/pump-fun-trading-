"""
Token - Data structure for tradeable tokens.

Usage:
    from trading.core import Token, TokenSource

    token = Token(
        address="So11111111111111111111111111111111111111112",
        symbol="SOL",
        name="Wrapped SOL",
        source=TokenSource.RAYDIUM,
        price=230.0,
        liquidity_usd=1_000_000,
        volume_5m=50000,
        momentum=15.5,
    )
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional
import time


class TokenSource(Enum):
    """Source of token data."""
    RAYDIUM = "raydium"
    PUMPFUN = "pumpfun"
    JUPITER = "jupiter"
    DEXSCREENER = "dexscreener"
    GECKOTERMINAL = "geckoterminal"


@dataclass
class Token:
    """
    Tradeable token with all relevant metrics.

    Immutable snapshot of token state at discovery time.
    """

    # Identity
    address: str
    symbol: str
    name: str
    source: TokenSource

    # Price data
    price: float = 0.0
    price_usd: float = 0.0  # Price in USD (if different from price)

    # Liquidity
    liquidity_usd: float = 0.0
    liquidity_sol: float = 0.0

    # Volume
    volume_5m: float = 0.0
    volume_1h: float = 0.0
    volume_24h: float = 0.0

    # Price changes
    change_5m: float = 0.0
    change_1h: float = 0.0
    change_24h: float = 0.0

    # Momentum (THE critical metric)
    momentum: float = 0.0

    # Transaction counts
    txns_5m: int = 0
    buys_5m: int = 0
    sells_5m: int = 0

    # Age
    created_at: float = field(default_factory=time.time)
    discovered_at: float = field(default_factory=time.time)

    # Pool info
    pool_address: str = ""
    base_token: str = ""
    quote_token: str = ""

    # Metadata
    fdv: float = 0.0  # Fully diluted valuation
    market_cap: float = 0.0

    # Raw data storage
    raw_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def buy_ratio(self) -> float:
        """Percentage of transactions that are buys."""
        total = self.buys_5m + self.sells_5m
        return self.buys_5m / total if total > 0 else 0.0

    @property
    def age_seconds(self) -> float:
        """Age since creation in seconds."""
        return time.time() - self.created_at

    @property
    def is_pumpfun(self) -> bool:
        """Check if token is from PumpFun."""
        return self.source == TokenSource.PUMPFUN

    @property
    def expected_edge(self) -> float:
        """
        Calculate expected edge based on momentum and buy pressure.

        Simple model: momentum * buy_ratio gives expected continuation.
        """
        return self.momentum * self.buy_ratio / 100 if self.momentum > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'address': self.address,
            'symbol': self.symbol,
            'name': self.name,
            'source': self.source.value,
            'price': self.price,
            'price_usd': self.price_usd,
            'liquidity_usd': self.liquidity_usd,
            'liquidity_sol': self.liquidity_sol,
            'volume_5m': self.volume_5m,
            'volume_1h': self.volume_1h,
            'volume_24h': self.volume_24h,
            'change_5m': self.change_5m,
            'change_1h': self.change_1h,
            'change_24h': self.change_24h,
            'momentum': self.momentum,
            'txns_5m': self.txns_5m,
            'buys_5m': self.buys_5m,
            'sells_5m': self.sells_5m,
            'buy_ratio': self.buy_ratio,
            'created_at': self.created_at,
            'discovered_at': self.discovered_at,
            'pool_address': self.pool_address,
            'fdv': self.fdv,
            'market_cap': self.market_cap,
            'age_seconds': self.age_seconds,
            'expected_edge': self.expected_edge,
        }

    @classmethod
    def from_geckoterminal(cls, pool_data: Dict[str, Any]) -> 'Token':
        """
        Create Token from GeckoTerminal API response.

        Args:
            pool_data: Pool object from GeckoTerminal API
        """
        attrs = pool_data.get('attributes', {})
        volume = attrs.get('volume_usd', {})
        price_change = attrs.get('price_change_percentage', {})
        txns = attrs.get('transactions', {}).get('m5', {})

        # Parse creation time
        created_str = attrs.get('pool_created_at', '')
        try:
            from datetime import datetime
            created_at = datetime.fromisoformat(created_str.replace('Z', '+00:00')).timestamp()
        except:
            created_at = time.time()

        return cls(
            address=attrs.get('address', ''),
            symbol=attrs.get('name', '').split('/')[0] if '/' in attrs.get('name', '') else attrs.get('name', ''),
            name=attrs.get('name', ''),
            source=TokenSource.GECKOTERMINAL,
            price=float(attrs.get('base_token_price_usd', 0) or 0),
            price_usd=float(attrs.get('base_token_price_usd', 0) or 0),
            liquidity_usd=float(attrs.get('reserve_in_usd', 0) or 0),
            volume_5m=float(volume.get('m5', 0) or 0),
            volume_1h=float(volume.get('h1', 0) or 0),
            volume_24h=float(volume.get('h24', 0) or 0),
            change_5m=float(price_change.get('m5', 0) or 0),
            change_1h=float(price_change.get('h1', 0) or 0),
            change_24h=float(price_change.get('h24', 0) or 0),
            momentum=float(price_change.get('m5', 0) or 0),  # Use 5m change as momentum
            txns_5m=int(txns.get('buys', 0) or 0) + int(txns.get('sells', 0) or 0),
            buys_5m=int(txns.get('buys', 0) or 0),
            sells_5m=int(txns.get('sells', 0) or 0),
            created_at=created_at,
            pool_address=pool_data.get('id', ''),
            fdv=float(attrs.get('fdv_usd', 0) or 0),
            market_cap=float(attrs.get('market_cap_usd', 0) or 0),
            raw_data=pool_data,
        )

    @classmethod
    def from_pumpfun(cls, token_data: Dict[str, Any]) -> 'Token':
        """
        Create Token from PumpFun WebSocket message.

        Args:
            token_data: Token object from PumpFun WS
        """
        return cls(
            address=token_data.get('mint', ''),
            symbol=token_data.get('symbol', ''),
            name=token_data.get('name', ''),
            source=TokenSource.PUMPFUN,
            price=float(token_data.get('price', 0) or 0),
            price_usd=float(token_data.get('usdMarketCap', 0) or 0) / 1e9 if token_data.get('usdMarketCap') else 0,
            liquidity_usd=float(token_data.get('vSolInBondingCurve', 0) or 0) * 230,  # Approx USD
            liquidity_sol=float(token_data.get('vSolInBondingCurve', 0) or 0),
            market_cap=float(token_data.get('usdMarketCap', 0) or 0),
            raw_data=token_data,
        )

    def __str__(self) -> str:
        return f"{self.symbol} | ${self.price:.8f} | Liq: ${self.liquidity_usd:,.0f} | Mom: {self.momentum:+.1f}%"

    def __repr__(self) -> str:
        return f"Token({self.symbol}, ${self.price:.8f}, liq=${self.liquidity_usd:,.0f})"
