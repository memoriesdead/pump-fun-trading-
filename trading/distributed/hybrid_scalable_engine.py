"""
Hybrid Scalable Engine
======================

BEST OF BOTH WORLDS:
1. DISCOVERY: Pump.fun WebSocket API (instant new token detection)
2. ANALYSIS: Solana blockchain (can't be rate limited)
3. EXECUTION: Direct Solana transactions (bonding curve math)

Scale with multiple VPS/IPs via Tor rotation for pump.fun WebSocket.

Usage:
    # Single VPS mode
    python -m trading.distributed.hybrid_scalable_engine --keypair ~/.config/solana/pumpfun.json

    # Multi-VPS mode (with Redis coordination)
    python -m trading.distributed.hybrid_scalable_engine --keypair ~/.config/solana/pumpfun.json --redis redis.host.com
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Optional, Set, List
from dataclasses import dataclass, asdict
from pathlib import Path

import aiohttp

try:
    from aiohttp_socks import ProxyConnector
except ImportError:
    ProxyConnector = None

try:
    from solana.rpc.async_api import AsyncClient
    from solders.pubkey import Pubkey
except ImportError:
    AsyncClient = None
    Pubkey = None

try:
    import redis
except ImportError:
    redis = None

# Import our Solana analyzer
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trading.distributed.solana_analyzer import SolanaAnalyzer, TokenMetrics


# Constants
PUMPFUN_WS_URL = "wss://pumpportal.fun/api/data"
PUMPFUN_PROGRAM_ID = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"


@dataclass
class DiscoveredToken:
    """Token discovered from pump.fun WebSocket"""
    mint: str
    name: str
    symbol: str
    creator: str
    initial_buy_sol: float
    discovered_at: datetime
    discovery_source: str  # 'pumpfun_ws', 'pumpfun_ws_tor_9050', etc.

    # Analysis results (filled by Solana analyzer)
    metrics: Optional[TokenMetrics] = None
    signal_strength: float = 0.0
    analyzed: bool = False
    traded: bool = False


class HybridScalableEngine:
    """
    Hybrid trading engine:
    - FAST discovery via pump.fun WebSocket (with Tor IP rotation)
    - RELIABLE analysis via direct Solana blockchain reads
    - SCALABLE execution via direct Solana transactions

    This gives you the speed of pump.fun API with the reliability
    of direct blockchain access.
    """

    def __init__(
        self,
        keypair_path: str = None,
        rpc_url: str = "https://api.mainnet-beta.solana.com",
        redis_host: str = None,
        redis_port: int = 6379,
        tor_ports: List[int] = None,
        paper_trading: bool = True,
        max_position_sol: float = 0.5,
        max_positions: int = 3,
        min_signal_strength: float = 0.7,
        node_id: str = "hybrid1",
    ):
        self.rpc_url = rpc_url
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.tor_ports = tor_ports or list(range(9050, 9060))  # 10 Tor ports
        self.paper_trading = paper_trading
        self.max_position_sol = max_position_sol
        self.max_positions = max_positions
        self.min_signal_strength = min_signal_strength
        self.node_id = node_id

        # Components
        self.analyzer = SolanaAnalyzer(rpc_url)
        self.trader = None
        if keypair_path:
            try:
                from trading.execution.pumpfun_trader import PumpfunTrader
                self.trader = PumpfunTrader(keypair_path, rpc_url)
            except ImportError:
                print("WARNING: PumpfunTrader not available")

        # State
        self.discovered_tokens: Dict[str, DiscoveredToken] = {}
        self.positions: Dict[str, dict] = {}
        self.seen_mints: Set[str] = set()
        self._running = False
        self._redis = None

        # Stats
        self.stats = {
            "tokens_discovered": 0,
            "tokens_analyzed": 0,
            "signals_generated": 0,
            "trades_executed": 0,
            "ws_connections": 0,
            "ws_reconnects": 0,
            "errors": 0,
        }

    async def start(self):
        """Start the hybrid engine"""
        print("=" * 60)
        print("HYBRID SCALABLE PUMP.FUN ENGINE")
        print("=" * 60)
        print("Discovery: Pump.fun WebSocket (with Tor rotation)")
        print("Analysis:  Direct Solana blockchain")
        print("Execution: Direct Solana transactions")
        print()

        # Connect to Solana RPC
        if not await self.analyzer.connect():
            print("ERROR: Failed to connect to Solana RPC")
            return

        # Connect to Redis if configured
        if self.redis_host:
            await self._connect_redis()

        # Connect trader
        if self.trader:
            if await self.trader.connect():
                balance = await self.trader.get_balance()
                print(f"Wallet connected. Balance: {balance:.4f} SOL")
            else:
                print("WARNING: Trader not connected. Paper trading only.")
                self.paper_trading = True

        mode = "PAPER" if self.paper_trading else "LIVE"
        print(f"\nMode: {mode} TRADING")
        print(f"Max position: {self.max_position_sol} SOL")
        print(f"Max positions: {self.max_positions}")
        print(f"Min signal: {self.min_signal_strength}")
        print(f"Tor ports: {len(self.tor_ports)}")
        print()

        self._running = True

        # Start all tasks
        tasks = [
            # Multiple WebSocket connections through different Tor ports
            *[self._pumpfun_websocket(port) for port in self.tor_ports[:5]],
            # Direct WebSocket without Tor (fastest)
            self._pumpfun_websocket(None),
            # Solana-based analysis loop
            self._analyze_discovered_tokens(),
            # Position monitoring
            self._monitor_positions(),
            # Stats reporting
            self._report_stats(),
        ]

        print(f"Starting {len(tasks)} concurrent tasks...")

        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            self._running = False
            print("\nShutting down...")

        await self.cleanup()

    async def cleanup(self):
        """Clean up connections"""
        await self.analyzer.close()
        if self.trader:
            await self.trader.close()

    async def _connect_redis(self):
        """Connect to Redis for multi-VPS coordination"""
        if redis is None:
            print("WARNING: redis package not installed")
            return False

        try:
            self._redis = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                decode_responses=True
            )
            self._redis.ping()
            print(f"Connected to Redis at {self.redis_host}:{self.redis_port}")

            # Register this node
            self._redis.hset("hybrid_nodes", self.node_id, json.dumps({
                "started": datetime.utcnow().isoformat(),
                "tor_ports": len(self.tor_ports),
            }))

            return True
        except Exception as e:
            print(f"Redis connection failed: {e}")
            return False

    async def _pumpfun_websocket(self, tor_port: Optional[int] = None):
        """
        Connect to pump.fun WebSocket for instant token discovery.

        Optionally route through Tor for IP rotation.
        """
        source = f"pumpfun_ws_tor_{tor_port}" if tor_port else "pumpfun_ws_direct"

        while self._running:
            try:
                connector = None
                if tor_port and ProxyConnector:
                    connector = ProxyConnector.from_url(f"socks5://127.0.0.1:{tor_port}")

                async with aiohttp.ClientSession(connector=connector) as session:
                    async with session.ws_connect(
                        PUMPFUN_WS_URL,
                        timeout=aiohttp.ClientTimeout(total=30),
                        heartbeat=20,
                    ) as ws:
                        # Subscribe to new token events
                        await ws.send_json({
                            "method": "subscribeNewToken"
                        })

                        self.stats["ws_connections"] += 1
                        print(f"[{self.node_id}] WebSocket connected (source={source})")

                        async for msg in ws:
                            if not self._running:
                                break

                            if msg.type == aiohttp.WSMsgType.TEXT:
                                await self._handle_pumpfun_message(msg.data, source)
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                print(f"[{source}] WebSocket error")
                                break
                            elif msg.type == aiohttp.WSMsgType.CLOSED:
                                break

            except asyncio.TimeoutError:
                print(f"[{source}] Connection timeout, reconnecting...")
            except Exception as e:
                self.stats["errors"] += 1
                print(f"[{source}] Error: {e}")

            self.stats["ws_reconnects"] += 1
            await asyncio.sleep(3)  # Reconnect delay

    async def _handle_pumpfun_message(self, data: str, source: str):
        """Parse pump.fun WebSocket message"""
        try:
            msg = json.loads(data)

            # New token creation event
            if msg.get("txType") == "create":
                mint = msg.get("mint", "")

                # Skip if already seen
                if mint in self.seen_mints:
                    return
                self.seen_mints.add(mint)

                # Limit seen mints to prevent memory bloat
                if len(self.seen_mints) > 50000:
                    self.seen_mints = set(list(self.seen_mints)[-25000:])

                # Check Redis for global deduplication
                if self._redis:
                    if self._redis.sismember("global_seen_mints", mint):
                        return
                    self._redis.sadd("global_seen_mints", mint)
                    self._redis.expire("global_seen_mints", 3600)  # 1 hour TTL

                # Create discovered token
                token = DiscoveredToken(
                    mint=mint,
                    name=msg.get("name", ""),
                    symbol=msg.get("symbol", ""),
                    creator=msg.get("traderPublicKey", ""),
                    initial_buy_sol=msg.get("solAmount", 0) / 1e9 if msg.get("solAmount") else 0,
                    discovered_at=datetime.utcnow(),
                    discovery_source=source,
                )

                self.discovered_tokens[mint] = token
                self.stats["tokens_discovered"] += 1

                print(f"[DISCOVER] {token.symbol or 'Unknown'} | {mint[:16]}... | "
                      f"Creator: {token.creator[:8]}... | InitBuy: {token.initial_buy_sol:.3f} SOL")

                # Publish to Redis for other nodes
                if self._redis:
                    self._redis.publish("token_discoveries", json.dumps({
                        "mint": mint,
                        "name": token.name,
                        "symbol": token.symbol,
                        "creator": token.creator,
                        "initial_buy_sol": token.initial_buy_sol,
                        "discovered_by": self.node_id,
                        "source": source,
                        "timestamp": token.discovered_at.isoformat(),
                    }))

        except json.JSONDecodeError:
            pass
        except Exception as e:
            self.stats["errors"] += 1

    async def _analyze_discovered_tokens(self):
        """
        Analyze discovered tokens using DIRECT SOLANA BLOCKCHAIN.

        This part can't be rate limited - we're reading the blockchain itself.
        """
        print("[ANALYZER] Starting Solana blockchain analysis loop...")

        while self._running:
            try:
                # Get tokens that haven't been analyzed yet
                tokens_to_analyze = [
                    t for t in self.discovered_tokens.values()
                    if not t.analyzed and not t.traded
                ][:10]  # Analyze 10 at a time

                for token in tokens_to_analyze:
                    if not self._running:
                        break

                    # Skip if too old (> 5 min)
                    age = (datetime.utcnow() - token.discovered_at).total_seconds()
                    if age > 300:
                        del self.discovered_tokens[token.mint]
                        continue

                    # Analyze via Solana blockchain
                    metrics = await self.analyzer.analyze_token(
                        token.mint,
                        creation_time=token.discovered_at
                    )

                    token.analyzed = True
                    self.stats["tokens_analyzed"] += 1

                    if metrics:
                        token.metrics = metrics
                        token.signal_strength = metrics.signal_strength

                        # Print analysis results
                        print(f"[ANALYZE] {token.symbol} | Signal: {metrics.signal_strength:.2f} | "
                              f"Wallets: {metrics.unique_wallets} | BuyP: {metrics.buy_pressure:.1%} | "
                              f"Vol: {metrics.total_volume_sol:.2f} SOL")

                        # Check if we should trade
                        if self._should_trade(token):
                            await self._execute_trade(token)

                await asyncio.sleep(0.5)  # Small delay between batches

            except Exception as e:
                self.stats["errors"] += 1
                await asyncio.sleep(1)

    def _should_trade(self, token: DiscoveredToken) -> bool:
        """Determine if we should trade this token"""
        # Check position limits
        if len(self.positions) >= self.max_positions:
            return False

        # Check if already in position
        if token.mint in self.positions:
            return False

        # Check signal strength
        if token.signal_strength < self.min_signal_strength:
            return False

        # Check metrics
        if not token.metrics:
            return False

        metrics = token.metrics

        # Minimum requirements
        if metrics.unique_wallets < 10:
            return False

        if metrics.buy_pressure < 0.5:
            return False

        if metrics.total_volume_sol < 1.0:
            return False

        # Check Redis to avoid duplicate trades across VPS
        if self._redis:
            # Try to acquire lock on this mint
            lock_key = f"trading_lock:{token.mint}"
            if not self._redis.setnx(lock_key, self.node_id):
                return False  # Another node is trading this
            self._redis.expire(lock_key, 60)  # Lock expires in 60s

        return True

    async def _execute_trade(self, token: DiscoveredToken):
        """Execute a buy trade"""
        token.traded = True
        self.stats["signals_generated"] += 1

        metrics = token.metrics
        mint = token.mint

        # Calculate position size
        confidence = token.signal_strength
        size_sol = min(
            self.max_position_sol * confidence,
            self.max_position_sol
        )

        print(f"\n{'='*50}")
        print(f"[TRADE SIGNAL] {token.symbol}")
        print(f"  Mint: {mint}")
        print(f"  Signal: {token.signal_strength:.2f}")
        print(f"  Wallets: {metrics.unique_wallets}")
        print(f"  Buy Pressure: {metrics.buy_pressure:.1%}")
        print(f"  Volume: {metrics.total_volume_sol:.2f} SOL")
        print(f"  Position Size: {size_sol:.3f} SOL")
        print(f"{'='*50}\n")

        if self.paper_trading:
            print(f"  [PAPER] Simulated buy")
            self.positions[mint] = {
                "mint": mint,
                "symbol": token.symbol,
                "entry_time": datetime.utcnow().isoformat(),
                "size_sol": size_sol,
                "entry_price": metrics.price,
                "paper": True,
            }
            self.stats["trades_executed"] += 1
        else:
            # LIVE TRADE via Solana blockchain
            result = await self.trader.buy(
                mint=mint,
                sol_amount=size_sol,
                slippage=0.05
            )

            if result.get("success"):
                print(f"  [LIVE] Buy executed: {result.get('signature', 'N/A')}")
                self.positions[mint] = {
                    "mint": mint,
                    "symbol": token.symbol,
                    "entry_time": datetime.utcnow().isoformat(),
                    "size_sol": size_sol,
                    "entry_price": result.get("price", metrics.price),
                    "signature": result.get("signature"),
                    "paper": False,
                }
                self.stats["trades_executed"] += 1
            else:
                print(f"  [ERROR] Buy failed: {result.get('error')}")
                token.traded = False

    async def _monitor_positions(self):
        """Monitor open positions for exit conditions"""
        print("[POSITIONS] Starting position monitor...")

        while self._running:
            try:
                for mint, position in list(self.positions.items()):
                    # Get current state from Solana blockchain
                    state = await self.analyzer.get_bonding_curve_state(mint)

                    if not state:
                        continue

                    current_price = state.price_per_token
                    entry_price = position.get("entry_price", 0)

                    if entry_price == 0:
                        continue

                    # Calculate P&L
                    pnl_pct = (current_price - entry_price) / entry_price

                    # Check exit conditions
                    should_exit = False
                    exit_reason = ""

                    # Stop loss at -30%
                    if pnl_pct <= -0.30:
                        should_exit = True
                        exit_reason = "STOP_LOSS"

                    # Take profit at +100%
                    if pnl_pct >= 1.00:
                        should_exit = True
                        exit_reason = "TAKE_PROFIT"

                    # Time exit after 10 min
                    entry_time = datetime.fromisoformat(position["entry_time"])
                    age = (datetime.utcnow() - entry_time).total_seconds()
                    if age > 600:
                        should_exit = True
                        exit_reason = "TIME_EXIT"

                    if should_exit:
                        await self._execute_sell(mint, current_price, exit_reason, pnl_pct)

                await asyncio.sleep(5)

            except Exception as e:
                self.stats["errors"] += 1
                await asyncio.sleep(5)

    async def _execute_sell(self, mint: str, price: float, reason: str, pnl_pct: float):
        """Execute sell trade"""
        position = self.positions.get(mint)
        if not position:
            return

        size_sol = position.get("size_sol", 0)
        pnl_sol = size_sol * pnl_pct

        print(f"\n[EXIT] {position.get('symbol', mint[:8])}")
        print(f"  Reason: {reason}")
        print(f"  P&L: {pnl_sol:+.4f} SOL ({pnl_pct:+.1%})")

        if self.paper_trading:
            print(f"  [PAPER] Simulated sell")
        else:
            result = await self.trader.sell(mint, token_pct=1.0)
            if result.get("success"):
                print(f"  [LIVE] Sell executed: {result.get('signature', 'N/A')}")
            else:
                print(f"  [ERROR] Sell failed: {result.get('error')}")

        del self.positions[mint]

    async def _report_stats(self):
        """Report stats periodically"""
        while self._running:
            await asyncio.sleep(60)

            print(f"\n[STATS] {datetime.utcnow().isoformat()}")
            print(f"  Tokens discovered: {self.stats['tokens_discovered']}")
            print(f"  Tokens analyzed: {self.stats['tokens_analyzed']}")
            print(f"  Signals generated: {self.stats['signals_generated']}")
            print(f"  Trades executed: {self.stats['trades_executed']}")
            print(f"  WS connections: {self.stats['ws_connections']}")
            print(f"  WS reconnects: {self.stats['ws_reconnects']}")
            print(f"  Watching: {len(self.discovered_tokens)} tokens")
            print(f"  Open positions: {len(self.positions)}")
            print(f"  Errors: {self.stats['errors']}")

            # Update Redis stats
            if self._redis:
                self._redis.hset("hybrid_stats", self.node_id, json.dumps({
                    **self.stats,
                    "timestamp": datetime.utcnow().isoformat(),
                }))


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid Scalable Pump.fun Engine")
    parser.add_argument("--keypair", help="Path to Solana keypair")
    parser.add_argument("--rpc", default="https://api.mainnet-beta.solana.com")
    parser.add_argument("--redis", help="Redis host for multi-VPS coordination")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--node-id", default="hybrid1", help="Unique node identifier")
    parser.add_argument("--live", action="store_true", help="Enable LIVE trading")
    parser.add_argument("--max-sol", type=float, default=0.5)
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--min-signal", type=float, default=0.7)
    parser.add_argument("--tor-start", type=int, default=9050, help="Starting Tor port")
    parser.add_argument("--tor-count", type=int, default=10, help="Number of Tor ports")
    args = parser.parse_args()

    tor_ports = list(range(args.tor_start, args.tor_start + args.tor_count))

    engine = HybridScalableEngine(
        keypair_path=args.keypair,
        rpc_url=args.rpc,
        redis_host=args.redis,
        redis_port=args.redis_port,
        tor_ports=tor_ports,
        paper_trading=not args.live,
        max_position_sol=args.max_sol,
        max_positions=args.max_positions,
        min_signal_strength=args.min_signal,
        node_id=args.node_id,
    )

    await engine.start()


if __name__ == "__main__":
    asyncio.run(main())
