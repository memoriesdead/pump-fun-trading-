"""
Discovery Node for Distributed Trading
======================================

Watches for new pump.fun tokens via multiple sources.
Publishes signals to Redis for execution nodes.

NO TRADING on this node - discovery only.

Usage:
    python -m trading.distributed.discovery_node --node-id vps1 --redis redis.your-server.com
"""

import asyncio
import json
import time
import hashlib
from datetime import datetime
from typing import Optional, Set, Dict
from dataclasses import dataclass, asdict
import aiohttp

try:
    import redis
except ImportError:
    redis = None


# Pump.fun constants
PUMPFUN_PROGRAM_ID = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
PUMPFUN_WS_URL = "wss://pumpportal.fun/api/data"


@dataclass
class TokenSignal:
    """Signal for a newly discovered token"""
    mint: str
    discovered_by: str
    discovery_source: str  # 'websocket', 'solana_logs', 'helius'
    timestamp: str
    block_slot: Optional[int] = None
    creator: Optional[str] = None
    name: Optional[str] = None
    symbol: Optional[str] = None
    initial_buy_sol: Optional[float] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class DiscoveryNode:
    """
    Multi-source token discovery node.

    Watches for new tokens from:
    1. Pump.fun WebSocket (fastest)
    2. Solana RPC logs (backup, no rate limit)
    3. Helius WebSocket (paid, reliable)
    """

    def __init__(
        self,
        node_id: str,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        tor_ports: list = None,
    ):
        self.node_id = node_id
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.tor_ports = tor_ports or list(range(9050, 9100))

        self.seen_tokens: Set[str] = set()
        self.stats = {
            "tokens_discovered": 0,
            "signals_published": 0,
            "duplicates_skipped": 0,
            "errors": 0,
        }

        self._redis = None
        self._running = False

    async def connect(self):
        """Initialize Redis connection"""
        if redis is None:
            print("WARNING: redis package not installed. Running without coordination.")
            return False

        try:
            self._redis = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                decode_responses=True
            )
            self._redis.ping()
            print(f"[{self.node_id}] Connected to Redis at {self.redis_host}:{self.redis_port}")

            # Register this node
            self._redis.hset("discovery_nodes", self.node_id, json.dumps({
                "started": datetime.utcnow().isoformat(),
                "tor_ports": len(self.tor_ports),
            }))

            return True
        except Exception as e:
            print(f"[{self.node_id}] Redis connection failed: {e}")
            return False

    async def publish_signal(self, signal: TokenSignal):
        """Publish token signal to Redis"""
        mint = signal.mint

        # Local deduplication
        if mint in self.seen_tokens:
            self.stats["duplicates_skipped"] += 1
            return

        self.seen_tokens.add(mint)
        self.stats["tokens_discovered"] += 1

        if self._redis:
            try:
                # Publish to channel (real-time)
                self._redis.publish("token_signals", signal.to_json())

                # Store in sorted set by timestamp (persistence)
                self._redis.zadd("recent_tokens", {mint: time.time()})

                # Store full signal data
                self._redis.hset("token_data", mint, signal.to_json())

                self.stats["signals_published"] += 1

            except Exception as e:
                print(f"[{self.node_id}] Redis publish error: {e}")
                self.stats["errors"] += 1

        print(f"[{self.node_id}] NEW TOKEN: {mint[:16]}... source={signal.discovery_source}")

    async def watch_pumpfun_websocket(self, tor_port: Optional[int] = None):
        """
        Watch pump.fun WebSocket for new tokens.

        Optionally route through Tor for IP rotation.
        """
        proxy = f"socks5://127.0.0.1:{tor_port}" if tor_port else None

        while self._running:
            try:
                connector = None
                if proxy:
                    from aiohttp_socks import ProxyConnector
                    connector = ProxyConnector.from_url(proxy)

                async with aiohttp.ClientSession(connector=connector) as session:
                    async with session.ws_connect(PUMPFUN_WS_URL) as ws:
                        # Subscribe to new token events
                        await ws.send_json({
                            "method": "subscribeNewToken"
                        })

                        print(f"[{self.node_id}] WebSocket connected (tor={tor_port})")

                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                await self._handle_ws_message(msg.data)
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                break

            except Exception as e:
                print(f"[{self.node_id}] WebSocket error: {e}")
                self.stats["errors"] += 1
                await asyncio.sleep(5)  # Reconnect delay

    async def _handle_ws_message(self, data: str):
        """Parse WebSocket message and publish signal"""
        try:
            msg = json.loads(data)

            # New token creation event
            if msg.get("txType") == "create":
                signal = TokenSignal(
                    mint=msg.get("mint", ""),
                    discovered_by=self.node_id,
                    discovery_source="websocket",
                    timestamp=datetime.utcnow().isoformat(),
                    creator=msg.get("traderPublicKey"),
                    name=msg.get("name"),
                    symbol=msg.get("symbol"),
                    initial_buy_sol=msg.get("solAmount"),
                )
                await self.publish_signal(signal)

        except json.JSONDecodeError:
            pass
        except Exception as e:
            self.stats["errors"] += 1

    async def watch_solana_logs(self, rpc_url: str = "https://api.mainnet-beta.solana.com"):
        """
        Watch Solana program logs directly.

        Backup method - no rate limits from pump.fun.
        Slightly slower but more reliable.
        """
        while self._running:
            try:
                async with aiohttp.ClientSession() as session:
                    # Subscribe to program logs
                    ws_url = rpc_url.replace("https://", "wss://").replace("http://", "ws://")

                    async with session.ws_connect(ws_url) as ws:
                        # Subscribe to pump.fun program
                        await ws.send_json({
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "logsSubscribe",
                            "params": [
                                {"mentions": [PUMPFUN_PROGRAM_ID]},
                                {"commitment": "confirmed"}
                            ]
                        })

                        print(f"[{self.node_id}] Solana logs subscription active")

                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                await self._handle_log_message(msg.data)

            except Exception as e:
                print(f"[{self.node_id}] Solana logs error: {e}")
                await asyncio.sleep(10)

    async def _handle_log_message(self, data: str):
        """Parse Solana log and extract token creates"""
        try:
            msg = json.loads(data)
            result = msg.get("params", {}).get("result", {})

            if not result:
                return

            logs = result.get("value", {}).get("logs", [])
            signature = result.get("value", {}).get("signature", "")

            # Look for token creation in logs
            for log in logs:
                if "Program log: Instruction: Create" in log:
                    # Extract mint from accounts (would need full tx decode)
                    # For now, use signature as identifier
                    signal = TokenSignal(
                        mint=signature,  # Placeholder - need full decode
                        discovered_by=self.node_id,
                        discovery_source="solana_logs",
                        timestamp=datetime.utcnow().isoformat(),
                    )
                    # Note: Full implementation would decode tx to get mint
                    break

        except Exception as e:
            pass

    async def watch_helius(self, api_key: str):
        """
        Watch via Helius WebSocket (paid service).

        Most reliable, but costs money.
        """
        ws_url = f"wss://atlas-mainnet.helius-rpc.com/?api-key={api_key}"

        while self._running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(ws_url) as ws:
                        # Subscribe to pump.fun transactions
                        await ws.send_json({
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "transactionSubscribe",
                            "params": [
                                {
                                    "accountInclude": [PUMPFUN_PROGRAM_ID],
                                },
                                {
                                    "commitment": "confirmed",
                                    "encoding": "jsonParsed",
                                    "transactionDetails": "full",
                                    "showRewards": False,
                                    "maxSupportedTransactionVersion": 0,
                                }
                            ]
                        })

                        print(f"[{self.node_id}] Helius subscription active")

                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                await self._handle_helius_message(msg.data)

            except Exception as e:
                print(f"[{self.node_id}] Helius error: {e}")
                await asyncio.sleep(5)

    async def _handle_helius_message(self, data: str):
        """Parse Helius enhanced transaction"""
        try:
            msg = json.loads(data)
            tx = msg.get("params", {}).get("result", {})

            if not tx:
                return

            # Check for token creation
            instructions = tx.get("transaction", {}).get("message", {}).get("instructions", [])

            for ix in instructions:
                if ix.get("programId") == PUMPFUN_PROGRAM_ID:
                    # Parse instruction to find create
                    accounts = ix.get("accounts", [])
                    if len(accounts) >= 3:
                        signal = TokenSignal(
                            mint=accounts[2],  # Mint is typically 3rd account
                            discovered_by=self.node_id,
                            discovery_source="helius",
                            timestamp=datetime.utcnow().isoformat(),
                            block_slot=tx.get("slot"),
                        )
                        await self.publish_signal(signal)

        except Exception as e:
            pass

    async def run(
        self,
        use_websocket: bool = True,
        use_solana_logs: bool = False,
        use_helius: bool = False,
        helius_api_key: str = None,
        solana_rpc: str = "https://api.mainnet-beta.solana.com",
    ):
        """
        Run discovery node with selected sources.

        Args:
            use_websocket: Use pump.fun WebSocket
            use_solana_logs: Use direct Solana log subscription
            use_helius: Use Helius enhanced API
        """
        await self.connect()

        self._running = True
        tasks = []

        if use_websocket:
            # Run multiple WebSocket connections through different Tor ports
            for i, port in enumerate(self.tor_ports[:5]):  # Use first 5 Tor ports
                tasks.append(self.watch_pumpfun_websocket(port))

        if use_solana_logs:
            tasks.append(self.watch_solana_logs(solana_rpc))

        if use_helius and helius_api_key:
            tasks.append(self.watch_helius(helius_api_key))

        # Stats reporter
        tasks.append(self._report_stats())

        print(f"[{self.node_id}] Starting discovery with {len(tasks)} sources")

        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            self._running = False
            print(f"\n[{self.node_id}] Shutting down...")
            print(f"Final stats: {self.stats}")

    async def _report_stats(self):
        """Report stats periodically"""
        while self._running:
            await asyncio.sleep(60)

            print(f"[{self.node_id}] Stats: {self.stats}")

            if self._redis:
                self._redis.hset("node_stats", self.node_id, json.dumps({
                    **self.stats,
                    "timestamp": datetime.utcnow().isoformat(),
                }))


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Discovery Node")
    parser.add_argument("--node-id", required=True, help="Unique node identifier")
    parser.add_argument("--redis", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--helius-key", help="Helius API key (optional)")
    parser.add_argument("--use-logs", action="store_true", help="Use Solana logs")
    args = parser.parse_args()

    node = DiscoveryNode(
        node_id=args.node_id,
        redis_host=args.redis,
        redis_port=args.redis_port,
    )

    await node.run(
        use_websocket=True,
        use_solana_logs=args.use_logs,
        use_helius=bool(args.helius_key),
        helius_api_key=args.helius_key,
    )


if __name__ == "__main__":
    asyncio.run(main())
