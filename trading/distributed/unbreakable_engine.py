"""
Unbreakable Trading Engine
==========================

ZERO pump.fun API dependency. Everything from Solana blockchain.

Flow:
1. DISCOVER: Solana logs subscription → New tokens
2. ANALYZE: Direct account reads → Metrics
3. EXECUTE: Build tx locally → Send to RPC

Can't be rate limited because:
- Solana logs are the blockchain itself
- Account reads are standard RPC calls
- We build transactions locally (bonding curve math)

Usage:
    python -m trading.distributed.unbreakable_engine --keypair ~/.config/solana/pumpfun.json
"""

import asyncio
import json
import struct
import base64
import time
from datetime import datetime
from typing import Dict, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    from solana.rpc.async_api import AsyncClient
    from solana.rpc.websocket_api import connect as ws_connect
    from solders.pubkey import Pubkey
    from solders.rpc.config import RpcTransactionLogsFilterMentions
except ImportError:
    print("ERROR: Install solana-py: pip install solana solders")
    AsyncClient = None
    RpcTransactionLogsFilterMentions = None

# Import our components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trading.distributed.solana_analyzer import SolanaAnalyzer, TokenMetrics
from trading.execution.pumpfun_trader import PumpfunTrader


# Pump.fun constants
PUMPFUN_PROGRAM_ID = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"


@dataclass
class WatchedToken:
    """Token being watched for trading signals"""
    mint: str
    discovered_at: datetime
    last_check: datetime
    check_count: int = 0
    metrics: Optional[TokenMetrics] = None
    signal_strength: float = 0.0
    traded: bool = False


class UnbreakableEngine:
    """
    Unbreakable pump.fun trading engine.

    NO PUMP.FUN API CALLS - everything from Solana:
    1. Token discovery via logsSubscribe
    2. Token analysis via account reads
    3. Trade execution via direct tx building

    This CANNOT be rate limited by pump.fun.
    """

    def __init__(
        self,
        keypair_path: str,
        rpc_url: str = "https://api.mainnet-beta.solana.com",
        ws_url: str = "wss://api.mainnet-beta.solana.com",
        paper_trading: bool = True,
        max_position_sol: float = 0.5,
        max_positions: int = 3,
        min_signal_strength: float = 0.7,
    ):
        self.rpc_url = rpc_url
        self.ws_url = ws_url
        self.paper_trading = paper_trading
        self.max_position_sol = max_position_sol
        self.max_positions = max_positions
        self.min_signal_strength = min_signal_strength

        # Components
        self.analyzer = SolanaAnalyzer(rpc_url)
        self.trader = PumpfunTrader(keypair_path, rpc_url) if keypair_path else None

        # State
        self.watched_tokens: Dict[str, WatchedToken] = {}
        self.positions: Dict[str, dict] = {}
        self.seen_signatures: Set[str] = set()
        self._running = False

        # Stats
        self.stats = {
            "tokens_discovered": 0,
            "tokens_analyzed": 0,
            "signals_generated": 0,
            "trades_executed": 0,
            "errors": 0,
        }

    async def start(self):
        """Start the unbreakable engine"""
        print("=" * 60)
        print("UNBREAKABLE PUMP.FUN ENGINE")
        print("=" * 60)
        print("NO pump.fun API - Direct Solana blockchain access")
        print()

        # Connect to Solana
        if not await self.analyzer.connect():
            print("ERROR: Failed to connect to Solana RPC")
            return

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
        print()

        self._running = True

        # Start all tasks
        tasks = [
            self._discover_tokens_via_logs(),  # Primary: Solana logs
            self._analyze_watched_tokens(),     # Continuous analysis
            self._monitor_positions(),          # Position management
            self._report_stats(),               # Stats reporting
        ]

        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            self._running = False
            print("\nShutting down...")
            print(f"Final stats: {self.stats}")

        await self.cleanup()

    async def cleanup(self):
        """Clean up connections"""
        await self.analyzer.close()
        if self.trader:
            await self.trader.close()

    async def _discover_tokens_via_logs(self):
        """
        PRIMARY DISCOVERY: Solana program logs subscription.

        This is UNLIMITED - you're watching the blockchain itself.
        pump.fun CAN'T rate limit this.
        """
        print("[DISCOVERY] Starting Solana logs subscription...")
        print(f"[DISCOVERY] Watching program: {PUMPFUN_PROGRAM_ID}")

        while self._running:
            try:
                async with ws_connect(self.ws_url) as ws:
                    # Subscribe to pump.fun program logs
                    # Use proper filter object for newer solana-py versions
                    filter_obj = RpcTransactionLogsFilterMentions(
                        Pubkey.from_string(PUMPFUN_PROGRAM_ID)
                    )
                    await ws.logs_subscribe(
                        filter_obj,
                        commitment="confirmed"
                    )

                    print("[DISCOVERY] Subscribed to pump.fun logs")

                    async for msg in ws:
                        if not self._running:
                            break

                        await self._handle_log_message(msg)

            except Exception as e:
                self.stats["errors"] += 1
                print(f"[DISCOVERY] Error: {e}")
                await asyncio.sleep(5)

    async def _handle_log_message(self, msg):
        """Parse Solana log message for token creation"""
        try:
            # Extract log data
            if hasattr(msg, 'result'):
                result = msg.result
            else:
                return

            if not hasattr(result, 'value'):
                return

            value = result.value
            logs = value.logs if hasattr(value, 'logs') else []
            signature = str(value.signature) if hasattr(value, 'signature') else ""

            # Skip if we've seen this signature
            if signature in self.seen_signatures:
                return
            self.seen_signatures.add(signature)

            # Limit seen signatures to prevent memory bloat
            if len(self.seen_signatures) > 10000:
                self.seen_signatures = set(list(self.seen_signatures)[-5000:])

            # Look for token creation
            is_create = any("Create" in log for log in logs)

            if is_create:
                # Try to extract mint from logs or get from transaction
                mint = await self._extract_mint_from_signature(signature)

                if mint and mint not in self.watched_tokens:
                    self.stats["tokens_discovered"] += 1

                    self.watched_tokens[mint] = WatchedToken(
                        mint=mint,
                        discovered_at=datetime.utcnow(),
                        last_check=datetime.utcnow(),
                    )

                    print(f"[DISCOVERY] NEW TOKEN: {mint[:20]}...")

        except Exception as e:
            pass  # Silent fail for parsing errors

    async def _extract_mint_from_signature(self, signature: str) -> Optional[str]:
        """Extract mint address from a transaction signature"""
        try:
            # Get transaction details
            response = await self.analyzer._client.get_transaction(
                signature,
                encoding="jsonParsed",
                max_supported_transaction_version=0
            )

            if not response.value:
                return None

            # Find the mint in account keys
            message = response.value.transaction.message
            account_keys = message.account_keys

            # The mint is typically in position 2-3 for create instructions
            for key in account_keys[1:5]:
                key_str = str(key.pubkey if hasattr(key, 'pubkey') else key)
                # Check if it's a valid mint (has a bonding curve PDA)
                state = await self.analyzer.get_bonding_curve_state(key_str)
                if state:
                    return key_str

            return None

        except Exception:
            return None

    async def _analyze_watched_tokens(self):
        """
        Continuously analyze watched tokens.

        All analysis is done via direct Solana account reads.
        NO pump.fun API calls.
        """
        print("[ANALYZER] Starting token analysis loop...")

        while self._running:
            try:
                # Get tokens to analyze (oldest first)
                tokens_to_check = sorted(
                    self.watched_tokens.values(),
                    key=lambda t: t.last_check
                )[:20]  # Check 20 at a time

                for token in tokens_to_check:
                    if not self._running:
                        break

                    # Skip if recently checked
                    age = (datetime.utcnow() - token.last_check).total_seconds()
                    if age < 5:  # Check every 5 seconds max
                        continue

                    # Skip if already traded
                    if token.traded:
                        continue

                    # Skip if too old (> 10 min)
                    token_age = (datetime.utcnow() - token.discovered_at).total_seconds()
                    if token_age > 600:
                        del self.watched_tokens[token.mint]
                        continue

                    # Analyze token
                    metrics = await self.analyzer.analyze_token(
                        token.mint,
                        creation_time=token.discovered_at
                    )

                    token.last_check = datetime.utcnow()
                    token.check_count += 1
                    self.stats["tokens_analyzed"] += 1

                    if metrics:
                        token.metrics = metrics
                        token.signal_strength = metrics.signal_strength

                        # Check if we should trade
                        if self._should_trade(token):
                            await self._execute_trade(token)

                await asyncio.sleep(1)

            except Exception as e:
                self.stats["errors"] += 1
                await asyncio.sleep(1)

    def _should_trade(self, token: WatchedToken) -> bool:
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

        return True

    async def _execute_trade(self, token: WatchedToken):
        """Execute a buy trade"""
        token.traded = True
        self.stats["signals_generated"] += 1

        metrics = token.metrics
        mint = token.mint

        # Calculate position size (Kelly-inspired)
        confidence = token.signal_strength
        size_sol = min(
            self.max_position_sol * confidence,
            self.max_position_sol
        )

        print(f"\n[TRADE] SIGNAL: {mint[:20]}...")
        print(f"  Signal Strength: {token.signal_strength:.2f}")
        print(f"  Unique Wallets: {metrics.unique_wallets}")
        print(f"  Buy Pressure: {metrics.buy_pressure:.1%}")
        print(f"  Position Size: {size_sol:.3f} SOL")

        if self.paper_trading:
            print(f"  [PAPER] Simulated buy")
            self.positions[mint] = {
                "mint": mint,
                "entry_time": datetime.utcnow().isoformat(),
                "size_sol": size_sol,
                "entry_price": metrics.price,
                "paper": True,
            }
            self.stats["trades_executed"] += 1

        else:
            # LIVE TRADE - Direct to Solana blockchain
            result = await self.trader.buy(
                mint=mint,
                sol_amount=size_sol,
                slippage=0.05
            )

            if result.get("success"):
                print(f"  [LIVE] Buy executed: {result.get('signature', 'N/A')}")
                self.positions[mint] = {
                    "mint": mint,
                    "entry_time": datetime.utcnow().isoformat(),
                    "size_sol": size_sol,
                    "entry_price": result.get("price", metrics.price),
                    "signature": result.get("signature"),
                    "paper": False,
                }
                self.stats["trades_executed"] += 1
            else:
                print(f"  [ERROR] Buy failed: {result.get('error')}")
                token.traded = False  # Allow retry

    async def _monitor_positions(self):
        """Monitor open positions for exit conditions"""
        print("[POSITIONS] Starting position monitor...")

        while self._running:
            try:
                for mint, position in list(self.positions.items()):
                    # Get current state
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

        print(f"\n[TRADE] EXIT: {mint[:20]}...")
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
            print(f"  Watching: {len(self.watched_tokens)} tokens")
            print(f"  Open positions: {len(self.positions)}")
            print(f"  Errors: {self.stats['errors']}")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Unbreakable Pump.fun Engine")
    parser.add_argument("--keypair", help="Path to Solana keypair")
    parser.add_argument("--rpc", default="https://api.mainnet-beta.solana.com")
    parser.add_argument("--live", action="store_true", help="Enable LIVE trading")
    parser.add_argument("--max-sol", type=float, default=0.5)
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--min-signal", type=float, default=0.7)
    args = parser.parse_args()

    engine = UnbreakableEngine(
        keypair_path=args.keypair,
        rpc_url=args.rpc,
        paper_trading=not args.live,
        max_position_sol=args.max_sol,
        max_positions=args.max_positions,
        min_signal_strength=args.min_signal,
    )

    await engine.start()


if __name__ == "__main__":
    asyncio.run(main())
