"""
Execution Node for Distributed Trading
======================================

Executes trades DIRECTLY on Solana blockchain.
NO pump.fun API dependency.

Subscribes to Redis for signals from discovery nodes.
Uses atomic operations to prevent duplicate trades across cluster.

Usage:
    python -m trading.distributed.execution_node \
        --node-id exec1 \
        --keypair ~/.config/solana/pumpfun.json \
        --redis redis.your-server.com
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    import redis
except ImportError:
    redis = None

# Import our direct trading modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trading.execution.pumpfun_trader import PumpfunTrader, PumpfunBondingCurve
from trading.signals import EarlyPatternScorer, PositionSizer, TokenScore


@dataclass
class Position:
    """Open position tracked across cluster"""
    mint: str
    entry_price: float
    entry_time: str
    size_sol: float
    tokens: int
    executor_id: str
    stop_loss_price: float
    take_profit_price: float


class ExecutionNode:
    """
    Distributed execution node.

    - Receives signals from Redis (published by discovery nodes)
    - Claims positions atomically to prevent duplicates
    - Executes trades directly on Solana (no pump.fun API)
    - Tracks positions in shared Redis state
    """

    def __init__(
        self,
        node_id: str,
        keypair_path: str,
        rpc_url: str = "https://api.mainnet-beta.solana.com",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        paper_trading: bool = True,
        max_position_sol: float = 0.5,
        max_positions: int = 3,
        min_score: float = 0.70,
        stop_loss_pct: float = 0.30,
        take_profit_pct: float = 1.00,
    ):
        self.node_id = node_id
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.paper_trading = paper_trading

        # Risk limits
        self.max_position_sol = max_position_sol
        self.max_positions = max_positions
        self.min_score = min_score
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        # Components
        self.trader = PumpfunTrader(keypair_path, rpc_url)
        self.scorer = EarlyPatternScorer(strict_mode=False)
        self.sizer = PositionSizer(max_position_pct=0.05)

        # State
        self._redis = None
        self._running = False
        self.local_positions: Dict[str, Position] = {}

        # Stats
        self.stats = {
            "signals_received": 0,
            "positions_opened": 0,
            "positions_closed": 0,
            "claim_wins": 0,
            "claim_losses": 0,
            "total_pnl_sol": 0.0,
        }

    async def connect(self) -> bool:
        """Initialize connections"""
        # Connect to Solana
        if not await self.trader.connect():
            print(f"[{self.node_id}] Failed to connect to Solana")
            return False

        balance = await self.trader.get_balance()
        print(f"[{self.node_id}] Solana connected. Balance: {balance:.4f} SOL")

        # Connect to Redis
        if redis is None:
            print(f"[{self.node_id}] WARNING: redis not installed")
            return False

        try:
            self._redis = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                decode_responses=True
            )
            self._redis.ping()
            print(f"[{self.node_id}] Redis connected at {self.redis_host}")

            # Register this executor
            self._redis.hset("execution_nodes", self.node_id, json.dumps({
                "started": datetime.utcnow().isoformat(),
                "paper_trading": self.paper_trading,
                "max_position_sol": self.max_position_sol,
            }))

            return True

        except Exception as e:
            print(f"[{self.node_id}] Redis connection failed: {e}")
            return False

    def can_open_position(self) -> bool:
        """Check if we can open a new position (cluster-wide check)"""
        if not self._redis:
            return len(self.local_positions) < self.max_positions

        # Count positions owned by this node
        all_positions = self._redis.hgetall("positions")
        my_positions = 0

        for mint, pos_json in all_positions.items():
            try:
                pos = json.loads(pos_json)
                if pos.get("executor_id") == self.node_id:
                    my_positions += 1
            except:
                pass

        if my_positions >= self.max_positions:
            return False

        # Check daily P&L circuit breaker
        daily_pnl = float(self._redis.get("daily_pnl") or 0)
        if daily_pnl <= -2.0:  # Lost 2 SOL today
            print(f"[{self.node_id}] Circuit breaker triggered: daily_pnl={daily_pnl:.2f}")
            return False

        return True

    def claim_position(self, mint: str) -> bool:
        """
        Atomically claim a position.

        Uses Redis SETNX to ensure only one executor gets it.
        Returns True if we won the race.
        """
        if not self._redis:
            # No Redis - local check only
            return mint not in self.local_positions

        # Try to set ownership atomically
        key = f"position_claim:{mint}"
        claimed = self._redis.setnx(key, self.node_id)

        if claimed:
            # Set expiry in case we crash
            self._redis.expire(key, 3600)  # 1 hour
            self.stats["claim_wins"] += 1
            return True
        else:
            self.stats["claim_losses"] += 1
            return False

    async def process_signal(self, signal_data: dict):
        """
        Process a token signal from discovery nodes.

        1. Score the token
        2. Check if we should trade
        3. Try to claim position
        4. Execute trade
        """
        self.stats["signals_received"] += 1

        mint = signal_data.get("mint")
        if not mint:
            return

        # Already in position?
        if mint in self.local_positions:
            return

        # Already claimed by another executor?
        if self._redis and self._redis.exists(f"position_claim:{mint}"):
            return

        # Can we open more positions?
        if not self.can_open_position():
            return

        # Score the token (using whatever data we have)
        data = signal_data.get("data", {})
        score = self.scorer.score(
            unique_wallets=data.get("unique_wallets", 5),
            unique_buyers=data.get("unique_buyers", 3),
            total_sol=data.get("total_sol", 1.0),
            buy_sol=data.get("buy_sol", 0.8),
            trade_count=data.get("trade_count", 10),
            age_seconds=data.get("age_seconds", 60),
            price_change=data.get("price_change", 0),
        )
        score.mint = mint

        # Check score threshold
        if score.total_score < self.min_score:
            return

        if score.signal not in ("strong_buy", "buy"):
            return

        # Try to claim position
        if not self.claim_position(mint):
            print(f"[{self.node_id}] Lost race for {mint[:16]}...")
            return

        # Calculate position size
        balance = await self.trader.get_balance()
        size_sol = self.sizer.calculate(
            score=score.total_score,
            confidence=score.confidence,
            portfolio_value=balance
        )
        size_sol = min(size_sol, self.max_position_sol)

        if size_sol < 0.01:  # Minimum 0.01 SOL
            return

        # Execute trade
        await self._execute_buy(mint, size_sol, score)

    async def _execute_buy(self, mint: str, size_sol: float, score: TokenScore):
        """Execute buy trade directly on Solana"""
        # Get current bonding curve state
        state = await self.trader.get_bonding_curve_state(mint)
        if not state:
            print(f"[{self.node_id}] Failed to get state for {mint[:16]}")
            return

        entry_price = PumpfunBondingCurve.get_price_per_token(state)

        # Calculate stop/take prices
        stop_loss_price = entry_price * (1 - self.stop_loss_pct)
        take_profit_price = entry_price * (1 + self.take_profit_pct)

        if self.paper_trading:
            print(f"[{self.node_id}] PAPER BUY {mint[:16]}... | {size_sol:.3f} SOL | score={score.total_score:.2f}")

            # Simulate position
            position = Position(
                mint=mint,
                entry_price=entry_price,
                entry_time=datetime.utcnow().isoformat(),
                size_sol=size_sol,
                tokens=int(size_sol / entry_price) if entry_price > 0 else 0,
                executor_id=self.node_id,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
            )

        else:
            # LIVE TRADE - Direct to Solana blockchain
            result = await self.trader.buy(
                mint=mint,
                sol_amount=size_sol,
                slippage=0.05,
            )

            if not result.get("success"):
                print(f"[{self.node_id}] Buy failed for {mint[:16]}: {result.get('error')}")
                return

            position = Position(
                mint=mint,
                entry_price=result.get("price", entry_price),
                entry_time=datetime.utcnow().isoformat(),
                size_sol=size_sol,
                tokens=result.get("expected_tokens", 0),
                executor_id=self.node_id,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
            )

            print(f"[{self.node_id}] LIVE BUY {mint[:16]}... | {size_sol:.3f} SOL | tx={result.get('signature', 'N/A')}")

        # Record position
        self.local_positions[mint] = position
        self.stats["positions_opened"] += 1

        if self._redis:
            self._redis.hset("positions", mint, json.dumps(asdict(position)))

    async def check_exits(self):
        """Check all positions for exit conditions"""
        for mint, position in list(self.local_positions.items()):
            # Get current price
            state = await self.trader.get_bonding_curve_state(mint)
            if not state:
                continue

            current_price = PumpfunBondingCurve.get_price_per_token(state)

            should_exit = False
            exit_reason = ""

            # Stop loss
            if current_price <= position.stop_loss_price:
                should_exit = True
                exit_reason = "STOP_LOSS"

            # Take profit
            if current_price >= position.take_profit_price:
                should_exit = True
                exit_reason = "TAKE_PROFIT"

            # Time-based exit (30 min max hold)
            entry_time = datetime.fromisoformat(position.entry_time)
            if (datetime.utcnow() - entry_time).total_seconds() > 1800:
                should_exit = True
                exit_reason = "TIME_EXIT"

            if should_exit:
                await self._execute_sell(mint, current_price, exit_reason)

    async def _execute_sell(self, mint: str, exit_price: float, reason: str):
        """Execute sell trade"""
        position = self.local_positions.get(mint)
        if not position:
            return

        # Calculate P&L
        if position.entry_price > 0:
            pnl_pct = (exit_price - position.entry_price) / position.entry_price
            pnl_sol = position.size_sol * pnl_pct
        else:
            pnl_pct = 0
            pnl_sol = 0

        if self.paper_trading:
            print(f"[{self.node_id}] PAPER SELL {mint[:16]}... | {reason} | PnL: {pnl_sol:+.4f} SOL ({pnl_pct:+.1%})")

        else:
            # LIVE TRADE - Direct to Solana
            result = await self.trader.sell(mint, token_pct=1.0)
            if result.get("success"):
                print(f"[{self.node_id}] LIVE SELL {mint[:16]}... | {reason} | PnL: {pnl_sol:+.4f} SOL")

        # Update stats
        self.stats["positions_closed"] += 1
        self.stats["total_pnl_sol"] += pnl_sol

        # Remove position
        del self.local_positions[mint]

        if self._redis:
            self._redis.hdel("positions", mint)
            self._redis.delete(f"position_claim:{mint}")

            # Update daily P&L
            self._redis.incrbyfloat("daily_pnl", pnl_sol)

            # Record trade in history
            self._redis.lpush("trade_history", json.dumps({
                "mint": mint,
                "executor": self.node_id,
                "entry_price": position.entry_price,
                "exit_price": exit_price,
                "size_sol": position.size_sol,
                "pnl_sol": pnl_sol,
                "pnl_pct": pnl_pct,
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat(),
            }))

    async def run(self):
        """
        Main execution loop.

        1. Subscribe to Redis signal channel
        2. Process incoming signals
        3. Monitor positions for exits
        """
        if not await self.connect():
            return

        self._running = True

        # Start background tasks
        tasks = [
            self._listen_for_signals(),
            self._monitor_positions(),
            self._report_stats(),
        ]

        mode = "PAPER" if self.paper_trading else "LIVE"
        print(f"[{self.node_id}] Starting execution node ({mode} TRADING)")

        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            self._running = False
            print(f"\n[{self.node_id}] Shutting down...")
            print(f"Final stats: {self.stats}")
            await self.trader.close()

    async def _listen_for_signals(self):
        """Subscribe to Redis signal channel"""
        if not self._redis:
            print(f"[{self.node_id}] No Redis - cannot listen for signals")
            return

        pubsub = self._redis.pubsub()
        pubsub.subscribe("token_signals")

        print(f"[{self.node_id}] Subscribed to token_signals channel")

        while self._running:
            try:
                message = pubsub.get_message(timeout=1.0)

                if message and message["type"] == "message":
                    signal_data = json.loads(message["data"])
                    await self.process_signal(signal_data)

            except Exception as e:
                print(f"[{self.node_id}] Signal error: {e}")
                await asyncio.sleep(1)

    async def _monitor_positions(self):
        """Periodically check positions for exits"""
        while self._running:
            try:
                await self.check_exits()
            except Exception as e:
                print(f"[{self.node_id}] Monitor error: {e}")

            await asyncio.sleep(5)  # Check every 5 seconds

    async def _report_stats(self):
        """Report stats periodically"""
        while self._running:
            await asyncio.sleep(60)

            print(f"[{self.node_id}] Stats: {self.stats}")

            if self._redis:
                self._redis.hset("executor_stats", self.node_id, json.dumps({
                    **self.stats,
                    "open_positions": len(self.local_positions),
                    "timestamp": datetime.utcnow().isoformat(),
                }))


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Execution Node")
    parser.add_argument("--node-id", required=True, help="Unique node identifier")
    parser.add_argument("--keypair", required=True, help="Path to Solana keypair")
    parser.add_argument("--rpc", default="https://api.mainnet-beta.solana.com")
    parser.add_argument("--redis", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--live", action="store_true", help="Enable LIVE trading")
    parser.add_argument("--max-sol", type=float, default=0.5)
    parser.add_argument("--max-positions", type=int, default=3)
    args = parser.parse_args()

    node = ExecutionNode(
        node_id=args.node_id,
        keypair_path=args.keypair,
        rpc_url=args.rpc,
        redis_host=args.redis,
        redis_port=args.redis_port,
        paper_trading=not args.live,
        max_position_sol=args.max_sol,
        max_positions=args.max_positions,
    )

    await node.run()


if __name__ == "__main__":
    asyncio.run(main())
