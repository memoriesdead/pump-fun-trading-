"""
Live Pump.fun Scalping Engine
=============================

Full signal-to-execution pipeline for pump.fun trading.

Architecture:
    [WebSocket] -> [Aggregator] -> [Scorer] -> [Sizer] -> [Trader]

Usage:
    python -m trading.execution.live_scalper --keypair ~/.config/solana/pumpfun.json
"""

import asyncio
import json
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from ..signals import EarlyPatternScorer, PositionSizer, TokenScore
from .pumpfun_trader import PumpfunTrader, PumpfunBondingCurve, BondingCurveState


@dataclass
class TokenState:
    """Real-time state for a single token"""
    mint: str
    first_seen: datetime
    trades: List[dict] = field(default_factory=list)
    unique_wallets: set = field(default_factory=set)
    unique_buyers: set = field(default_factory=set)
    total_sol: float = 0.0
    buy_sol: float = 0.0
    last_price: float = 0.0
    first_price: float = 0.0

    @property
    def age_seconds(self) -> float:
        return (datetime.now() - self.first_seen).total_seconds()

    @property
    def trade_count(self) -> int:
        return len(self.trades)

    @property
    def price_change(self) -> float:
        if self.first_price == 0:
            return 0
        return (self.last_price - self.first_price) / self.first_price


@dataclass
class Position:
    """Open position"""
    mint: str
    entry_price: float
    entry_time: datetime
    size_sol: float
    tokens: int
    stop_loss_price: float
    take_profit_price: float


class LiveTokenAggregator:
    """
    Real-time token aggregation from trade stream.

    Maintains rolling window of metrics per token.
    """

    def __init__(self, window_minutes: int = 5):
        self.window_minutes = window_minutes
        self.tokens: Dict[str, TokenState] = {}
        self.expired: set = set()

    def update(self, trade: dict) -> Optional[TokenState]:
        """
        Update aggregator with new trade.

        Args:
            trade: Dict with mint, wallet, sol, tokens, type

        Returns:
            Updated TokenState if within window, None if expired
        """
        mint = trade.get("mint")
        if not mint or mint in self.expired:
            return None

        now = datetime.now()

        if mint not in self.tokens:
            self.tokens[mint] = TokenState(
                mint=mint,
                first_seen=now
            )

        token = self.tokens[mint]

        # Check if expired (beyond window)
        if token.age_seconds > self.window_minutes * 60:
            self.expired.add(mint)
            del self.tokens[mint]
            return None

        # Update metrics
        wallet = trade.get("wallet", "")
        sol = trade.get("sol", 0)
        tokens = trade.get("tokens", 0)
        trade_type = trade.get("type", "")

        token.trades.append(trade)
        token.unique_wallets.add(wallet)
        token.total_sol += sol

        if trade_type == "buy":
            token.unique_buyers.add(wallet)
            token.buy_sol += sol

        # Update price
        if tokens > 0:
            price = sol / tokens
            token.last_price = price
            if token.first_price == 0:
                token.first_price = price

        return token

    def get_active_tokens(self) -> List[TokenState]:
        """Get all tokens within active window"""
        return list(self.tokens.values())


class RiskManager:
    """
    Risk management with hard limits.

    Controls:
    - Max position size
    - Max concurrent positions
    - Daily loss limit
    - Stop loss / take profit
    """

    def __init__(
        self,
        max_position_sol: float = 0.5,
        max_positions: int = 3,
        daily_loss_limit_sol: float = 2.0,
        stop_loss_pct: float = 0.30,
        take_profit_pct: float = 1.00,
    ):
        self.max_position_sol = max_position_sol
        self.max_positions = max_positions
        self.daily_loss_limit_sol = daily_loss_limit_sol
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        # Track daily P&L
        self.daily_pnl = 0.0
        self.last_reset = datetime.now().date()

    def can_open_position(self, current_positions: int) -> bool:
        """Check if new position allowed"""
        self._maybe_reset_daily()

        if current_positions >= self.max_positions:
            return False

        if self.daily_pnl <= -self.daily_loss_limit_sol:
            return False

        return True

    def calculate_stop_take(self, entry_price: float) -> tuple:
        """Calculate stop loss and take profit prices"""
        stop_loss = entry_price * (1 - self.stop_loss_pct)
        take_profit = entry_price * (1 + self.take_profit_pct)
        return (stop_loss, take_profit)

    def clamp_size(self, size_sol: float) -> float:
        """Clamp position size to maximum"""
        return min(size_sol, self.max_position_sol)

    def record_pnl(self, pnl_sol: float):
        """Record trade P&L"""
        self._maybe_reset_daily()
        self.daily_pnl += pnl_sol

    def _maybe_reset_daily(self):
        """Reset daily P&L at midnight"""
        today = datetime.now().date()
        if today > self.last_reset:
            self.daily_pnl = 0.0
            self.last_reset = today


class LiveScalpingEngine:
    """
    Complete live scalping engine.

    Integrates:
    - Token aggregation
    - Signal scoring
    - Position sizing
    - Risk management
    - Trade execution
    """

    def __init__(
        self,
        trader: PumpfunTrader,
        scorer: EarlyPatternScorer = None,
        sizer: PositionSizer = None,
        risk_manager: RiskManager = None,
        min_score: float = 0.70,
        paper_trading: bool = True,
    ):
        self.trader = trader
        self.scorer = scorer or EarlyPatternScorer()
        self.sizer = sizer or PositionSizer()
        self.risk_manager = risk_manager or RiskManager()
        self.min_score = min_score
        self.paper_trading = paper_trading

        self.aggregator = LiveTokenAggregator()
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[dict] = []

        # Stats
        self.signals_generated = 0
        self.trades_executed = 0
        self.total_pnl = 0.0

    async def process_trade(self, trade: dict):
        """
        Process incoming trade from WebSocket.

        Full pipeline:
        1. Update aggregator
        2. Score token if new enough
        3. Check for entry signals
        4. Check for exit signals
        """
        # Update aggregator
        token = self.aggregator.update(trade)
        if not token:
            return

        # Check exit conditions for existing positions
        mint = trade.get("mint")
        if mint in self.positions:
            await self._check_exit(mint, trade)

        # Only consider new entries for young tokens
        if token.age_seconds > 300:  # 5 minutes
            return

        # Score token
        score = self.scorer.score(
            unique_wallets=len(token.unique_wallets),
            unique_buyers=len(token.unique_buyers),
            total_sol=token.total_sol,
            buy_sol=token.buy_sol,
            trade_count=token.trade_count,
            age_seconds=token.age_seconds,
            price_change=token.price_change,
        )
        score.mint = mint

        self.signals_generated += 1

        # Check entry conditions
        if score.signal in ("strong_buy", "buy") and score.total_score >= self.min_score:
            await self._try_enter(mint, token, score)

    async def _try_enter(self, mint: str, token: TokenState, score: TokenScore):
        """Attempt to enter position"""
        # Already in position?
        if mint in self.positions:
            return

        # Risk check
        if not self.risk_manager.can_open_position(len(self.positions)):
            print(f"[RISK] Cannot open position for {mint[:8]}...")
            return

        # Get portfolio value (simplified - would fetch from wallet)
        portfolio_value = await self.trader.get_balance()

        # Calculate position size
        raw_size = self.sizer.calculate(
            score=score.total_score,
            confidence=score.confidence,
            portfolio_value=portfolio_value
        )
        size_sol = self.risk_manager.clamp_size(raw_size)

        if size_sol < 0.01:  # Min 0.01 SOL
            return

        # Calculate stop/take
        stop_loss, take_profit = self.risk_manager.calculate_stop_take(token.last_price)

        # Execute trade
        if self.paper_trading:
            print(f"[PAPER] BUY {mint[:8]}... | {size_sol:.3f} SOL | score={score.total_score:.2f}")
            # Simulate position
            self.positions[mint] = Position(
                mint=mint,
                entry_price=token.last_price,
                entry_time=datetime.now(),
                size_sol=size_sol,
                tokens=0,  # Would be filled from actual trade
                stop_loss_price=stop_loss,
                take_profit_price=take_profit,
            )
        else:
            result = await self.trader.buy(mint, size_sol)
            if result.get("success"):
                self.positions[mint] = Position(
                    mint=mint,
                    entry_price=result.get("price", token.last_price),
                    entry_time=datetime.now(),
                    size_sol=size_sol,
                    tokens=result.get("expected_tokens", 0),
                    stop_loss_price=stop_loss,
                    take_profit_price=take_profit,
                )
                print(f"[LIVE] BUY {mint[:8]}... | {size_sol:.3f} SOL | tx={result.get('signature', 'N/A')}")

        self.trades_executed += 1

    async def _check_exit(self, mint: str, trade: dict):
        """Check if position should be closed"""
        position = self.positions.get(mint)
        if not position:
            return

        current_price = trade.get("sol", 0) / trade.get("tokens", 1) if trade.get("tokens") else 0
        if current_price == 0:
            return

        should_exit = False
        exit_reason = ""

        # Stop loss check
        if current_price <= position.stop_loss_price:
            should_exit = True
            exit_reason = "STOP_LOSS"

        # Take profit check
        if current_price >= position.take_profit_price:
            should_exit = True
            exit_reason = "TAKE_PROFIT"

        # Time-based exit (30 min max hold)
        if (datetime.now() - position.entry_time).total_seconds() > 1800:
            should_exit = True
            exit_reason = "TIME_EXIT"

        if should_exit:
            await self._exit_position(mint, current_price, exit_reason)

    async def _exit_position(self, mint: str, exit_price: float, reason: str):
        """Exit position"""
        position = self.positions.get(mint)
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
            print(f"[PAPER] SELL {mint[:8]}... | {reason} | PnL: {pnl_sol:+.4f} SOL ({pnl_pct:+.1%})")
        else:
            result = await self.trader.sell(mint, token_pct=1.0)
            if result.get("success"):
                print(f"[LIVE] SELL {mint[:8]}... | {reason} | PnL: {pnl_sol:+.4f} SOL")

        # Record
        self.risk_manager.record_pnl(pnl_sol)
        self.total_pnl += pnl_sol

        self.trade_history.append({
            "mint": mint,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "size_sol": position.size_sol,
            "pnl_sol": pnl_sol,
            "pnl_pct": pnl_pct,
            "reason": reason,
            "hold_time": (datetime.now() - position.entry_time).total_seconds(),
        })

        del self.positions[mint]

    def get_stats(self) -> dict:
        """Get engine statistics"""
        return {
            "signals_generated": self.signals_generated,
            "trades_executed": self.trades_executed,
            "open_positions": len(self.positions),
            "total_pnl_sol": self.total_pnl,
            "daily_pnl_sol": self.risk_manager.daily_pnl,
            "trade_history_count": len(self.trade_history),
        }


async def run_live_scalper(
    keypair_path: str,
    rpc_url: str = "https://api.mainnet-beta.solana.com",
    paper_trading: bool = True,
):
    """
    Run live scalping engine.

    Args:
        keypair_path: Path to Solana keypair JSON
        rpc_url: Solana RPC endpoint
        paper_trading: If True, simulate trades without execution
    """
    # Initialize trader
    trader = PumpfunTrader(keypair_path, rpc_url)
    if not await trader.connect():
        print("Failed to connect to Solana")
        return

    balance = await trader.get_balance()
    print(f"Connected. Balance: {balance:.4f} SOL")
    print(f"Mode: {'PAPER TRADING' if paper_trading else 'LIVE TRADING'}")

    # Initialize engine
    engine = LiveScalpingEngine(
        trader=trader,
        paper_trading=paper_trading,
        min_score=0.70,
    )

    print("Scalping engine ready. Waiting for trades...")

    # In production, connect to WebSocket and process trades
    # For now, simulate with placeholder
    try:
        while True:
            # Would receive trades from WebSocket here
            # trade = await websocket.recv()
            # await engine.process_trade(trade)

            await asyncio.sleep(1)

            # Print stats every minute
            stats = engine.get_stats()
            if stats["signals_generated"] % 100 == 0 and stats["signals_generated"] > 0:
                print(f"Stats: {stats}")

    except KeyboardInterrupt:
        print("\nShutting down...")
        stats = engine.get_stats()
        print(f"Final stats: {stats}")

    finally:
        await trader.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pump.fun Live Scalper")
    parser.add_argument("--keypair", required=True, help="Path to keypair JSON")
    parser.add_argument("--rpc", default="https://api.mainnet-beta.solana.com")
    parser.add_argument("--live", action="store_true", help="Enable live trading")
    args = parser.parse_args()

    asyncio.run(run_live_scalper(
        keypair_path=args.keypair,
        rpc_url=args.rpc,
        paper_trading=not args.live,
    ))
