"""
Active Token Scanner & Trader
=============================

Watches ALL incoming trades on pump.fun and identifies the hottest tokens:
1. High volume tokens (lots of SOL flowing)
2. Strong buy momentum
3. Many unique traders

Builds a dynamic watchlist from real-time trade flow.

Usage:
    python -m trading.active_scanner --mode paper --capital 100
"""

import asyncio
import aiohttp
import json
import time
import argparse
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from collections import defaultdict

from .engine import TradingEngine
from .config import TradingConfig, DEFAULT_CONFIG, AGGRESSIVE_CONFIG
from .models import Position
from .executors.paper import PaperExecutor
from .executors.real import RealExecutor

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================
# CONSTANTS
# ============================================================

PUMPPORTAL_WS = "wss://pumpportal.fun/api/data"

# Scanner settings - ULTRA AGGRESSIVE for testing
MIN_VOLUME_1M = 0.01  # Almost any volume
MIN_VOLUME_5M = 0.05  # Almost any volume
MIN_TRADES_1M = 1     # Minimum 1 trade
HOT_TOKEN_THRESHOLD = 0.5  # Very low threshold
MAX_TOKENS_WATCH = 500    # Track lots of tokens


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class LiveToken:
    """Token tracked from live trade stream"""
    mint: str
    symbol: str = "???"
    name: str = ""
    bonding_curve: str = ""
    creator: str = ""

    # Price data
    price_sol: float = 0.0
    virtual_sol: float = 0.0
    virtual_tokens: float = 0.0
    market_cap_sol: float = 0.0

    # Trade tracking
    trades: List[Dict] = field(default_factory=list)
    unique_traders: Set[str] = field(default_factory=set)

    # Timestamps
    created_at: float = 0.0
    last_trade: float = 0.0

    def volume_1m(self) -> float:
        """Volume in last 1 minute"""
        cutoff = time.time() - 60
        return sum(t['sol_amount'] for t in self.trades if t['timestamp'] > cutoff)

    def volume_5m(self) -> float:
        """Volume in last 5 minutes"""
        cutoff = time.time() - 300
        return sum(t['sol_amount'] for t in self.trades if t['timestamp'] > cutoff)

    def trades_1m(self) -> int:
        """Trade count in last minute"""
        cutoff = time.time() - 60
        return sum(1 for t in self.trades if t['timestamp'] > cutoff)

    def buy_ratio_1m(self) -> float:
        """Buy ratio in last minute (0-1)"""
        cutoff = time.time() - 60
        recent = [t for t in self.trades if t['timestamp'] > cutoff]
        if not recent:
            return 0.5
        buys = sum(1 for t in recent if t['is_buy'])
        return buys / len(recent)

    def bonding_progress(self) -> float:
        """Estimated bonding curve progress (0-1)"""
        if self.virtual_sol > 0:
            # 85 SOL graduation threshold
            real_sol = max(0, self.virtual_sol - 30)  # Subtract initial virtual
            return min(1.0, real_sol / 85)
        return 0.0

    def hotness_score(self) -> float:
        """
        Score how "hot" this token is right now.
        Higher = more active, more momentum
        """
        vol_1m = self.volume_1m()
        vol_5m = self.volume_5m()
        trades_1m = self.trades_1m()
        buy_ratio = self.buy_ratio_1m()
        traders = len(self.unique_traders)
        progress = self.bonding_progress()

        # Volume score (0-10)
        vol_score = min(vol_1m * 5, 10)  # 2 SOL/min = max score

        # Trade frequency (0-5)
        freq_score = min(trades_1m, 5)

        # Buy momentum (0-5)
        momentum_score = (buy_ratio - 0.5) * 10  # -5 to +5

        # Trader diversity (0-3)
        trader_score = min(traders / 10, 3)

        # Bonding progress bonus (0-2)
        progress_score = progress * 2

        return vol_score + freq_score + momentum_score + trader_score + progress_score

    def age_seconds(self) -> float:
        """Token age in seconds"""
        if self.created_at > 0:
            return time.time() - self.created_at
        return 0

    def prune_old_trades(self, max_age: float = 600):
        """Remove trades older than max_age seconds"""
        cutoff = time.time() - max_age
        self.trades = [t for t in self.trades if t['timestamp'] > cutoff]


@dataclass
class ScannerStats:
    """Scanner statistics"""
    start_time: float = field(default_factory=time.time)
    tokens_seen: int = 0
    tokens_hot: int = 0
    trades_received: int = 0
    trades_entered: int = 0
    trades_exited: int = 0
    trades_won: int = 0
    trades_lost: int = 0
    total_pnl_sol: float = 0.0
    total_volume: float = 0.0

    @property
    def uptime_str(self) -> str:
        secs = int(time.time() - self.start_time)
        hours, remainder = divmod(secs, 3600)
        mins, secs = divmod(remainder, 60)
        return f"{hours:02d}:{mins:02d}:{secs:02d}"

    @property
    def win_rate(self) -> float:
        total = self.trades_won + self.trades_lost
        return self.trades_won / total if total > 0 else 0.0


# ============================================================
# ACTIVE TOKEN TRADER
# ============================================================

class ActiveTokenTrader:
    """
    Watches ALL trades on pump.fun and identifies the hottest tokens.

    Strategy:
    1. Subscribe to subscribeNewToken to discover new tokens
    2. Subscribe to specific token trades after discovery
    3. Build dynamic watchlist based on real-time activity
    4. Trade tokens with strong momentum signals

    NOTE: PumpPortal requires specific mint addresses for subscribeTokenTrade.
    You CANNOT subscribe to ALL trades with empty keys.
    """

    def __init__(
        self,
        paper_mode: bool = True,
        capital: float = 100.0,
        keypair_path: Optional[str] = None,
        rpc_url: str = "https://api.mainnet-beta.solana.com",
        config: Optional[TradingConfig] = None,
    ):
        self.paper_mode = paper_mode
        self.capital = capital
        self.rpc_url = rpc_url
        # Use AGGRESSIVE config for paper trading to get more action
        self.config = config or (AGGRESSIVE_CONFIG if paper_mode else DEFAULT_CONFIG)

        # Engine
        self.engine = TradingEngine(
            paper_mode=paper_mode,
            capital=capital,
            config=self.config,
        )
        logger.info(f"Config: min_quality_signals={self.config.min_quality_signals}, stop_loss={self.config.stop_loss}")

        # Executor
        if paper_mode:
            self.executor = PaperExecutor()
        else:
            if not keypair_path:
                raise ValueError("keypair_path required for real trading")
            self.executor = RealExecutor(keypair_path=keypair_path, rpc_url=rpc_url)

        self.engine.executor = self.executor

        # State - track ALL tokens we see trades for
        self.running = False
        self.stats = ScannerStats()
        self.tokens: Dict[str, LiveToken] = {}  # mint -> LiveToken

        # WebSocket subscription management
        self.ws = None
        self.subscribed_mints: Set[str] = set()  # Mints we're subscribed to
        self.pending_subscribe: List[str] = []  # Mints waiting to be subscribed

    async def start(self):
        """Start the active trader"""
        self.running = True
        self.stats.start_time = time.time()

        # Print banner
        mode_str = 'PAPER' if self.paper_mode else 'REAL'
        print(f"""
+====================================================================+
|              ACTIVE TOKEN SCANNER & TRADER                         |
+====================================================================+
|  Mode:     {mode_str:12}                                         |
|  Capital:  {self.capital:>8.2f} SOL                                       |
|  Strategy: Trade hottest tokens by volume & momentum               |
|  Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):19}                            |
+====================================================================+
        """)

        logger.info("Connecting to pump.fun trade stream...")

        # Run loops
        await asyncio.gather(
            self._websocket_loop(),
            self._position_monitor_loop(),
            self._status_loop(),
            self._cleanup_loop(),
            return_exceptions=True,
        )

    async def stop(self):
        """Stop gracefully"""
        logger.info("Shutting down...")
        self.running = False
        self._print_final_report()

    async def _subscribe_to_tokens(self, mints: List[str]):
        """Subscribe to specific token trades via WebSocket"""
        if not self.ws or not mints:
            return

        # Filter out already subscribed
        new_mints = [m for m in mints if m not in self.subscribed_mints]
        if not new_mints:
            return

        # Batch limit of 20 per request
        batch = new_mints[:20]
        try:
            await self.ws.send_json({
                "method": "subscribeTokenTrade",
                "keys": batch
            })
            for mint in batch:
                self.subscribed_mints.add(mint)
            logger.info(f"Subscribed to {len(batch)} token(s) trades | Total: {len(self.subscribed_mints)}")
        except Exception as e:
            logger.error(f"Subscription error: {e}")

    async def _websocket_loop(self):
        """WebSocket connection - discovers tokens and subscribes to their trades"""
        reconnect_delay = 5

        while self.running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(
                        PUMPPORTAL_WS,
                        timeout=aiohttp.ClientTimeout(total=30),
                        heartbeat=30,
                    ) as ws:
                        self.ws = ws

                        # Only subscribe to new token creations
                        # We'll then subscribe to specific token trades after discovery
                        await ws.send_json({"method": "subscribeNewToken"})
                        logger.info("Connected! Listening for new tokens...")
                        logger.info("Will subscribe to token trades as we discover them...")

                        reconnect_delay = 5
                        last_sub_check = time.time()

                        async for msg in ws:
                            if not self.running:
                                break

                            if msg.type == aiohttp.WSMsgType.TEXT:
                                await self._handle_message(msg.data)

                                # Periodically process pending subscriptions
                                if self.pending_subscribe and (time.time() - last_sub_check) > 1:
                                    await self._subscribe_to_tokens(self.pending_subscribe[:20])
                                    self.pending_subscribe = self.pending_subscribe[20:]
                                    last_sub_check = time.time()

                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")

            # Reset subscriptions on disconnect
            self.subscribed_mints.clear()
            self.ws = None

            if self.running:
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60)

    async def _handle_message(self, data: str):
        """Handle WebSocket message"""
        try:
            msg = json.loads(data)

            # Skip confirmation messages
            if 'message' in msg:
                return

            # Error message
            if 'errors' in msg:
                logger.error(f"WebSocket error: {msg['errors']}")
                return

            tx_type = msg.get('txType', '')

            # Token creation event (txType == "create")
            if tx_type == 'create' and 'mint' in msg:
                await self._on_new_token(msg)

            # Trade event (buy or sell)
            elif tx_type in ('buy', 'sell') and 'mint' in msg:
                await self._on_trade(msg)

        except Exception:
            pass

    async def _on_new_token(self, msg: dict):
        """Handle new token creation"""
        mint = msg.get('mint', '')
        if not mint:
            return

        self.stats.tokens_seen += 1
        symbol = msg.get('symbol', '???')
        logger.info(f"NEW TOKEN: {symbol:12} | Mint: {mint[:16]}...")

        # Create token entry
        if mint not in self.tokens:
            self.tokens[mint] = LiveToken(
                mint=mint,
                symbol=symbol,
                name=msg.get('name', ''),
                bonding_curve=msg.get('bondingCurveKey', ''),
                creator=msg.get('traderPublicKey', ''),
                created_at=time.time(),
                virtual_sol=float(msg.get('vSolInBondingCurve', 30)),
                virtual_tokens=float(msg.get('vTokensInBondingCurve', 1e9)),
                market_cap_sol=float(msg.get('marketCapSol', 0)),
            )

            # Calculate initial price
            token = self.tokens[mint]
            if token.virtual_tokens > 0:
                token.price_sol = token.virtual_sol / token.virtual_tokens

            # Queue for trade subscription - THIS IS THE KEY FIX
            # We subscribe to token trades after discovering the token
            if mint not in self.subscribed_mints and mint not in self.pending_subscribe:
                self.pending_subscribe.append(mint)

    async def _on_trade(self, msg: dict):
        """Handle trade event"""
        mint = msg.get('mint', '')
        if not mint:
            return

        self.stats.trades_received += 1

        # Log every trade for debugging
        is_buy = msg.get('txType', '').lower() == 'buy'
        sol_amount = float(msg.get('solAmount', 0)) / 1e9
        symbol = msg.get('symbol', '???')
        action = "BUY " if is_buy else "SELL"
        if sol_amount > 0.01:  # Log trades > 0.01 SOL
            logger.info(f"TRADE: {symbol:8} | {action} {sol_amount:.4f} SOL | Mint: {mint[:12]}...")

        # Create token if we haven't seen it
        if mint not in self.tokens:
            self.tokens[mint] = LiveToken(
                mint=mint,
                symbol=msg.get('symbol', '???'),
            )

        token = self.tokens[mint]

        # Parse trade
        is_buy = msg.get('txType', '').lower() == 'buy'
        sol_amount = float(msg.get('solAmount', 0)) / 1e9
        token_amount = int(msg.get('tokenAmount', 0))
        trader = msg.get('traderPublicKey', '')

        # Record trade
        token.trades.append({
            'is_buy': is_buy,
            'sol_amount': sol_amount,
            'token_amount': token_amount,
            'trader': trader,
            'timestamp': time.time(),
        })
        token.unique_traders.add(trader)
        token.last_trade = time.time()

        # Update price from reserves
        v_sol = float(msg.get('vSolInBondingCurve', 0))
        v_tokens = float(msg.get('vTokensInBondingCurve', 0))
        if v_sol > 0 and v_tokens > 0:
            token.virtual_sol = v_sol
            token.virtual_tokens = v_tokens
            token.price_sol = v_sol / v_tokens

        token.market_cap_sol = float(msg.get('marketCapSol', 0))

        # Check if this token is HOT
        score = token.hotness_score()
        if score >= HOT_TOKEN_THRESHOLD:
            self.stats.tokens_hot = len([t for t in self.tokens.values() if t.hotness_score() >= HOT_TOKEN_THRESHOLD])

            # Log hot tokens occasionally
            if sol_amount >= 1.0:
                action = "BUY " if is_buy else "SELL"
                logger.info(f"HOT: {token.symbol} | {action} {sol_amount:.2f} SOL | Score: {score:.1f} | Vol/1m: {token.volume_1m():.2f}")

        # Evaluate for entry if not already in position
        if mint not in self.engine.positions:
            await self._evaluate_entry(token)
        else:
            # Update position PnL
            position = self.engine.positions[mint]
            if token.price_sol > 0:
                current_value = position.tokens * token.price_sol
                position.current_pnl = (current_value - position.sol_invested) / position.sol_invested
                if position.current_pnl > position.peak_pnl:
                    position.peak_pnl = position.current_pnl

    async def _evaluate_entry(self, token: LiveToken):
        """Evaluate token for entry"""
        score = token.hotness_score()

        # Entry criteria (RELAXED for testing):
        # 1. Hot enough (score >= threshold)
        # 2. Some buy momentum (>50% buys)
        # 3. Any recent volume
        # 4. At least 1 trader

        if score < HOT_TOKEN_THRESHOLD:
            return

        if token.buy_ratio_1m() < 0.5:
            return

        if token.volume_1m() < MIN_VOLUME_1M:
            return

        if len(token.unique_traders) < 1:
            return

        # Don't enter very old tokens
        if token.age_seconds() > 600:  # 10 min max age
            return

        # Build token data for engine
        token_data = {
            'mint': token.mint,
            'symbol': token.symbol,
            'name': token.name,
            'price': token.price_sol,
            'liquidity_sol': token.virtual_sol,
            'bonding_curve_progress': token.bonding_progress(),
            'unique_traders': len(token.unique_traders),
            'volume_5m': token.volume_5m(),
        }

        # TEMPORARY: Use simple entry logic for testing
        # Bypass signal engine to verify trading flow works
        if score >= 5.0 and token.volume_1m() >= 0.5 and len(self.engine.positions) < 3:
            # Direct paper entry for testing
            position_size = min(self.config.max_position_sol, self.capital * 0.1)
            if position_size < 0.01 or token.price_sol <= 0:
                return

            from .models import Position
            position = Position(
                mint=token.mint,
                entry_price=token.price_sol,
                entry_time=time.time(),
                tokens=position_size / token.price_sol,
                sol_invested=position_size,
                entry_signals={},
                paper_mode=True,
                tx_signature="paper_test",
            )
            self.engine.positions[token.mint] = position
            self.capital -= position_size
            self.stats.trades_entered += 1
            self.stats.total_volume += position_size

            print(f"""
======================================================================
  BUY (TEST) | {datetime.now().strftime('%H:%M:%S')} | {token.symbol}
======================================================================
  Mint:      {token.mint[:20]}...
  Size:      {position_size:.4f} SOL
  Price:     {token.price_sol:.10f} SOL/token
  Hotness:   {score:.1f}
  Vol/1m:    {token.volume_1m():.2f} SOL
  Buy Ratio: {token.buy_ratio_1m():.0%}
======================================================================
""")
            return

        decision = await self.engine.on_new_token(
            token_data=token_data,
            trades=token.trades,
        )

        if decision.action == 'BUY':
            self.stats.trades_entered += 1
            self.stats.total_volume += decision.size or 0

            print(f"""
======================================================================
  BUY | {datetime.now().strftime('%H:%M:%S')} | {token.symbol}
======================================================================
  Mint:      {token.mint[:20]}...{token.mint[-8:] if len(token.mint) > 8 else ''}
  Size:      {decision.size:.4f} SOL
  Price:     {token.price_sol:.10f} SOL/token
  -------------------------------------
  Hotness:   {score:.1f}
  Vol/1m:    {token.volume_1m():.2f} SOL
  Vol/5m:    {token.volume_5m():.2f} SOL
  Buy Ratio: {token.buy_ratio_1m():.0%}
  Traders:   {len(token.unique_traders)}
  Progress:  {token.bonding_progress():.1%}
  Age:       {token.age_seconds():.0f}s
======================================================================
""")
        elif decision.action == 'SKIP' and decision.reason:
            # Log skips occasionally to understand why we're not entering
            if not hasattr(self, '_skip_counter'):
                self._skip_counter = 0
            self._skip_counter += 1

            # Only log every 100th skip to avoid spam
            if self._skip_counter % 100 == 1:
                logger.info(f"SKIP #{self._skip_counter}: {token.symbol} - {decision.reason}")

    async def _position_monitor_loop(self):
        """Monitor positions for exits"""
        while self.running:
            try:
                for mint, position in list(self.engine.positions.items()):
                    token = self.tokens.get(mint)
                    if not token or token.price_sol <= 0:
                        continue

                    # Check exit via engine
                    token_data = {
                        'mint': mint,
                        'symbol': token.symbol,
                        'current_price': token.price_sol,
                    }

                    exit_decision = await self.engine.on_price_update(
                        mint=mint,
                        price=token.price_sol,
                        token_data=token_data,
                    )

                    if exit_decision and exit_decision.action == 'SELL':
                        self.stats.trades_exited += 1

                        pnl_sol = position.current_pnl * position.sol_invested
                        self.stats.total_pnl_sol += pnl_sol
                        self.stats.total_volume += abs(pnl_sol + position.sol_invested)

                        if pnl_sol >= 0:
                            self.stats.trades_won += 1
                        else:
                            self.stats.trades_lost += 1

                        pnl_color = '\033[32m' if pnl_sol >= 0 else '\033[31m'
                        reset = '\033[0m'

                        print(f"""
======================================================================
  SELL | {datetime.now().strftime('%H:%M:%S')} | {token.symbol}
======================================================================
  {pnl_color}PnL:       {pnl_sol:+.4f} SOL ({position.current_pnl:+.1%}){reset}
  Hold Time: {time.time() - position.entry_time:.1f}s
  Reason:    {exit_decision.reason}
======================================================================
""")

            except Exception as e:
                logger.error(f"Position monitor error: {e}")

            await asyncio.sleep(0.5)

    async def _status_loop(self):
        """Print status periodically"""
        while self.running:
            await asyncio.sleep(30)

            if not self.running:
                break

            pnl_color = '\033[32m' if self.stats.total_pnl_sol >= 0 else '\033[31m'
            reset = '\033[0m'

            logger.info(
                f"STATUS | Up: {self.stats.uptime_str} | "
                f"Tokens: {len(self.tokens)} | "
                f"Hot: {self.stats.tokens_hot} | "
                f"Trades: {self.stats.trades_received} | "
                f"In/Out: {self.stats.trades_entered}/{self.stats.trades_exited} | "
                f"W/L: {self.stats.trades_won}/{self.stats.trades_lost} | "
                f"{pnl_color}PnL: {self.stats.total_pnl_sol:+.4f} SOL{reset} | "
                f"Open: {len(self.engine.positions)}"
            )

            # Show hottest tokens
            hot_tokens = sorted(
                self.tokens.values(),
                key=lambda t: t.hotness_score(),
                reverse=True
            )[:5]

            if hot_tokens:
                print("  TOP 5 HOT TOKENS:")
                for t in hot_tokens:
                    score = t.hotness_score()
                    if score > 0:
                        in_pos = "[HOLDING]" if t.mint in self.engine.positions else ""
                        print(f"    {t.symbol:8} | Score: {score:5.1f} | Vol/1m: {t.volume_1m():6.2f} SOL | Buy: {t.buy_ratio_1m():3.0%} | Traders: {len(t.unique_traders):3} {in_pos}")

    async def _cleanup_loop(self):
        """Periodically clean up old token data"""
        while self.running:
            await asyncio.sleep(60)  # Every minute

            if not self.running:
                break

            # Prune old trades from tokens
            for token in self.tokens.values():
                token.prune_old_trades(max_age=600)  # Keep 10 min of trades

            # Remove completely inactive tokens
            cutoff = time.time() - 600  # 10 min
            to_remove = [
                mint for mint, token in self.tokens.items()
                if token.last_trade < cutoff and mint not in self.engine.positions
            ]
            for mint in to_remove:
                del self.tokens[mint]

            if to_remove:
                logger.debug(f"Cleaned up {len(to_remove)} inactive tokens")

    def _print_final_report(self):
        """Print final session report"""
        print(f"""
+====================================================================+
|                      SESSION COMPLETE                              |
+====================================================================+
|  Runtime:          {self.stats.uptime_str:>10}                                 |
|  ----------------------------------------------------------------  |
|  Tokens Seen:        {self.stats.tokens_seen:>6}                                     |
|  Tokens Hot:         {self.stats.tokens_hot:>6}                                     |
|  Trades Received:    {self.stats.trades_received:>6}                                     |
|  ----------------------------------------------------------------  |
|  Trades Entered:     {self.stats.trades_entered:>6}                                     |
|  Trades Exited:      {self.stats.trades_exited:>6}                                     |
|  Wins:               {self.stats.trades_won:>6}                                     |
|  Losses:             {self.stats.trades_lost:>6}                                     |
|  Win Rate:           {self.stats.win_rate:>6.1%}                                     |
|  ----------------------------------------------------------------  |
|  Total PnL:          {self.stats.total_pnl_sol:>+10.4f} SOL                           |
|  Total Volume:       {self.stats.total_volume:>10.4f} SOL                           |
+====================================================================+
""")


async def run_active_trader(args):
    """Run the active token trader"""
    trader = ActiveTokenTrader(
        paper_mode=(args.mode == 'paper'),
        capital=args.capital,
        keypair_path=args.keypair,
        rpc_url=args.rpc,
    )

    try:
        await trader.start()
    except KeyboardInterrupt:
        await trader.stop()


def main():
    parser = argparse.ArgumentParser(description='Active Token Scanner & Trader')
    parser.add_argument('--mode', choices=['paper', 'real'], default='paper')
    parser.add_argument('--capital', type=float, default=100.0)
    parser.add_argument('--keypair', type=str, default=None)
    parser.add_argument('--rpc', type=str, default='https://api.mainnet-beta.solana.com')

    args = parser.parse_args()

    if args.mode == 'real' and not args.keypair:
        parser.error("--keypair required for real mode")

    asyncio.run(run_active_trader(args))


if __name__ == '__main__':
    main()
