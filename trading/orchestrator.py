"""
Live Trading Orchestrator
=========================

Connects all components for real trading:
1. LiveStream (data) → Token discovery + price updates
2. TradingEngine (logic) → Entry/exit decisions
3. RealExecutor (execution) → Solana transactions

This is the MAIN entry point for live trading.

Usage:
    python -m trading.orchestrator --mode paper --capital 100
    python -m trading.orchestrator --mode real --capital 100 --keypair ~/.config/solana/trading.json

RenTech Principle: Paper and Real modes use IDENTICAL logic.
The only difference is the executor.
"""

import asyncio
import argparse
import json
import logging
import signal
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp

from .engine import TradingEngine
from .config import TradingConfig, DEFAULT_CONFIG, AGGRESSIVE_CONFIG
from .models import Order, OrderSide, TokenData, Position
from .executors.paper import PaperExecutor
from .executors.real import RealExecutor

# Configure logging with colors for terminal
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors"""
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[37m',      # White
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m',
        'GREEN': '\033[32m',
        'RED': '\033[31m',
        'YELLOW': '\033[33m',
        'BLUE': '\033[34m',
        'CYAN': '\033[36m',
    }

    def format(self, record):
        # Add color based on level
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        record.msg = f"{color}{record.msg}{reset}"
        return super().format(record)


# Setup logging
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

PUMPPORTAL_WS = "wss://pumpportal.fun/api/data"
PUMP_API_BASE = "https://frontend-api.pump.fun"

# Reconnection settings
RECONNECT_DELAY = 5
MAX_RECONNECT_DELAY = 60

# Trading settings
TOKEN_EVAL_DELAY = 2.0  # Seconds to wait before evaluating new token
POSITION_CHECK_INTERVAL = 1.0  # Seconds between exit checks


# ============================================================
# HELPERS
# ============================================================

def format_sol(amount: float) -> str:
    """Format SOL amount"""
    if amount >= 1000:
        return f"{amount/1000:.2f}K"
    elif amount >= 1:
        return f"{amount:.2f}"
    else:
        return f"{amount:.4f}"


def format_mcap(mcap_sol: float, sol_price_usd: float = 230.0) -> str:
    """Format market cap in USD"""
    mcap_usd = mcap_sol * sol_price_usd
    if mcap_usd >= 1_000_000:
        return f"${mcap_usd/1_000_000:.2f}M"
    elif mcap_usd >= 1_000:
        return f"${mcap_usd/1_000:.1f}K"
    else:
        return f"${mcap_usd:.0f}"


def format_time_ago(timestamp: float) -> str:
    """Format time ago"""
    diff = time.time() - timestamp
    if diff < 60:
        return f"{int(diff)}s ago"
    elif diff < 3600:
        return f"{int(diff/60)}m ago"
    else:
        return f"{diff/3600:.1f}h ago"


@dataclass
class TokenMetrics:
    """Rich token metrics from pump.fun data"""
    mint: str
    symbol: str
    name: str

    # Price & Market
    price_sol: float = 0.0
    market_cap_sol: float = 0.0
    liquidity_sol: float = 0.0

    # Volume
    volume_sol_1m: float = 0.0
    volume_sol_5m: float = 0.0
    buy_volume_sol: float = 0.0
    sell_volume_sol: float = 0.0

    # Trades
    total_trades: int = 0
    buys: int = 0
    sells: int = 0
    unique_traders: int = 0

    # Holders (estimated from trades)
    estimated_holders: int = 0
    top_holder_pct: float = 0.0

    # Bonding curve
    bonding_progress: float = 0.0
    virtual_sol_reserves: float = 0.0
    virtual_token_reserves: float = 0.0

    # Creator
    creator: str = ""
    created_at: float = 0.0

    def to_log_string(self) -> str:
        """Format for logging"""
        return (
            f"{self.symbol} | "
            f"MCap: {format_mcap(self.market_cap_sol)} | "
            f"Liq: {format_sol(self.liquidity_sol)} SOL | "
            f"Vol: {format_sol(self.volume_sol_5m)} SOL/5m | "
            f"Trades: {self.total_trades} ({self.buys}B/{self.sells}S) | "
            f"Traders: {self.unique_traders} | "
            f"Bond: {self.bonding_progress:.1%}"
        )


@dataclass
class OrchestratorStats:
    """Track orchestrator performance"""
    start_time: float = field(default_factory=time.time)
    tokens_discovered: int = 0
    tokens_evaluated: int = 0
    tokens_skipped: int = 0
    trades_entered: int = 0
    trades_exited: int = 0
    trades_won: int = 0
    trades_lost: int = 0
    total_pnl_sol: float = 0.0
    total_volume_traded: float = 0.0
    reconnections: int = 0
    errors: int = 0

    # Best/worst trades
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0
    best_trade_token: str = ""
    worst_trade_token: str = ""

    @property
    def uptime_hours(self) -> float:
        return (time.time() - self.start_time) / 3600

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

    def to_dict(self) -> dict:
        return {
            'uptime': self.uptime_str,
            'tokens_discovered': self.tokens_discovered,
            'tokens_evaluated': self.tokens_evaluated,
            'trades_entered': self.trades_entered,
            'trades_exited': self.trades_exited,
            'win_rate': f"{self.win_rate:.1%}",
            'total_pnl_sol': f"{self.total_pnl_sol:+.4f}",
            'volume_traded': f"{self.total_volume_traded:.2f}",
        }


@dataclass
class TradeLog:
    """Detailed trade log entry"""
    timestamp: float
    action: str  # 'BUY' or 'SELL'
    mint: str
    symbol: str

    # Execution
    size_sol: float
    tokens: float
    price: float
    slippage: float

    # Token metrics at time of trade
    market_cap_sol: float
    liquidity_sol: float
    volume_5m: float
    trade_count: int
    unique_traders: int
    bonding_progress: float

    # For exits
    pnl_sol: float = 0.0
    pnl_pct: float = 0.0
    hold_time_secs: float = 0.0
    exit_reason: str = ""

    def to_log_string(self) -> str:
        """Format for console output"""
        time_str = datetime.fromtimestamp(self.timestamp).strftime('%H:%M:%S')

        if self.action == 'BUY':
            return (
                f"\n{'='*70}\n"
                f"  BUY | {time_str} | {self.symbol}\n"
                f"{'='*70}\n"
                f"  Mint:      {self.mint[:20]}...{self.mint[-8:]}\n"
                f"  Size:      {self.size_sol:.4f} SOL\n"
                f"  Price:     {self.price:.10f} SOL/token\n"
                f"  Tokens:    {self.tokens:,.0f}\n"
                f"  Slippage:  {self.slippage:.2%}\n"
                f"  -------------------------------------\n"
                f"  MCap:      {format_mcap(self.market_cap_sol)}\n"
                f"  Liquidity: {format_sol(self.liquidity_sol)} SOL\n"
                f"  Volume:    {format_sol(self.volume_5m)} SOL (5m)\n"
                f"  Trades:    {self.trade_count}\n"
                f"  Traders:   {self.unique_traders}\n"
                f"  Bonding:   {self.bonding_progress:.1%}\n"
                f"{'='*70}"
            )
        else:
            pnl_color = '\033[32m' if self.pnl_sol >= 0 else '\033[31m'
            reset = '\033[0m'
            return (
                f"\n{'='*70}\n"
                f"  SELL | {time_str} | {self.symbol}\n"
                f"{'='*70}\n"
                f"  Mint:      {self.mint[:20]}...{self.mint[-8:]}\n"
                f"  Size:      {self.size_sol:.4f} SOL\n"
                f"  Price:     {self.price:.10f} SOL/token\n"
                f"  Slippage:  {self.slippage:.2%}\n"
                f"  -------------------------------------\n"
                f"  {pnl_color}PnL:       {self.pnl_sol:+.4f} SOL ({self.pnl_pct:+.1%}){reset}\n"
                f"  Hold Time: {self.hold_time_secs:.1f}s\n"
                f"  Reason:    {self.exit_reason}\n"
                f"{'='*70}"
            )


class LiveTradingOrchestrator:
    """
    Main orchestrator for live trading.

    Connects:
    - PumpPortal WebSocket → New tokens + trades
    - TradingEngine → Decision logic (identical paper/real)
    - Executor → Paper or Real Solana execution

    The orchestrator handles:
    - Token discovery and evaluation
    - Position monitoring and exit checking
    - Error recovery and reconnection
    - Rich logging with full market data
    """

    def __init__(
        self,
        paper_mode: bool = True,
        capital: float = 100.0,
        keypair_path: Optional[str] = None,
        rpc_url: str = "https://api.mainnet-beta.solana.com",
        config: Optional[TradingConfig] = None,
        dry_run: bool = False,
    ):
        self.paper_mode = paper_mode
        self.capital = capital
        # Use AGGRESSIVE config for paper mode by default
        self.config = config or (AGGRESSIVE_CONFIG if paper_mode else DEFAULT_CONFIG)
        self.dry_run = dry_run

        # Initialize engine (SAME for paper and real)
        self.engine = TradingEngine(
            paper_mode=paper_mode,
            capital=capital,
            config=self.config,
        )

        # Initialize executor (DIFFERENT for paper vs real)
        if paper_mode:
            self.executor = PaperExecutor()
        else:
            if not keypair_path:
                raise ValueError("keypair_path required for real trading")
            self.executor = RealExecutor(
                keypair_path=keypair_path,
                rpc_url=rpc_url,
                dry_run=dry_run,
            )

        # Replace engine's executor with ours
        self.engine.executor = self.executor

        # State
        self.running = False
        self.stats = OrchestratorStats()
        self.token_cache: Dict[str, Dict[str, Any]] = {}
        self.pending_evaluations: Dict[str, float] = {}
        self.trade_logs: List[TradeLog] = []
        self.trader_sets: Dict[str, set] = {}  # mint -> set of trader addresses

        # WebSocket subscription management
        self.ws = None  # Current WebSocket connection
        self.subscribed_mints: set = set()  # Tokens we're subscribed to
        self.pending_subscribe: List[str] = []  # Tokens waiting to subscribe

    def _compute_token_metrics(self, mint: str) -> Optional[TokenMetrics]:
        """Compute rich metrics for a token"""
        if mint not in self.token_cache:
            return None

        data = self.token_cache[mint]
        trades = data.get('trades', [])

        # Compute metrics from trades
        now = time.time()
        volume_1m = sum(t['sol_amount'] for t in trades if now - t['timestamp'] < 60)
        volume_5m = sum(t['sol_amount'] for t in trades if now - t['timestamp'] < 300)
        buy_volume = sum(t['sol_amount'] for t in trades if t['is_buy'])
        sell_volume = sum(t['sol_amount'] for t in trades if not t['is_buy'])
        buys = sum(1 for t in trades if t['is_buy'])
        sells = len(trades) - buys

        # Unique traders
        traders = self.trader_sets.get(mint, set())

        # Estimate market cap from liquidity
        liquidity = data.get('liquidity_sol', 0) or data.get('vSolInBondingCurve', 0)
        bonding_progress = min(liquidity / 85.0, 1.0) if liquidity else 0

        # Market cap estimate (bonding curve math)
        mcap_sol = liquidity * 2 if liquidity else 0

        return TokenMetrics(
            mint=mint,
            symbol=data.get('symbol', '???'),
            name=data.get('name', 'Unknown'),
            price_sol=data.get('current_price', 0),
            market_cap_sol=mcap_sol,
            liquidity_sol=liquidity,
            volume_sol_1m=volume_1m,
            volume_sol_5m=volume_5m,
            buy_volume_sol=buy_volume,
            sell_volume_sol=sell_volume,
            total_trades=len(trades),
            buys=buys,
            sells=sells,
            unique_traders=len(traders),
            bonding_progress=bonding_progress,
            virtual_sol_reserves=data.get('vSolInBondingCurve', 0),
            virtual_token_reserves=data.get('vTokensInBondingCurve', 0),
            creator=data.get('creator', ''),
            created_at=data.get('discovered_at', 0),
        )

    async def start(self):
        """Start the orchestrator"""
        self.running = True
        self.stats.start_time = time.time()

        # Print startup banner
        mode_str = 'PAPER' if self.paper_mode else ('DRY RUN' if self.dry_run else 'REAL')
        mode_color = '\033[33m' if self.paper_mode else ('\033[36m' if self.dry_run else '\033[31m')
        reset = '\033[0m'

        print(f"""
{mode_color}+====================================================================+
|                    PUMP.FUN TRADING BOT                           |
+====================================================================+
|  Mode:     {mode_str:12}                                         |
|  Capital:  {self.capital:>8.2f} SOL                                       |
|  Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):19}                            |
+====================================================================+{reset}
""")

        # Connect executor if real
        if not self.paper_mode:
            if not await self.executor.connect():
                logger.error("Failed to connect executor")
                return

        # Handle shutdown signals
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))
            except NotImplementedError:
                pass  # Windows

        # Run main loops
        await asyncio.gather(
            self._websocket_loop(),
            self._evaluation_loop(),
            self._position_monitor_loop(),
            self._status_loop(),
            return_exceptions=True,
        )

    async def stop(self):
        """Stop the orchestrator gracefully"""
        print("\n")
        logger.info("Shutting down...")
        self.running = False

        # Close executor
        if hasattr(self.executor, 'close'):
            await self.executor.close()

        # Print final stats
        self._print_final_report()

    def _print_final_report(self):
        """Print detailed final report"""
        pnl_color = '\033[32m' if self.stats.total_pnl_sol >= 0 else '\033[31m'
        reset = '\033[0m'

        print(f"""
+====================================================================+
|                      SESSION COMPLETE                              |
+====================================================================+
|  Runtime:          {self.stats.uptime_str:>10}                                 |
|  ----------------------------------------------------------------  |
|  Tokens Discovered:  {self.stats.tokens_discovered:>6}                                     |
|  Tokens Evaluated:   {self.stats.tokens_evaluated:>6}                                     |
|  ----------------------------------------------------------------  |
|  Trades Entered:     {self.stats.trades_entered:>6}                                     |
|  Trades Exited:      {self.stats.trades_exited:>6}                                     |
|  Wins:               {self.stats.trades_won:>6}                                     |
|  Losses:             {self.stats.trades_lost:>6}                                     |
|  Win Rate:           {self.stats.win_rate:>6.1%}                                     |
|  ----------------------------------------------------------------  |
|  {pnl_color}Total PnL:           {self.stats.total_pnl_sol:>+10.4f} SOL{reset}                           |
|  Volume Traded:      {self.stats.total_volume_traded:>10.4f} SOL                           |
|  ----------------------------------------------------------------  |
|  Best Trade:         {self.stats.best_trade_pnl:>+10.4f} SOL ({self.stats.best_trade_token:>6})           |
|  Worst Trade:        {self.stats.worst_trade_pnl:>+10.4f} SOL ({self.stats.worst_trade_token:>6})           |
+====================================================================+
""")

    async def _websocket_loop(self):
        """Main WebSocket connection loop with reconnection"""
        reconnect_delay = RECONNECT_DELAY

        while self.running:
            try:
                logger.info("Connecting to PumpPortal WebSocket...")

                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(
                        PUMPPORTAL_WS,
                        timeout=aiohttp.ClientTimeout(total=30),
                        heartbeat=30,
                    ) as ws:
                        # Store WebSocket reference for subscriptions
                        self.ws = ws
                        self.subscribed_mints.clear()  # Clear on reconnect

                        # Subscribe to new tokens
                        await ws.send_json({"method": "subscribeNewToken"})
                        logger.info("Connected! Listening for new tokens...")

                        reconnect_delay = RECONNECT_DELAY

                        async for msg in ws:
                            if not self.running:
                                break

                            if msg.type == aiohttp.WSMsgType.TEXT:
                                await self._handle_message(msg.data)

                                # Process any pending subscriptions
                                if self.pending_subscribe:
                                    await self._subscribe_pending_tokens()

                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                logger.error(f"WebSocket error: {ws.exception()}")
                                break

                        self.ws = None  # Clear on disconnect

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.stats.errors += 1
                self.stats.reconnections += 1
                self.ws = None

            if self.running:
                logger.info(f"Reconnecting in {reconnect_delay}s...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, MAX_RECONNECT_DELAY)

    async def _subscribe_pending_tokens(self):
        """Subscribe to trades for pending tokens"""
        if not self.ws or not self.pending_subscribe:
            return

        # Get tokens that need subscription (max 20 at a time)
        to_subscribe = []
        while self.pending_subscribe and len(to_subscribe) < 20:
            mint = self.pending_subscribe.pop(0)
            if mint not in self.subscribed_mints:
                to_subscribe.append(mint)

        if to_subscribe:
            try:
                await self.ws.send_json({
                    "method": "subscribeTokenTrade",
                    "keys": to_subscribe
                })
                self.subscribed_mints.update(to_subscribe)
                logger.debug(f"Subscribed to {len(to_subscribe)} tokens (total: {len(self.subscribed_mints)})")
            except Exception as e:
                logger.error(f"Subscribe error: {e}")
                # Put back failed subscriptions
                self.pending_subscribe.extend(to_subscribe)

    async def _handle_message(self, data: str):
        """Handle incoming WebSocket message"""
        try:
            msg = json.loads(data)
            tx_type = msg.get('txType', '')

            # New token creation (txType == 'create')
            if tx_type == 'create' and 'mint' in msg:
                await self._on_new_token(msg)

            # Trade event (buy or sell)
            elif tx_type in ('buy', 'sell') and 'mint' in msg:
                await self._on_trade(msg)

        except Exception as e:
            logger.debug(f"Message parse error: {e}")

    async def _on_new_token(self, msg: dict):
        """Handle new token discovery"""
        mint = msg.get('mint', '')
        if not mint:
            return

        self.stats.tokens_discovered += 1

        symbol = msg.get('symbol', '???')
        name = msg.get('name', 'Unknown')

        # Cache token data with all available fields
        self.token_cache[mint] = {
            'mint': mint,
            'name': name,
            'symbol': symbol,
            'creator': msg.get('traderPublicKey', ''),
            'bonding_curve': msg.get('bondingCurveKey', ''),
            'discovered_at': time.time(),
            'trades': [],
            'current_price': 0.0,
            'market_cap_sol': msg.get('marketCapSol', 0),
            'liquidity_sol': msg.get('vSolInBondingCurve', 0),
            'vSolInBondingCurve': msg.get('vSolInBondingCurve', 0),
            'vTokensInBondingCurve': msg.get('vTokensInBondingCurve', 0),
            'uri': msg.get('uri', ''),
        }

        # Initialize trader set
        self.trader_sets[mint] = set()

        # Queue for subscription to get trades
        if mint not in self.subscribed_mints:
            self.pending_subscribe.append(mint)

        # Schedule evaluation after delay
        self.pending_evaluations[mint] = time.time() + TOKEN_EVAL_DELAY

        # Log new token discovery
        logger.info(f"NEW: {symbol} | {mint[:8]}...{mint[-4:]}")

    async def _on_trade(self, msg: dict):
        """Handle trade event for price updates"""
        mint = msg.get('mint', '')
        if not mint or mint not in self.token_cache:
            return

        # Update token data
        token = self.token_cache[mint]

        trade_data = {
            'is_buy': msg.get('txType', '').lower() == 'buy',
            'sol_amount': float(msg.get('solAmount', 0)) / 1e9,  # Convert from lamports
            'token_amount': int(msg.get('tokenAmount', 0)),
            'timestamp': time.time(),
            'trader': msg.get('traderPublicKey', ''),
        }
        token['trades'].append(trade_data)

        # Track unique traders
        if trade_data['trader']:
            self.trader_sets.setdefault(mint, set()).add(trade_data['trader'])

        # Update price from virtual reserves
        v_sol = float(msg.get('vSolInBondingCurve', 0))
        v_tokens = float(msg.get('vTokensInBondingCurve', 0))
        if v_sol > 0 and v_tokens > 0:
            token['current_price'] = v_sol / v_tokens
            token['vSolInBondingCurve'] = v_sol
            token['vTokensInBondingCurve'] = v_tokens
            token['liquidity_sol'] = v_sol
            token['market_cap_sol'] = msg.get('marketCapSol', v_sol * 2)

        # Update position PnL if we have one
        if mint in self.engine.positions:
            position = self.engine.positions[mint]
            if token['current_price'] > 0:
                current_value = position.tokens * token['current_price']
                position.current_pnl = (current_value - position.sol_invested) / position.sol_invested

                # Update peak PnL for trailing stop
                if position.current_pnl > position.peak_pnl:
                    position.peak_pnl = position.current_pnl

    async def _evaluation_loop(self):
        """Evaluate pending tokens for entry"""
        while self.running:
            try:
                now = time.time()
                to_evaluate = []

                for mint, eval_time in list(self.pending_evaluations.items()):
                    if now >= eval_time:
                        to_evaluate.append(mint)

                for mint in to_evaluate:
                    del self.pending_evaluations[mint]
                    await self._evaluate_token(mint)

            except Exception as e:
                logger.error(f"Evaluation loop error: {e}")
                self.stats.errors += 1

            await asyncio.sleep(0.1)

    async def _evaluate_token(self, mint: str):
        """Evaluate a token for entry"""
        if mint not in self.token_cache:
            return

        token_data = self.token_cache[mint]
        self.stats.tokens_evaluated += 1

        # Get rich metrics
        metrics = self._compute_token_metrics(mint)

        # ============================================================
        # PAPER MODE: Simple activity-based entry for testing
        # ============================================================
        if self.paper_mode and metrics:
            # Entry if: has some activity AND not at max positions
            has_activity = (
                metrics.total_trades >= 3 and  # At least 3 trades
                metrics.unique_traders >= 2 and  # At least 2 unique traders
                metrics.volume_sol_5m >= 0.1  # At least 0.1 SOL volume
            )

            can_enter = len(self.engine.positions) < self.config.max_open_positions
            current_price = token_data.get('current_price', 0)

            if has_activity and can_enter and current_price > 0 and mint not in self.engine.positions:
                # Calculate position size (10% of capital, max 2 SOL)
                position_size = min(self.capital * 0.10, 2.0, self.config.max_position_sol)

                logger.info(f"ENTRY: {token_data.get('symbol', '???')} | Trades: {metrics.total_trades} | Traders: {metrics.unique_traders} | Vol: {metrics.volume_sol_5m:.2f} SOL")

                # Execute directly via paper executor
                from .models import Order, OrderSide, OrderType
                order = Order(
                    mint=mint,
                    side=OrderSide.BUY,
                    amount_sol=position_size,
                    expected_price=current_price,
                    order_type=OrderType.MARKET,
                    context={'liquidity': metrics.liquidity_sol},
                )

                result = await self.executor.execute(order)

                if result.success:
                    # Create position
                    position = Position(
                        mint=mint,
                        entry_price=result.fill_price,
                        tokens=result.tokens,
                        sol_invested=position_size,
                        entry_time=time.time(),
                    )
                    self.engine.positions[mint] = position
                    self.engine.capital -= position_size
                    self.stats.trades_entered += 1
                    self.stats.total_volume_traded += position_size

                    # Log entry
                    trade_log = TradeLog(
                        timestamp=time.time(),
                        action='BUY',
                        mint=mint,
                        symbol=token_data.get('symbol', '???'),
                        size_sol=position_size,
                        tokens=result.tokens,
                        price=result.fill_price,
                        slippage=result.slippage,
                        market_cap_sol=metrics.market_cap_sol,
                        liquidity_sol=metrics.liquidity_sol,
                        volume_5m=metrics.volume_sol_5m,
                        trade_count=metrics.total_trades,
                        unique_traders=metrics.unique_traders,
                        bonding_progress=metrics.bonding_progress,
                    )
                    self.trade_logs.append(trade_log)
                    print(trade_log.to_log_string())
                    return

        # ============================================================
        # Standard engine evaluation (for real mode or if above skipped)
        # ============================================================
        try:
            # Build enriched token data for engine
            enriched_data = {
                **token_data,
                'price': token_data.get('current_price', 0),
                'liquidity_sol': metrics.liquidity_sol if metrics else 0,
                'bonding_curve_progress': metrics.bonding_progress if metrics else 0,
                'unique_traders': metrics.unique_traders if metrics else 0,
                'volume_5m': metrics.volume_sol_5m if metrics else 0,
            }

            decision = await self.engine.on_new_token(
                token_data=enriched_data,
                trades=token_data.get('trades', []),
            )

            if decision.action == 'BUY':
                self.stats.trades_entered += 1
                self.stats.total_volume_traded += decision.size or 0

                # Create detailed trade log
                trade_log = TradeLog(
                    timestamp=time.time(),
                    action='BUY',
                    mint=mint,
                    symbol=token_data.get('symbol', '???'),
                    size_sol=decision.size or 0,
                    tokens=decision.result.tokens if decision.result else 0,
                    price=decision.result.fill_price if decision.result else 0,
                    slippage=decision.result.slippage if decision.result else 0,
                    market_cap_sol=metrics.market_cap_sol if metrics else 0,
                    liquidity_sol=metrics.liquidity_sol if metrics else 0,
                    volume_5m=metrics.volume_sol_5m if metrics else 0,
                    trade_count=metrics.total_trades if metrics else 0,
                    unique_traders=metrics.unique_traders if metrics else 0,
                    bonding_progress=metrics.bonding_progress if metrics else 0,
                )
                self.trade_logs.append(trade_log)

                # Print detailed entry
                print(trade_log.to_log_string())

            else:
                self.stats.tokens_skipped += 1
                # Only log interesting skips
                if metrics and metrics.volume_sol_5m > 1:
                    logger.debug(f"SKIP: {metrics.to_log_string()} | Reason: {decision.reason}")

        except Exception as e:
            logger.error(f"Token evaluation error: {e}")
            self.stats.errors += 1

    async def _position_monitor_loop(self):
        """Monitor positions for exits"""
        while self.running:
            try:
                for mint, position in list(self.engine.positions.items()):
                    if mint not in self.token_cache:
                        continue

                    token = self.token_cache[mint]
                    current_price = token.get('current_price', 0)

                    if current_price <= 0:
                        continue

                    # Check exit via engine
                    exit_decision = await self.engine.on_price_update(
                        mint=mint,
                        price=current_price,
                        token_data=token,
                    )

                    if exit_decision and exit_decision.action == 'SELL':
                        self.stats.trades_exited += 1

                        # Calculate PnL
                        pnl_sol = exit_decision.result.sol_amount - position.sol_invested if exit_decision.result else 0
                        pnl_pct = position.current_pnl
                        hold_time = time.time() - position.entry_time

                        self.stats.total_pnl_sol += pnl_sol
                        self.stats.total_volume_traded += exit_decision.result.sol_amount if exit_decision.result else 0

                        if pnl_sol > 0:
                            self.stats.trades_won += 1
                            if pnl_sol > self.stats.best_trade_pnl:
                                self.stats.best_trade_pnl = pnl_sol
                                self.stats.best_trade_token = token.get('symbol', '???')
                        else:
                            self.stats.trades_lost += 1
                            if pnl_sol < self.stats.worst_trade_pnl:
                                self.stats.worst_trade_pnl = pnl_sol
                                self.stats.worst_trade_token = token.get('symbol', '???')

                        # Get metrics at exit
                        metrics = self._compute_token_metrics(mint)

                        # Create exit log
                        trade_log = TradeLog(
                            timestamp=time.time(),
                            action='SELL',
                            mint=mint,
                            symbol=token.get('symbol', '???'),
                            size_sol=exit_decision.result.sol_amount if exit_decision.result else 0,
                            tokens=position.tokens,
                            price=current_price,
                            slippage=exit_decision.result.slippage if exit_decision.result else 0,
                            market_cap_sol=metrics.market_cap_sol if metrics else 0,
                            liquidity_sol=metrics.liquidity_sol if metrics else 0,
                            volume_5m=metrics.volume_sol_5m if metrics else 0,
                            trade_count=metrics.total_trades if metrics else 0,
                            unique_traders=metrics.unique_traders if metrics else 0,
                            bonding_progress=metrics.bonding_progress if metrics else 0,
                            pnl_sol=pnl_sol,
                            pnl_pct=pnl_pct,
                            hold_time_secs=hold_time,
                            exit_reason=exit_decision.reason or '',
                        )
                        self.trade_logs.append(trade_log)

                        # Print detailed exit
                        print(trade_log.to_log_string())

            except Exception as e:
                logger.error(f"Position monitor error: {e}")
                self.stats.errors += 1

            await asyncio.sleep(POSITION_CHECK_INTERVAL)

    async def _status_loop(self):
        """Print status periodically"""
        while self.running:
            await asyncio.sleep(30)  # Every 30 seconds

            if not self.running:
                break

            # Compact status line
            pnl_color = '\033[32m' if self.stats.total_pnl_sol >= 0 else '\033[31m'
            reset = '\033[0m'

            # Count total trades received
            total_trades = sum(len(t.get('trades', [])) for t in self.token_cache.values())

            status = (
                f"STATUS | "
                f"Up: {self.stats.uptime_str} | "
                f"Tokens: {self.stats.tokens_discovered} | "
                f"Subs: {len(self.subscribed_mints)} | "
                f"Trades Rx: {total_trades} | "
                f"In/Out: {self.stats.trades_entered}/{self.stats.trades_exited} | "
                f"W/L: {self.stats.trades_won}/{self.stats.trades_lost} | "
                f"{pnl_color}PnL: {self.stats.total_pnl_sol:+.4f} SOL{reset} | "
                f"Open: {len(self.engine.positions)} | "
                f"Cap: {self.engine.capital:.2f} SOL"
            )
            logger.info(status)

            # Show open positions
            for mint, pos in self.engine.positions.items():
                token = self.token_cache.get(mint, {})
                symbol = token.get('symbol', '???')
                pnl_color = '\033[32m' if pos.current_pnl >= 0 else '\033[31m'
                hold_time = time.time() - pos.entry_time
                print(f"  +- {symbol}: {pnl_color}{pos.current_pnl:+.1%}{reset} | Hold: {hold_time:.0f}s | Size: {pos.sol_invested:.4f} SOL")


async def run_orchestrator(args):
    """Run the orchestrator with command line args"""
    orchestrator = LiveTradingOrchestrator(
        paper_mode=(args.mode == 'paper'),
        capital=args.capital,
        keypair_path=args.keypair,
        rpc_url=args.rpc,
        dry_run=args.dry_run,
    )

    await orchestrator.start()


def main():
    parser = argparse.ArgumentParser(description='Live Trading Orchestrator')
    parser.add_argument(
        '--mode',
        choices=['paper', 'real'],
        default='paper',
        help='Trading mode (default: paper)'
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=100.0,
        help='Starting capital in SOL (default: 100)'
    )
    parser.add_argument(
        '--keypair',
        type=str,
        default=None,
        help='Path to Solana keypair JSON (required for real mode)'
    )
    parser.add_argument(
        '--rpc',
        type=str,
        default='https://api.mainnet-beta.solana.com',
        help='Solana RPC URL'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Real mode but don\'t send transactions'
    )

    args = parser.parse_args()

    if args.mode == 'real' and not args.keypair:
        parser.error("--keypair is required for real mode")

    asyncio.run(run_orchestrator(args))


if __name__ == '__main__':
    main()
