#!/usr/bin/env python3
"""
RENTECH MONTE CARLO SIMULATION ENGINE
======================================

"In God we trust. All others must bring data." - W. Edwards Deming

This engine implements:
1. Historical trade replay against 1000+ signals
2. Monte Carlo simulations for position sizing
3. Walk-forward optimization to prevent overfitting
4. Signal performance ranking with statistical significance

Architecture:
- Stream trades from data lake by timestamp
- Compute all signals for each token in real-time
- Simulate entries/exits with realistic slippage
- Run 10,000+ Monte Carlo paths for each strategy
- Walk-forward: Train on N days, test on M days, roll forward

Target: Find the top 50 uncorrelated alpha signals
"""

import os
import sys
import json
import gzip
import time
import random
import sqlite3
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator, Set
from dataclasses import dataclass, field, asdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections import defaultdict
import math
import hashlib

# ============================================================
# CONFIGURATION
# ============================================================

# Simulation parameters
DEFAULT_INITIAL_CAPITAL = 10.0  # SOL
DEFAULT_MAX_POSITION_SOL = 0.5  # Max per trade
DEFAULT_SLIPPAGE_BPS = 50       # 0.5% slippage
DEFAULT_FEE_BPS = 30            # 0.3% fees (Pump.fun + priority)

# Monte Carlo settings
MC_SIMULATIONS = 10000
MC_CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]

# Walk-forward settings
TRAIN_DAYS = 14       # 2 weeks training
TEST_DAYS = 7         # 1 week testing
ROLL_DAYS = 3         # Roll forward by 3 days
MIN_TRADES_FOR_SIGNIFICANCE = 30

# Pump.fun specific
GRADUATION_SOL = 85.0  # ~85 SOL to graduate
BONDING_CURVE_K = 1_000_000_000  # y = x² / K


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Trade:
    """Single trade from data lake"""
    timestamp: int
    slot: int
    signature: str
    mint: str
    trader: str
    is_buy: bool
    sol_amount: int
    token_amount: int

    @property
    def sol(self) -> float:
        return self.sol_amount / 1e9


@dataclass
class TokenState:
    """Current state of a token during simulation"""
    mint: str
    first_seen: int
    trades: List[Trade] = field(default_factory=list)
    total_sol_volume: float = 0.0
    buy_count: int = 0
    sell_count: int = 0
    unique_traders: Set[str] = field(default_factory=set)
    bonding_curve_sol: float = 0.0  # Current position on curve
    graduated: bool = False

    def add_trade(self, trade: Trade):
        self.trades.append(trade)
        self.total_sol_volume += trade.sol
        self.unique_traders.add(trade.trader)

        if trade.is_buy:
            self.buy_count += 1
            self.bonding_curve_sol += trade.sol
        else:
            self.sell_count += 1
            self.bonding_curve_sol = max(0, self.bonding_curve_sol - trade.sol)

        if self.bonding_curve_sol >= GRADUATION_SOL:
            self.graduated = True

    def get_price(self) -> float:
        """Get current price from bonding curve: y = x² / K"""
        if self.bonding_curve_sol <= 0:
            return 0.0
        x = self.bonding_curve_sol
        return (2 * x) / BONDING_CURVE_K

    def age_seconds(self, current_time: int) -> float:
        return (current_time - self.first_seen) / 1000.0


@dataclass
class Position:
    """Open position during simulation"""
    mint: str
    entry_time: int
    entry_price: float
    entry_sol: float
    token_amount: float
    signal_id: int
    signal_value: float


@dataclass
class ClosedTrade:
    """Completed trade for analysis"""
    mint: str
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    entry_sol: float
    exit_sol: float
    pnl_sol: float
    pnl_pct: float
    hold_time_sec: float
    signal_id: int
    signal_value: float
    graduated: bool


@dataclass
class SignalResult:
    """Result from a signal computation"""
    value: float       # -1 to 1 (strength)
    confidence: float  # 0 to 1 (signal quality)


@dataclass
class SignalPerformance:
    """Performance metrics for a signal"""
    signal_id: int
    signal_name: str

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # PnL metrics
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    max_pnl: float = 0.0
    min_pnl: float = 0.0

    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0

    # Win rate
    win_rate: float = 0.0
    profit_factor: float = 0.0

    # Timing
    avg_hold_time: float = 0.0

    # Statistical significance
    t_statistic: float = 0.0
    p_value: float = 1.0
    is_significant: bool = False

    # Monte Carlo
    mc_var_90: float = 0.0
    mc_var_95: float = 0.0
    mc_var_99: float = 0.0
    mc_expected_return: float = 0.0


@dataclass
class WalkForwardResult:
    """Result from a single walk-forward fold"""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    in_sample_sharpe: float
    out_sample_sharpe: float
    out_sample_pnl: float
    out_sample_trades: int

    @property
    def degradation(self) -> float:
        """How much performance degraded out-of-sample"""
        if self.in_sample_sharpe == 0:
            return 0.0
        return (self.in_sample_sharpe - self.out_sample_sharpe) / abs(self.in_sample_sharpe)


# ============================================================
# SIGNAL INTERFACE (imported from signal factory)
# ============================================================

class BaseSignal:
    """Base class for all signals"""
    signal_id: int = 0
    name: str = "BaseSignal"
    category: str = "base"

    def compute(self, token: TokenState, current_time: int) -> SignalResult:
        """Compute signal value for a token"""
        raise NotImplementedError


def load_all_signals() -> Dict[int, BaseSignal]:
    """Load all signals from signal factories"""
    signals = {}

    # Try to import signal factories
    try:
        from formulas.pumpfun.signal_factory import SIGNAL_REGISTRY, create_all_signals
        for sig in create_all_signals():
            signals[sig.signal_id] = sig
    except ImportError:
        pass

    try:
        from formulas.pumpfun.signal_factory_advanced import ADVANCED_SIGNAL_REGISTRY, create_all_advanced_signals
        for sig in create_all_advanced_signals():
            signals[sig.signal_id] = sig
    except ImportError:
        pass

    try:
        from formulas.pumpfun.signal_factory_extended import EXTENDED_SIGNAL_REGISTRY, create_all_extended_signals
        for sig in create_all_extended_signals():
            signals[sig.signal_id] = sig
    except ImportError:
        pass

    return signals


# ============================================================
# TRADE REPLAY ENGINE
# ============================================================

class TradeReplayEngine:
    """
    Replays historical trades from data lake in chronological order.
    Maintains token state for signal computation.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.trade_files = []
        self.current_file_idx = 0
        self.current_file_trades = []
        self.current_trade_idx = 0
        self.token_states: Dict[str, TokenState] = {}

    def initialize(self, start_date: datetime, end_date: datetime):
        """Find all trade files in date range"""
        self.trade_files = []

        # Check multiple possible data directories
        data_dirs = [
            self.data_dir / "lake" / "raw",
            self.data_dir / "pumpfun_live",
            self.data_dir / "complete_trades",
            self.data_dir / "pumpfun_2025",
        ]

        for data_path in data_dirs:
            if not data_path.exists():
                continue

            for f in sorted(data_path.glob("*.jsonl.gz")):
                # Extract date from filename
                try:
                    date_str = f.stem.split("_")[-1].replace(".jsonl", "")
                    file_date = datetime.strptime(date_str, "%Y-%m-%d")
                    if start_date <= file_date.replace(tzinfo=timezone.utc) <= end_date:
                        self.trade_files.append(f)
                except:
                    continue

        self.trade_files = sorted(self.trade_files)
        print(f"Found {len(self.trade_files)} trade files in date range")

    def stream_trades(self) -> Generator[Tuple[Trade, TokenState], None, None]:
        """Stream trades and their token states in chronological order"""

        for file_path in self.trade_files:
            print(f"  Processing {file_path.name}...")

            try:
                with gzip.open(file_path, 'rt') as f:
                    trades = []
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            trade = Trade(
                                timestamp=data.get('timestamp', 0),
                                slot=data.get('slot', 0),
                                signature=data.get('signature', ''),
                                mint=data.get('mint', ''),
                                trader=data.get('trader', ''),
                                is_buy=data.get('is_buy', True),
                                sol_amount=data.get('sol_amount', 0),
                                token_amount=data.get('token_amount', 0)
                            )
                            trades.append(trade)
                        except:
                            continue

                    # Sort by timestamp
                    trades.sort(key=lambda t: t.timestamp)

                    for trade in trades:
                        # Update token state
                        if trade.mint not in self.token_states:
                            self.token_states[trade.mint] = TokenState(
                                mint=trade.mint,
                                first_seen=trade.timestamp
                            )

                        token_state = self.token_states[trade.mint]
                        token_state.add_trade(trade)

                        yield trade, token_state

            except Exception as e:
                print(f"  Error processing {file_path}: {e}")
                continue

    def get_token_state(self, mint: str) -> Optional[TokenState]:
        """Get current state for a token"""
        return self.token_states.get(mint)


# ============================================================
# MONTE CARLO SIMULATOR
# ============================================================

class MonteCarloSimulator:
    """
    Monte Carlo simulation for portfolio returns.

    Methods:
    1. Bootstrap resampling - Sample from historical returns with replacement
    2. Parametric - Fit distribution and simulate
    3. Block bootstrap - Preserve autocorrelation
    """

    def __init__(self, n_simulations: int = MC_SIMULATIONS):
        self.n_simulations = n_simulations

    def bootstrap_returns(
        self,
        returns: np.ndarray,
        n_periods: int
    ) -> np.ndarray:
        """
        Bootstrap simulation by resampling historical returns.

        Args:
            returns: Historical returns array
            n_periods: Number of periods to simulate

        Returns:
            Array of shape (n_simulations, n_periods) with simulated returns
        """
        if len(returns) == 0:
            return np.zeros((self.n_simulations, n_periods))

        simulated = np.zeros((self.n_simulations, n_periods))

        for i in range(self.n_simulations):
            # Random sample with replacement
            indices = np.random.randint(0, len(returns), size=n_periods)
            simulated[i] = returns[indices]

        return simulated

    def parametric_simulation(
        self,
        returns: np.ndarray,
        n_periods: int,
        distribution: str = 'normal'
    ) -> np.ndarray:
        """
        Parametric simulation fitting distribution to returns.

        Supports: 'normal', 'student_t', 'laplace'
        """
        if len(returns) == 0:
            return np.zeros((self.n_simulations, n_periods))

        mean = np.mean(returns)
        std = np.std(returns)

        if distribution == 'normal':
            simulated = np.random.normal(mean, std, (self.n_simulations, n_periods))
        elif distribution == 'student_t':
            # Fit degrees of freedom (typically 3-5 for financial returns)
            df = 4
            simulated = mean + std * np.random.standard_t(df, (self.n_simulations, n_periods))
        elif distribution == 'laplace':
            scale = std / np.sqrt(2)
            simulated = np.random.laplace(mean, scale, (self.n_simulations, n_periods))
        else:
            simulated = np.random.normal(mean, std, (self.n_simulations, n_periods))

        return simulated

    def calculate_var(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """Calculate Value at Risk at given confidence level"""
        if len(returns) == 0:
            return 0.0
        return -np.percentile(returns, (1 - confidence) * 100)

    def calculate_cvar(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """Calculate Conditional VaR (Expected Shortfall)"""
        if len(returns) == 0:
            return 0.0
        var = self.calculate_var(returns, confidence)
        return -np.mean(returns[returns <= -var])

    def run_portfolio_simulation(
        self,
        closed_trades: List[ClosedTrade],
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
        n_periods: int = 252  # ~1 year of trading days
    ) -> Dict:
        """
        Run full Monte Carlo simulation on portfolio.

        Returns:
            Dictionary with simulation results and statistics
        """
        # Extract returns
        returns = np.array([t.pnl_pct for t in closed_trades if t.pnl_pct != 0])

        if len(returns) < MIN_TRADES_FOR_SIGNIFICANCE:
            return {
                'n_trades': len(returns),
                'insufficient_data': True
            }

        # Run simulations
        bootstrap_paths = self.bootstrap_returns(returns, n_periods)
        parametric_paths = self.parametric_simulation(returns, n_periods, 'student_t')

        # Calculate final portfolio values
        bootstrap_final = initial_capital * np.prod(1 + bootstrap_paths, axis=1)
        parametric_final = initial_capital * np.prod(1 + parametric_paths, axis=1)

        # Calculate statistics
        results = {
            'n_trades': len(returns),
            'n_simulations': self.n_simulations,
            'n_periods': n_periods,

            # Bootstrap results
            'bootstrap': {
                'mean_final_value': float(np.mean(bootstrap_final)),
                'median_final_value': float(np.median(bootstrap_final)),
                'std_final_value': float(np.std(bootstrap_final)),
                'var_90': float(self.calculate_var(bootstrap_paths.flatten(), 0.90)),
                'var_95': float(self.calculate_var(bootstrap_paths.flatten(), 0.95)),
                'var_99': float(self.calculate_var(bootstrap_paths.flatten(), 0.99)),
                'cvar_95': float(self.calculate_cvar(bootstrap_paths.flatten(), 0.95)),
                'prob_profit': float(np.mean(bootstrap_final > initial_capital)),
                'prob_double': float(np.mean(bootstrap_final > 2 * initial_capital)),
                'prob_ruin': float(np.mean(bootstrap_final < 0.1 * initial_capital)),
            },

            # Parametric results
            'parametric': {
                'mean_final_value': float(np.mean(parametric_final)),
                'median_final_value': float(np.median(parametric_final)),
                'var_95': float(self.calculate_var(parametric_paths.flatten(), 0.95)),
                'prob_profit': float(np.mean(parametric_final > initial_capital)),
            },

            # Percentiles
            'percentiles': {
                '1': float(np.percentile(bootstrap_final, 1)),
                '5': float(np.percentile(bootstrap_final, 5)),
                '10': float(np.percentile(bootstrap_final, 10)),
                '25': float(np.percentile(bootstrap_final, 25)),
                '50': float(np.percentile(bootstrap_final, 50)),
                '75': float(np.percentile(bootstrap_final, 75)),
                '90': float(np.percentile(bootstrap_final, 90)),
                '95': float(np.percentile(bootstrap_final, 95)),
                '99': float(np.percentile(bootstrap_final, 99)),
            }
        }

        return results


# ============================================================
# WALK-FORWARD OPTIMIZER
# ============================================================

class WalkForwardOptimizer:
    """
    Walk-forward analysis to prevent overfitting.

    Process:
    1. Divide data into sequential folds
    2. For each fold: Train on N days, test on M days
    3. Roll forward and repeat
    4. Aggregate out-of-sample performance
    """

    def __init__(
        self,
        train_days: int = TRAIN_DAYS,
        test_days: int = TEST_DAYS,
        roll_days: int = ROLL_DAYS
    ):
        self.train_days = train_days
        self.test_days = test_days
        self.roll_days = roll_days

    def generate_folds(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        Generate walk-forward folds.

        Returns:
            List of (train_start, train_end, test_start, test_end) tuples
        """
        folds = []
        current_train_start = start_date

        while True:
            train_end = current_train_start + timedelta(days=self.train_days)
            test_start = train_end
            test_end = test_start + timedelta(days=self.test_days)

            if test_end > end_date:
                break

            folds.append((current_train_start, train_end, test_start, test_end))
            current_train_start += timedelta(days=self.roll_days)

        return folds

    def calculate_sharpe(self, returns: List[float]) -> float:
        """Calculate annualized Sharpe ratio"""
        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        # Assume 252 trading days per year
        daily_sharpe = mean_return / std_return
        annualized_sharpe = daily_sharpe * np.sqrt(252)

        return float(annualized_sharpe)

    def run_fold(
        self,
        trades: List[ClosedTrade],
        fold: Tuple[datetime, datetime, datetime, datetime]
    ) -> WalkForwardResult:
        """Run a single walk-forward fold"""
        train_start, train_end, test_start, test_end = fold

        # Convert to timestamps
        train_start_ts = int(train_start.timestamp() * 1000)
        train_end_ts = int(train_end.timestamp() * 1000)
        test_start_ts = int(test_start.timestamp() * 1000)
        test_end_ts = int(test_end.timestamp() * 1000)

        # Split trades
        train_trades = [t for t in trades
                       if train_start_ts <= t.entry_time < train_end_ts]
        test_trades = [t for t in trades
                      if test_start_ts <= t.entry_time < test_end_ts]

        # Calculate metrics
        train_returns = [t.pnl_pct for t in train_trades]
        test_returns = [t.pnl_pct for t in test_trades]

        in_sample_sharpe = self.calculate_sharpe(train_returns)
        out_sample_sharpe = self.calculate_sharpe(test_returns)
        out_sample_pnl = sum(t.pnl_sol for t in test_trades)

        return WalkForwardResult(
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            in_sample_sharpe=in_sample_sharpe,
            out_sample_sharpe=out_sample_sharpe,
            out_sample_pnl=out_sample_pnl,
            out_sample_trades=len(test_trades)
        )

    def analyze_signal(
        self,
        trades: List[ClosedTrade],
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """
        Run full walk-forward analysis for a signal.

        Returns comprehensive statistics on out-of-sample performance.
        """
        folds = self.generate_folds(start_date, end_date)

        if not folds:
            return {'error': 'Insufficient date range for walk-forward'}

        results = []
        for fold in folds:
            result = self.run_fold(trades, fold)
            results.append(result)

        # Aggregate statistics
        out_sample_sharpes = [r.out_sample_sharpe for r in results if r.out_sample_trades > 0]
        degradations = [r.degradation for r in results if r.out_sample_trades > 0]
        total_out_sample_pnl = sum(r.out_sample_pnl for r in results)
        total_out_sample_trades = sum(r.out_sample_trades for r in results)

        return {
            'n_folds': len(folds),
            'avg_out_sample_sharpe': float(np.mean(out_sample_sharpes)) if out_sample_sharpes else 0.0,
            'std_out_sample_sharpe': float(np.std(out_sample_sharpes)) if out_sample_sharpes else 0.0,
            'avg_degradation': float(np.mean(degradations)) if degradations else 0.0,
            'total_out_sample_pnl': float(total_out_sample_pnl),
            'total_out_sample_trades': total_out_sample_trades,
            'fold_results': [asdict(r) for r in results],

            # Quality metrics
            'consistency': float(np.mean([1 if r.out_sample_sharpe > 0 else 0 for r in results])),
            'robustness_score': float(np.mean(out_sample_sharpes) / (np.std(out_sample_sharpes) + 0.001)) if out_sample_sharpes else 0.0
        }


# ============================================================
# SIGNAL BACKTESTER
# ============================================================

class SignalBacktester:
    """
    Backtest signals on historical data.

    Entry rules:
    - Signal value > threshold
    - Signal confidence > min_confidence
    - Token age within range
    - Not already in position

    Exit rules:
    - Take profit at target %
    - Stop loss at threshold %
    - Max hold time
    - Token graduated
    """

    def __init__(
        self,
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
        max_position_sol: float = DEFAULT_MAX_POSITION_SOL,
        slippage_bps: int = DEFAULT_SLIPPAGE_BPS,
        fee_bps: int = DEFAULT_FEE_BPS,
        entry_threshold: float = 0.3,
        min_confidence: float = 0.5,
        take_profit_pct: float = 0.5,   # 50% profit
        stop_loss_pct: float = 0.2,     # 20% loss
        max_hold_seconds: float = 300,  # 5 minutes
        min_token_age_sec: float = 10,
        max_token_age_sec: float = 600
    ):
        self.initial_capital = initial_capital
        self.max_position_sol = max_position_sol
        self.slippage_bps = slippage_bps
        self.fee_bps = fee_bps
        self.entry_threshold = entry_threshold
        self.min_confidence = min_confidence
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_hold_seconds = max_hold_seconds
        self.min_token_age_sec = min_token_age_sec
        self.max_token_age_sec = max_token_age_sec

        # State
        self.capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[ClosedTrade] = []

    def reset(self):
        """Reset backtest state"""
        self.capital = self.initial_capital
        self.positions = {}
        self.closed_trades = []

    def apply_slippage(self, price: float, is_buy: bool) -> float:
        """Apply slippage to price"""
        slippage = price * (self.slippage_bps / 10000)
        return price + slippage if is_buy else price - slippage

    def apply_fees(self, sol_amount: float) -> float:
        """Apply trading fees"""
        fee = sol_amount * (self.fee_bps / 10000)
        return sol_amount - fee

    def check_entry(
        self,
        token: TokenState,
        signal_result: SignalResult,
        signal_id: int,
        current_time: int
    ) -> bool:
        """Check if entry conditions are met"""
        # Already in position
        if token.mint in self.positions:
            return False

        # Signal thresholds
        if signal_result.value < self.entry_threshold:
            return False
        if signal_result.confidence < self.min_confidence:
            return False

        # Token age
        age = token.age_seconds(current_time)
        if age < self.min_token_age_sec or age > self.max_token_age_sec:
            return False

        # Already graduated
        if token.graduated:
            return False

        # Sufficient capital
        if self.capital < 0.01:
            return False

        return True

    def enter_position(
        self,
        token: TokenState,
        signal_result: SignalResult,
        signal_id: int,
        current_time: int
    ):
        """Enter a new position"""
        price = token.get_price()
        if price <= 0:
            return

        # Position sizing
        position_sol = min(self.max_position_sol, self.capital * 0.1)
        entry_price = self.apply_slippage(price, is_buy=True)
        entry_sol = self.apply_fees(position_sol)
        token_amount = entry_sol / entry_price

        # Create position
        position = Position(
            mint=token.mint,
            entry_time=current_time,
            entry_price=entry_price,
            entry_sol=entry_sol,
            token_amount=token_amount,
            signal_id=signal_id,
            signal_value=signal_result.value
        )

        self.positions[token.mint] = position
        self.capital -= position_sol

    def check_exit(
        self,
        position: Position,
        token: TokenState,
        current_time: int
    ) -> Tuple[bool, str]:
        """Check if exit conditions are met"""
        price = token.get_price()
        if price <= 0:
            return True, 'invalid_price'

        # Current value
        current_value = position.token_amount * price
        pnl_pct = (current_value - position.entry_sol) / position.entry_sol

        # Take profit
        if pnl_pct >= self.take_profit_pct:
            return True, 'take_profit'

        # Stop loss
        if pnl_pct <= -self.stop_loss_pct:
            return True, 'stop_loss'

        # Max hold time
        hold_time = (current_time - position.entry_time) / 1000.0
        if hold_time >= self.max_hold_seconds:
            return True, 'max_hold_time'

        # Token graduated (great exit!)
        if token.graduated:
            return True, 'graduated'

        return False, ''

    def exit_position(
        self,
        position: Position,
        token: TokenState,
        current_time: int,
        exit_reason: str
    ):
        """Exit a position"""
        price = token.get_price()
        if price <= 0:
            price = position.entry_price * 0.5  # Assume 50% loss if price invalid

        exit_price = self.apply_slippage(price, is_buy=False)
        gross_sol = position.token_amount * exit_price
        exit_sol = self.apply_fees(gross_sol)

        pnl_sol = exit_sol - position.entry_sol
        pnl_pct = pnl_sol / position.entry_sol
        hold_time = (current_time - position.entry_time) / 1000.0

        # Record closed trade
        closed = ClosedTrade(
            mint=position.mint,
            entry_time=position.entry_time,
            exit_time=current_time,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_sol=position.entry_sol,
            exit_sol=exit_sol,
            pnl_sol=pnl_sol,
            pnl_pct=pnl_pct,
            hold_time_sec=hold_time,
            signal_id=position.signal_id,
            signal_value=position.signal_value,
            graduated=token.graduated
        )

        self.closed_trades.append(closed)
        self.capital += exit_sol
        del self.positions[position.mint]

    def process_trade(
        self,
        trade: Trade,
        token: TokenState,
        signal: BaseSignal
    ):
        """Process a single trade event"""
        current_time = trade.timestamp

        # Check exits for existing positions
        if trade.mint in self.positions:
            position = self.positions[trade.mint]
            should_exit, reason = self.check_exit(position, token, current_time)
            if should_exit:
                self.exit_position(position, token, current_time, reason)

        # Check entry for this token
        if trade.mint not in self.positions:
            try:
                # Compute signal (simplified for backtest)
                signal_data = {
                    'trades': [asdict(t) for t in token.trades[-100:]],
                    'total_volume': token.total_sol_volume,
                    'unique_traders': len(token.unique_traders),
                    'buy_count': token.buy_count,
                    'sell_count': token.sell_count,
                    'token_age_sec': token.age_seconds(current_time),
                    'bonding_curve_sol': token.bonding_curve_sol,
                    'created_at': token.first_seen,
                }

                result = signal.compute(signal_data)

                if self.check_entry(token, result, signal.signal_id, current_time):
                    self.enter_position(token, result, signal.signal_id, current_time)

            except Exception as e:
                pass  # Signal computation failed

    def get_performance(self) -> SignalPerformance:
        """Calculate performance metrics"""
        if not self.closed_trades:
            return SignalPerformance(signal_id=0, signal_name="Unknown")

        # Basic stats
        total_trades = len(self.closed_trades)
        winning = [t for t in self.closed_trades if t.pnl_sol > 0]
        losing = [t for t in self.closed_trades if t.pnl_sol <= 0]

        # PnL
        pnls = [t.pnl_sol for t in self.closed_trades]
        total_pnl = sum(pnls)
        avg_pnl = total_pnl / total_trades

        # Returns for Sharpe
        returns = [t.pnl_pct for t in self.closed_trades]
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Sharpe (annualized assuming ~100 trades/day)
        sharpe = (mean_return / std_return * np.sqrt(252 * 100)) if std_return > 0 else 0

        # Sortino (downside deviation)
        negative_returns = [r for r in returns if r < 0]
        downside_std = np.std(negative_returns) if negative_returns else 0.001
        sortino = (mean_return / downside_std * np.sqrt(252 * 100)) if downside_std > 0 else 0

        # Max drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative + self.initial_capital)
        drawdown = (running_max - (cumulative + self.initial_capital)) / running_max
        max_drawdown = float(np.max(drawdown))

        # Calmar
        calmar = (total_pnl / self.initial_capital) / max_drawdown if max_drawdown > 0 else 0

        # Win rate and profit factor
        win_rate = len(winning) / total_trades if total_trades > 0 else 0
        gross_profit = sum(t.pnl_sol for t in winning)
        gross_loss = abs(sum(t.pnl_sol for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Avg hold time
        avg_hold = np.mean([t.hold_time_sec for t in self.closed_trades])

        # T-statistic
        if total_trades >= MIN_TRADES_FOR_SIGNIFICANCE and std_return > 0:
            t_stat = mean_return / (std_return / np.sqrt(total_trades))
            # Two-tailed p-value approximation
            p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / np.sqrt(2))))
        else:
            t_stat = 0
            p_value = 1.0

        return SignalPerformance(
            signal_id=self.closed_trades[0].signal_id if self.closed_trades else 0,
            signal_name="",
            total_trades=total_trades,
            winning_trades=len(winning),
            losing_trades=len(losing),
            total_pnl=float(total_pnl),
            avg_pnl=float(avg_pnl),
            max_pnl=float(max(pnls)),
            min_pnl=float(min(pnls)),
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            max_drawdown=float(max_drawdown),
            calmar_ratio=float(calmar),
            win_rate=float(win_rate),
            profit_factor=float(profit_factor),
            avg_hold_time=float(avg_hold),
            t_statistic=float(t_stat),
            p_value=float(p_value),
            is_significant=p_value < 0.05
        )


# ============================================================
# FULL SIMULATION ORCHESTRATOR
# ============================================================

class SimulationOrchestrator:
    """
    Orchestrates the full simulation:
    1. Load signals
    2. Replay trades
    3. Backtest each signal
    4. Run Monte Carlo
    5. Walk-forward analysis
    6. Rank signals
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.signals = load_all_signals()
        self.replay_engine = TradeReplayEngine(data_dir)
        self.monte_carlo = MonteCarloSimulator()
        self.walk_forward = WalkForwardOptimizer()
        self.results: Dict[int, Dict] = {}

    def run_full_simulation(
        self,
        start_date: datetime,
        end_date: datetime,
        signal_ids: Optional[List[int]] = None
    ) -> Dict:
        """
        Run complete simulation on all (or selected) signals.
        """
        print("=" * 70)
        print("  RENTECH MONTE CARLO SIMULATION ENGINE")
        print("=" * 70)
        print(f"  Date range: {start_date.date()} to {end_date.date()}")
        print(f"  Signals loaded: {len(self.signals)}")
        print()

        # Initialize replay
        self.replay_engine.initialize(start_date, end_date)

        # Select signals to test
        if signal_ids:
            test_signals = {sid: self.signals[sid] for sid in signal_ids if sid in self.signals}
        else:
            test_signals = self.signals

        print(f"  Testing {len(test_signals)} signals...")

        # Backtest each signal
        signal_results = {}

        for signal_id, signal in test_signals.items():
            print(f"\n  [{signal_id}] {signal.name}...")

            # Create fresh backtester
            backtester = SignalBacktester()

            # Replay trades
            trade_count = 0
            for trade, token_state in self.replay_engine.stream_trades():
                backtester.process_trade(trade, token_state, signal)
                trade_count += 1

                if trade_count % 100000 == 0:
                    print(f"    Processed {trade_count:,} trades, {len(backtester.closed_trades)} closed positions")

            # Get performance
            perf = backtester.get_performance()
            perf.signal_name = signal.name

            # Monte Carlo simulation
            mc_results = self.monte_carlo.run_portfolio_simulation(backtester.closed_trades)

            # Walk-forward analysis
            wf_results = self.walk_forward.analyze_signal(
                backtester.closed_trades,
                start_date,
                end_date
            )

            # Combine results
            signal_results[signal_id] = {
                'performance': asdict(perf),
                'monte_carlo': mc_results,
                'walk_forward': wf_results,
                'n_trades': len(backtester.closed_trades)
            }

            print(f"    Trades: {perf.total_trades}, Sharpe: {perf.sharpe_ratio:.2f}, "
                  f"Win Rate: {perf.win_rate:.1%}, PnL: {perf.total_pnl:.2f} SOL")

            # Reset replay engine for next signal
            self.replay_engine.token_states = {}

        self.results = signal_results
        return self.rank_signals()

    def rank_signals(self) -> Dict:
        """
        Rank signals by composite score.

        Composite score considers:
        - Sharpe ratio (30%)
        - Walk-forward consistency (25%)
        - Win rate (15%)
        - Profit factor (15%)
        - Statistical significance (15%)
        """
        rankings = []

        for signal_id, result in self.results.items():
            perf = result['performance']
            wf = result.get('walk_forward', {})

            # Normalize metrics (0-1 scale)
            sharpe_score = min(1.0, max(0.0, perf['sharpe_ratio'] / 3.0))  # Cap at 3.0
            consistency = wf.get('consistency', 0.0)
            win_rate = perf['win_rate']
            pf_score = min(1.0, max(0.0, (perf['profit_factor'] - 1.0) / 2.0))  # 1.0-3.0 -> 0-1
            significance = 1.0 if perf['is_significant'] else 0.0

            # Composite score
            composite = (
                0.30 * sharpe_score +
                0.25 * consistency +
                0.15 * win_rate +
                0.15 * pf_score +
                0.15 * significance
            )

            rankings.append({
                'signal_id': signal_id,
                'signal_name': perf['signal_name'],
                'composite_score': composite,
                'sharpe': perf['sharpe_ratio'],
                'win_rate': perf['win_rate'],
                'total_pnl': perf['total_pnl'],
                'n_trades': perf['total_trades'],
                'is_significant': perf['is_significant'],
                'wf_consistency': consistency,
                'wf_avg_sharpe': wf.get('avg_out_sample_sharpe', 0.0)
            })

        # Sort by composite score
        rankings.sort(key=lambda x: x['composite_score'], reverse=True)

        return {
            'rankings': rankings,
            'top_10': rankings[:10],
            'bottom_10': rankings[-10:],
            'significant_signals': [r for r in rankings if r['is_significant']],
            'summary': {
                'total_signals': len(rankings),
                'significant_count': sum(1 for r in rankings if r['is_significant']),
                'avg_sharpe': np.mean([r['sharpe'] for r in rankings]),
                'avg_win_rate': np.mean([r['win_rate'] for r in rankings])
            }
        }

    def save_results(self, output_path: Path):
        """Save results to JSON"""
        with open(output_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'results': self.results,
                'rankings': self.rank_signals()
            }, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    """Run simulation"""
    import argparse

    parser = argparse.ArgumentParser(description='RenTech Monte Carlo Simulation')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--start', type=str, default='2024-10-01', help='Start date')
    parser.add_argument('--end', type=str, default='2024-12-01', help='End date')
    parser.add_argument('--signals', type=str, default='', help='Comma-separated signal IDs to test')
    parser.add_argument('--output', type=str, default='simulation_results.json', help='Output file')

    args = parser.parse_args()

    # Parse dates
    start_date = datetime.strptime(args.start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(args.end, '%Y-%m-%d').replace(tzinfo=timezone.utc)

    # Parse signal IDs
    signal_ids = None
    if args.signals:
        signal_ids = [int(s) for s in args.signals.split(',')]

    # Run simulation
    orchestrator = SimulationOrchestrator(Path(args.data_dir))
    results = orchestrator.run_full_simulation(start_date, end_date, signal_ids)

    # Print top 10
    print("\n" + "=" * 70)
    print("  TOP 10 SIGNALS")
    print("=" * 70)

    for i, sig in enumerate(results['top_10'], 1):
        print(f"  {i}. [{sig['signal_id']}] {sig['signal_name']}")
        print(f"     Score: {sig['composite_score']:.3f} | Sharpe: {sig['sharpe']:.2f} | "
              f"Win: {sig['win_rate']:.1%} | PnL: {sig['total_pnl']:.2f} SOL")

    # Save results
    orchestrator.save_results(Path(args.output))


if __name__ == "__main__":
    main()
