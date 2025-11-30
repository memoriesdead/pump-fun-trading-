"""
ADAPTIVE VOLATILITY-SCALED TRADING SYSTEM
==========================================
The parameters SCALE with current market volatility.
Works at ANY timeframe because everything is relative to volatility.

This is the FUNDAMENTAL FIX for the infinite timeframe problem.
"""
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class AdaptiveParameters:
    """Trading parameters that scale with volatility"""
    take_profit: float      # As percentage (volatility-scaled)
    stop_loss: float        # As percentage (volatility-scaled)
    tp_price: float         # Actual TP price level
    sl_price: float         # Actual SL price level
    expected_hold_secs: float  # Expected time to reach TP
    current_volatility: float  # Volatility used for calculation
    vol_regime: str         # 'low', 'medium', 'high', 'extreme'


class AdaptiveVolatilityTrader:
    """
    Trading system where ALL parameters scale with real-time volatility.

    Key insight from Guillaume et al. (1997):
    - 1% move in 1 second = same significance as 1% move in 1 hour
    - Parameters should be volatility-relative, not time-fixed

    This means:
    - In HIGH volatility: TP/SL are larger (in price), hold time shorter
    - In LOW volatility: TP/SL are smaller (in price), hold time longer
    - The RELATIVE size (in volatility units) stays constant

    The edge comes from:
    - Signal predicts direction with >50% accuracy
    - TP > SL gives favorable risk/reward
    - Volatility scaling ensures parameters always match market conditions
    """

    def __init__(
        self,
        tp_vol_multiple: float = 2.0,   # TP at 2x current volatility
        sl_vol_multiple: float = 1.5,   # SL at 1.5x current volatility
        vol_lookback: int = 100,        # Samples for volatility estimation
        min_vol_pct: float = 0.0001,    # Minimum 0.01% volatility floor
    ):
        self.tp_vol_multiple = tp_vol_multiple
        self.sl_vol_multiple = sl_vol_multiple
        self.vol_lookback = vol_lookback
        self.min_vol_pct = min_vol_pct

        self.prices = deque(maxlen=vol_lookback)
        self.timestamps = deque(maxlen=vol_lookback)
        self.returns = deque(maxlen=vol_lookback)

    def update(self, price: float, timestamp: float) -> None:
        """Update price history"""
        if len(self.prices) > 0:
            ret = (price - self.prices[-1]) / self.prices[-1]
            self.returns.append(ret)
        self.prices.append(price)
        self.timestamps.append(timestamp)

    def get_current_volatility(self) -> Tuple[float, str]:
        """
        Calculate current realized volatility.
        Returns (volatility_pct, regime)
        """
        if len(self.returns) < 10:
            return self.min_vol_pct, 'unknown'

        returns = np.array(self.returns)
        vol = np.std(returns)
        vol = max(vol, self.min_vol_pct)

        # Determine regime
        if vol < 0.0002:      # < 0.02%
            regime = 'low'
        elif vol < 0.0005:    # < 0.05%
            regime = 'medium'
        elif vol < 0.001:     # < 0.1%
            regime = 'high'
        else:
            regime = 'extreme'

        return vol, regime

    def get_adaptive_parameters(self, current_price: float, position_type: str = 'LONG') -> AdaptiveParameters:
        """
        Get volatility-scaled trading parameters.

        This is the KEY function - everything scales with volatility.
        """
        vol, regime = self.get_current_volatility()

        # TP and SL as multiples of current volatility
        tp_pct = vol * self.tp_vol_multiple
        sl_pct = vol * self.sl_vol_multiple

        # Convert to price levels based on position type
        if position_type == 'LONG':
            tp_price = current_price * (1 + tp_pct)
            sl_price = current_price * (1 - sl_pct)
        else:  # SHORT
            tp_price = current_price * (1 - tp_pct)
            sl_price = current_price * (1 + sl_pct)

        # Estimate time to reach TP based on current volatility
        # Using random walk theory: E[time to move X] ~ (X/sigma)^2
        if len(self.timestamps) >= 2:
            avg_interval = (self.timestamps[-1] - self.timestamps[0]) / (len(self.timestamps) - 1)
            expected_samples = self.tp_vol_multiple ** 2  # Variance scales with sqrt(t)
            expected_hold_secs = expected_samples * avg_interval

            # MINIMUM HOLD TIME based on regime to prevent premature timeout
            # In low volatility, price needs more time to move
            if regime == 'low':
                min_hold = 30.0  # At least 30 seconds in low vol
            elif regime == 'medium':
                min_hold = 15.0  # At least 15 seconds in medium vol
            elif regime == 'high':
                min_hold = 5.0   # At least 5 seconds in high vol
            else:  # extreme
                min_hold = 2.0   # At least 2 seconds in extreme vol

            expected_hold_secs = max(expected_hold_secs, min_hold)
        else:
            expected_hold_secs = 60.0  # Default 1 minute

        return AdaptiveParameters(
            take_profit=tp_pct,
            stop_loss=sl_pct,
            tp_price=tp_price,
            sl_price=sl_price,
            expected_hold_secs=expected_hold_secs,
            current_volatility=vol,
            vol_regime=regime,
        )

    def should_exit(self, entry_price: float, current_price: float,
                    position_type: str, params: AdaptiveParameters) -> Tuple[bool, str, bool]:
        """
        Check if position should exit based on ADAPTIVE TP/SL.

        Returns: (should_exit, reason, is_win)
        """
        if position_type == 'LONG':
            pnl_pct = (current_price - entry_price) / entry_price
            if pnl_pct >= params.take_profit:
                return True, 'TP', True
            elif pnl_pct <= -params.stop_loss:
                return True, 'SL', False
        else:  # SHORT
            pnl_pct = (entry_price - current_price) / entry_price
            if pnl_pct >= params.take_profit:
                return True, 'TP', True
            elif pnl_pct <= -params.stop_loss:
                return True, 'SL', False

        return False, '', False

    def get_breakeven_winrate(self) -> float:
        """
        Calculate breakeven win rate for current TP/SL ratio.

        Formula: BE_WR = SL / (SL + TP)

        With TP=2*vol and SL=1.5*vol:
        BE_WR = 1.5 / (1.5 + 2.0) = 1.5 / 3.5 = 42.86%

        So we only need >42.86% accuracy to be profitable!
        """
        return self.sl_vol_multiple / (self.sl_vol_multiple + self.tp_vol_multiple)

    def get_expected_edge(self, win_rate: float) -> float:
        """
        Calculate expected edge per trade.

        Formula: Edge = WR * TP - (1-WR) * SL

        Example with 55% win rate:
        Edge = 0.55 * 2.0 - 0.45 * 1.5 = 1.1 - 0.675 = 0.425 volatility units
        """
        return win_rate * self.tp_vol_multiple - (1 - win_rate) * self.sl_vol_multiple


# Test it
if __name__ == '__main__':
    trader = AdaptiveVolatilityTrader()

    print('=' * 60)
    print('ADAPTIVE VOLATILITY-SCALED TRADING SYSTEM')
    print('=' * 60)

    print(f'\nBreakeven win rate: {trader.get_breakeven_winrate()*100:.2f}%')
    print(f'Edge at 55% WR: {trader.get_expected_edge(0.55):.3f} vol units')
    print(f'Edge at 52% WR: {trader.get_expected_edge(0.52):.3f} vol units')

    # Simulate LOW volatility (stable market like now)
    base = 91000
    np.random.seed(42)
    for i in range(100):
        price = base + np.random.randn() * 5  # ~$5 noise = 0.005%
        trader.update(price, i * 0.5)  # 0.5 second intervals

    params = trader.get_adaptive_parameters(base, 'LONG')
    print(f'\n{"="*60}')
    print('LOW VOLATILITY REGIME (current BTC market):')
    print(f'{"="*60}')
    print(f'  Current volatility: {params.current_volatility*100:.4f}%')
    print(f'  Regime: {params.vol_regime}')
    print(f'  TP: {params.take_profit*100:.4f}% (${params.tp_price - base:.2f})')
    print(f'  SL: {params.stop_loss*100:.4f}% (${base - params.sl_price:.2f})')
    print(f'  Expected hold: {params.expected_hold_secs:.1f} seconds')

    # Simulate HIGH volatility
    trader2 = AdaptiveVolatilityTrader()
    for i in range(100):
        price = base + np.random.randn() * 200  # ~$200 noise = 0.22%
        trader2.update(price, i * 0.5)

    params2 = trader2.get_adaptive_parameters(base, 'LONG')
    print(f'\n{"="*60}')
    print('HIGH VOLATILITY REGIME:')
    print(f'{"="*60}')
    print(f'  Current volatility: {params2.current_volatility*100:.4f}%')
    print(f'  Regime: {params2.vol_regime}')
    print(f'  TP: {params2.take_profit*100:.4f}% (${params2.tp_price - base:.2f})')
    print(f'  SL: {params2.stop_loss*100:.4f}% (${base - params2.sl_price:.2f})')
    print(f'  Expected hold: {params2.expected_hold_secs:.1f} seconds')

    print(f'\n{"="*60}')
    print('KEY INSIGHT:')
    print(f'{"="*60}')
    print('  In LOW vol: Small TP/SL in $ but same in vol units')
    print('  In HIGH vol: Large TP/SL in $ but same in vol units')
    print('  The EDGE stays constant because parameters scale!')
