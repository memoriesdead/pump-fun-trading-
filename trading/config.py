"""
Trading Configuration
=====================

Single config used by BOTH paper and real trading.
RenTech 1:1 parity - same parameters for both modes.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class TradingConfig:
    """
    Trading configuration - IDENTICAL for paper and real.

    All thresholds, sizing, and exit rules must be the same
    to ensure 1:1 parity between paper and real trading.
    """

    # === Position Sizing ===
    max_position_pct: float = 0.20          # Max 20% of capital per trade
    min_position_sol: float = 0.01          # Minimum 0.01 SOL
    max_position_sol: float = 5.0           # Maximum 5 SOL per trade
    kelly_fraction: float = 0.5             # Half-Kelly for safety

    # === Entry Criteria (Rug Detection) ===
    max_rug_score: float = 0.30             # AggregateRugScore (9550)
    max_instant_rug: float = 0.20           # InstantRugAlert (9541)
    max_mint_risk: float = 0.50             # MintAuthorityRisk (9521)
    max_honeypot: float = 0.30              # HoneypotIndicator (9524)

    # === Entry Criteria (Quality) ===
    min_quality_signals: int = 3            # Need 3+ quality signals
    min_graduation_velocity: float = 0.30   # GraduationVelocity (9401)
    min_momentum: float = 0.50              # VelocityAcceleration (9353)
    min_entry_window: float = 0.50          # OptimalEntryWindow (9391)
    min_directional: float = 0.60           # DirectionalIntensity (9356)
    min_burst: float = 0.40                 # BurstIntensity (9361)

    # === Exit Rules ===
    stop_loss: float = 0.10                 # -10% stop loss (tight for scalping)
    take_profit_1: float = 0.15             # +15% first target (quick scalp)
    take_profit_2: float = 0.30             # +30% second target
    take_profit_3: float = 0.50             # +50% final target (rare)
    trailing_stop: float = 0.08             # 8% trailing stop (tight)
    max_hold_time: float = 5.0              # 5 SECONDS max hold - pump.fun is microseconds

    # === Risk Management ===
    max_open_positions: int = 3             # Max concurrent positions
    max_daily_trades: int = 20              # Max trades per day
    max_daily_loss_pct: float = 0.30        # Stop trading at -30% daily
    max_drawdown_pct: float = 0.50          # Stop trading at -50% drawdown

    # === Execution ===
    slippage_tolerance: float = 0.05        # 5% max slippage
    priority_fee_lamports: int = 100000     # Priority fee (0.0001 SOL)
    confirmation_timeout: float = 30.0      # 30 second confirmation timeout

    # === Signal IDs to use ===
    safety_signals: List[int] = field(default_factory=lambda: [
        9550,   # AggregateRugScore
        9541,   # InstantRugAlert
        9521,   # MintAuthorityRisk
        9524,   # HoneypotIndicator
    ])

    quality_signals: List[int] = field(default_factory=lambda: [
        9401,   # GraduationVelocity
        9353,   # VelocityAcceleration
        9391,   # OptimalEntryWindow
        9356,   # DirectionalIntensity
        9361,   # BurstIntensity
    ])

    def to_dict(self) -> dict:
        return {
            'max_position_pct': self.max_position_pct,
            'stop_loss': self.stop_loss,
            'take_profit_1': self.take_profit_1,
            'take_profit_2': self.take_profit_2,
            'take_profit_3': self.take_profit_3,
            'trailing_stop': self.trailing_stop,
            'max_hold_time': self.max_hold_time,
            'max_open_positions': self.max_open_positions,
            'max_daily_trades': self.max_daily_trades,
        }


# Default configuration
DEFAULT_CONFIG = TradingConfig()


# Aggressive config for testing
AGGRESSIVE_CONFIG = TradingConfig(
    max_position_pct=0.25,
    min_quality_signals=2,
    stop_loss=0.25,
    take_profit_1=0.25,
    take_profit_2=0.40,
    take_profit_3=0.75,
    max_hold_time=1800,  # 30 minutes
)


# Conservative config
CONSERVATIVE_CONFIG = TradingConfig(
    max_position_pct=0.10,
    min_quality_signals=4,
    max_rug_score=0.20,
    stop_loss=0.15,
    take_profit_1=0.40,
    take_profit_2=0.60,
    take_profit_3=1.20,
    max_open_positions=2,
)
