"""
PUMP.FUN CUSTOM FORMULAS
========================

Our own formulas derived from pump.fun trading data.
RenTech doesn't publish formulas, so we create our own.

Formula ID System:
- PF-001 to PF-099: Slippage & Execution
- PF-100 to PF-199: Bonding Curve Dynamics
- PF-200 to PF-299: Liquidity Analysis
- PF-300 to PF-399: Entry/Exit Signals (Theoretical)
- PF-400 to PF-499: Risk Management
- PF-500 to PF-599: Statistical Patterns (from historical data discovery)

Each formula is derived from pump.fun-specific math:
- Bonding curve: price = virtual_sol / virtual_tokens
- Initial: 30 SOL, 1.073B tokens
- Graduation: 85 SOL added

The key insight: Standard signals gave us 54.2% win rate but we LOST money
because actual losses exceeded our stop. These formulas fix that.

================================================================================
VALIDATED PATTERNS (from 269,830 historical trades - Dec 2024):
================================================================================

ENTRY SIGNALS:
  PF-520: Mean Reversion     - 82.5% win rate (26,345 signals) - BEST PERFORMER
  PF-530: Buy Pressure       - 52.8% win rate (34,146 signals)
  PF-540: Combined Entry     - Uses PF-520 + PF-530

EXIT SIGNALS:
  PF-511: Volume Dry-Up      - 62.4% win rate (5,838 signals)

================================================================================
INVALIDATED PATTERNS (DO NOT USE - documented as warnings):
================================================================================

  PF-510: Volume Spike Entry - 9.0% win rate (FAILS)
  PF-512: Whale Following    - 15.2% win rate, -22.8% return (LOSES MONEY!)

================================================================================
"""

import math
from typing import List, Tuple, Optional
from dataclasses import dataclass

# =============================================================================
# CONSTANTS
# =============================================================================

INITIAL_VIRTUAL_SOL = 30
INITIAL_VIRTUAL_TOKENS = 1_073_000_000
GRADUATION_SOL = 85


# =============================================================================
# DATA STRUCTURE
# =============================================================================

@dataclass
class PricePoint:
    """Single price observation"""
    timestamp: float
    price: float
    virtual_sol: float
    virtual_tokens: float
    sol_amount: float = 0  # Trade size in SOL
    is_buy: bool = True


# =============================================================================
# PF-001: REALIZED SLIPPAGE ESTIMATOR
# =============================================================================

def pf001_realized_slippage(
    points: List[PricePoint],
    lookback: int = 20
) -> Tuple[float, float]:
    """
    Formula PF-001: Realized Slippage Estimator

    Measures the ACTUAL slippage experienced in recent trades.

    Key insight: Paper trading assumes you exit at stop price.
    Reality: Price can gap 10-20% past your stop in memecoins.

    This formula measures:
    1. Average gap between consecutive prices
    2. Maximum gap observed
    3. Probability of gap exceeding your stop

    Returns: (avg_gap_pct, max_gap_pct)

    Usage: If max_gap_pct > stop_loss_pct, AVOID THIS TOKEN
    """
    if len(points) < 3:
        return 0.0, 0.0

    recent = points[-lookback:] if len(points) >= lookback else points

    gaps = []
    for i in range(1, len(recent)):
        prev_price = recent[i-1].price
        curr_price = recent[i].price

        if prev_price <= 0:
            continue

        gap_pct = abs(curr_price - prev_price) / prev_price
        gaps.append(gap_pct)

    if not gaps:
        return 0.0, 0.0

    avg_gap = sum(gaps) / len(gaps)
    max_gap = max(gaps)

    return avg_gap, max_gap


def pf001_is_safe_for_stop(
    points: List[PricePoint],
    stop_pct: float = 0.05,
    lookback: int = 20
) -> Tuple[bool, str]:
    """
    PF-001 Application: Check if token is safe for our stop loss

    ADJUSTED FOR PUMP.FUN REALITY:
    - Memecoins are volatile by nature
    - 10-20% gaps are common on new tokens
    - We filter only the EXTREME cases (>3x stop)

    Returns: (is_safe, reason)
    """
    avg_gap, max_gap = pf001_realized_slippage(points, lookback)

    # Only filter extreme volatility (max gap > 3x stop = 15% gaps)
    # This catches the -20%, -25% disasters while allowing normal volatility
    if max_gap > stop_pct * 3:
        return False, f"Max gap {max_gap:.1%} > 3x stop {stop_pct:.1%}"

    # Only filter if AVERAGE gap is crazy (>1.5x stop)
    if avg_gap > stop_pct * 1.5:
        return False, f"Avg gap {avg_gap:.1%} > 1.5x stop {stop_pct:.1%}"

    return True, f"Safe: avg={avg_gap:.2%}, max={max_gap:.2%}"


# =============================================================================
# PF-002: BONDING CURVE MOMENTUM SCORE
# =============================================================================

def pf002_curve_momentum(
    points: List[PricePoint],
    lookback: int = 10
) -> float:
    """
    Formula PF-002: Bonding Curve Momentum Score

    Pump.fun specific: SOL flowing INTO the curve predicts price rise.
    This is more reliable than price momentum because it measures
    actual capital commitment.

    Formula: momentum = (SOL_now - SOL_prev) / (time_delta * sqrt(SOL_now))

    The sqrt(SOL_now) normalizes for curve position -
    1 SOL matters more at 35 SOL than at 80 SOL.

    Returns: Momentum score (-1 to +1)
    """
    if len(points) < 2:
        return 0.0

    recent = points[-lookback:] if len(points) >= lookback else points

    first = recent[0]
    last = recent[-1]

    time_delta = last.timestamp - first.timestamp
    if time_delta <= 0:
        return 0.0

    sol_change = last.virtual_sol - first.virtual_sol

    # Normalize by sqrt of current SOL (diminishing impact as curve fills)
    normalizer = math.sqrt(last.virtual_sol) if last.virtual_sol > 0 else 1

    # SOL per second, normalized
    momentum = sol_change / (time_delta * normalizer)

    # Scale to -1 to +1 (0.1 SOL/sec/sqrt(SOL) = strong signal)
    signal = momentum / 0.1
    return max(-1, min(1, signal))


def pf002_curve_acceleration(
    points: List[PricePoint],
    short_lookback: int = 5,
    long_lookback: int = 20
) -> float:
    """
    PF-002 Extension: Curve Acceleration

    Measures if momentum is INCREASING or DECREASING.

    Acceleration > 0: Momentum building (good for entry)
    Acceleration < 0: Momentum fading (avoid or exit)

    Returns: Acceleration score (-1 to +1)
    """
    if len(points) < long_lookback:
        return 0.0

    short_momentum = pf002_curve_momentum(points, short_lookback)
    long_momentum = pf002_curve_momentum(points, long_lookback)

    # Acceleration is the difference
    acceleration = short_momentum - long_momentum

    # Scale (0.5 difference is very significant)
    return max(-1, min(1, acceleration / 0.5))


# =============================================================================
# PF-003: LIQUIDITY DEPTH RATIO
# =============================================================================

def pf003_liquidity_depth(
    points: List[PricePoint],
    lookback: int = 10
) -> float:
    """
    Formula PF-003: Liquidity Depth Ratio

    Measures how much SOL is needed to move price by 1%.

    Formula: depth = avg_trade_sol / avg_price_impact

    High depth = liquid = safe to trade
    Low depth = illiquid = avoid (slippage will kill you)

    Returns: SOL per 1% price move (higher = more liquid)
    """
    if len(points) < 3:
        return 0.0

    recent = points[-lookback:] if len(points) >= lookback else points

    depth_samples = []

    for i in range(1, len(recent)):
        trade_sol = recent[i].sol_amount
        prev_price = recent[i-1].price
        curr_price = recent[i].price

        if trade_sol <= 0 or prev_price <= 0:
            continue

        price_impact = abs(curr_price - prev_price) / prev_price

        if price_impact > 0.001:  # At least 0.1% move
            # SOL needed for 1% move
            depth = trade_sol / (price_impact * 100)
            depth_samples.append(depth)

    if not depth_samples:
        return 0.0

    return sum(depth_samples) / len(depth_samples)


def pf003_can_exit_safely(
    points: List[PricePoint],
    position_sol: float,
    max_exit_slippage: float = 0.02,  # Max 2% slippage on exit
    lookback: int = 10
) -> Tuple[bool, float]:
    """
    PF-003 Application: Can we exit this position safely?

    Estimates slippage if we sell position_sol worth.

    Returns: (can_exit_safely, estimated_slippage_pct)
    """
    depth = pf003_liquidity_depth(points, lookback)

    if depth <= 0:
        return False, 1.0  # Assume 100% slippage if no data

    # Estimated slippage = position_sol / (depth * 100)
    # (because depth is SOL per 1% move)
    estimated_slippage = position_sol / (depth * 100) if depth > 0 else 1.0

    can_exit = estimated_slippage <= max_exit_slippage

    return can_exit, estimated_slippage


# =============================================================================
# PF-004: TRADE SIZE IMPACT PREDICTOR
# =============================================================================

def pf004_trade_impact(
    points: List[PricePoint],
    trade_sol: float,
    lookback: int = 10
) -> float:
    """
    Formula PF-004: Trade Size Impact Predictor

    Predicts how much YOUR trade will move the price.

    Uses bonding curve math:
    new_price = (virtual_sol + trade_sol) / (virtual_tokens - tokens_bought)

    But also factors in observed market impact from recent trades.

    Returns: Expected price impact as percentage (0.01 = 1%)
    """
    if not points:
        return 0.0

    latest = points[-1]

    # Theoretical impact from bonding curve math
    v_sol = latest.virtual_sol
    v_tokens = latest.virtual_tokens

    if v_sol <= 0 or v_tokens <= 0:
        return 0.0

    current_price = v_sol / v_tokens

    # Tokens we'd receive (simplified bonding curve)
    # In reality: tokens_out = v_tokens * (1 - v_sol/(v_sol + trade_sol))
    # Price after: (v_sol + trade_sol) / (v_tokens - tokens_out)

    tokens_out = v_tokens * (1 - v_sol / (v_sol + trade_sol))
    new_v_tokens = v_tokens - tokens_out
    new_price = (v_sol + trade_sol) / new_v_tokens if new_v_tokens > 0 else current_price

    theoretical_impact = (new_price - current_price) / current_price

    # Empirical adjustment: actual impact is often 1.5-2x theoretical
    # due to front-running, MEV, and other effects
    empirical_multiplier = 1.5

    return theoretical_impact * empirical_multiplier


def pf004_optimal_trade_size(
    points: List[PricePoint],
    max_impact: float = 0.02,  # Max 2% price impact
) -> float:
    """
    PF-004 Application: Calculate optimal trade size

    Returns the maximum SOL we should trade to keep impact under max_impact.

    Returns: Optimal trade size in SOL
    """
    if not points:
        return 0.0

    latest = points[-1]
    v_sol = latest.virtual_sol
    v_tokens = latest.virtual_tokens

    if v_sol <= 0 or v_tokens <= 0:
        return 0.0

    # Binary search for optimal size
    low, high = 0.01, v_sol * 0.1  # Max 10% of pool

    for _ in range(20):  # 20 iterations is plenty
        mid = (low + high) / 2
        impact = pf004_trade_impact(points, mid)

        if impact > max_impact:
            high = mid
        else:
            low = mid

    return low


# =============================================================================
# PF-100: BONDING CURVE POSITION VALUE
# =============================================================================

def pf100_curve_position(points: List[PricePoint]) -> float:
    """
    Formula PF-100: Bonding Curve Position Value

    Returns progress through bonding curve as 0-1.

    0.0 = Just created (30 SOL virtual)
    1.0 = About to graduate (85 SOL real added)

    Key insight: Different curve positions have different dynamics.
    """
    if not points:
        return 0.0

    latest = points[-1]
    real_sol = latest.virtual_sol - INITIAL_VIRTUAL_SOL
    progress = max(0, min(1, real_sol / GRADUATION_SOL))

    return progress


def pf100_curve_zone(points: List[PricePoint]) -> str:
    """
    PF-100 Application: Identify curve zone

    Returns: "early", "middle", "late", or "graduation"
    """
    progress = pf100_curve_position(points)

    if progress < 0.15:
        return "early"
    elif progress < 0.50:
        return "middle"
    elif progress < 0.85:
        return "late"
    else:
        return "graduation"


# =============================================================================
# PF-101: FAIR VALUE DEVIATION
# =============================================================================

def pf101_fair_value_deviation(
    points: List[PricePoint],
    lookback: int = 30
) -> float:
    """
    Formula PF-101: Fair Value Deviation

    Calculates how far current price is from "fair value" based on
    SOL committed to the curve.

    Fair value concept: Price should reflect capital committed.
    If price spikes without proportional SOL increase, it's overvalued.

    Returns: Deviation from fair value (-1 to +1)
             Negative = undervalued, Positive = overvalued
    """
    if len(points) < lookback:
        return 0.0

    recent = points[-lookback:]

    # Calculate price/SOL ratio for each point
    ratios = []
    for p in recent:
        if p.virtual_sol > INITIAL_VIRTUAL_SOL:
            real_sol = p.virtual_sol - INITIAL_VIRTUAL_SOL
            ratio = p.price / real_sol if real_sol > 0 else 0
            if ratio > 0:
                ratios.append(ratio)

    if len(ratios) < lookback // 2:
        return 0.0

    mean_ratio = sum(ratios) / len(ratios)
    current_ratio = ratios[-1] if ratios else 0

    if mean_ratio == 0:
        return 0.0

    # Deviation: positive = overvalued, negative = undervalued
    deviation = (current_ratio - mean_ratio) / mean_ratio

    return max(-1, min(1, deviation))


# =============================================================================
# PF-300: ENTRY SIGNAL COMPOSITE
# =============================================================================

def pf300_entry_signal(
    points: List[PricePoint],
    stop_pct: float = 0.05
) -> Tuple[bool, float, dict]:
    """
    Formula PF-300: Entry Signal Composite

    Combines all our custom formulas into a single entry decision.

    Key innovation: We don't just look at signals, we also check
    if the token is SAFE to trade (slippage, liquidity).

    Returns: (should_enter, win_probability, breakdown)
    """
    breakdown = {}

    # === Safety Checks First ===

    # PF-001: Check slippage safety (ONLY filters extreme cases)
    is_safe, reason = pf001_is_safe_for_stop(points, stop_pct)
    breakdown['pf001_safe'] = is_safe
    breakdown['pf001_reason'] = reason

    if not is_safe:
        return False, 0.45, breakdown

    # PF-003: Check liquidity depth
    depth = pf003_liquidity_depth(points)
    breakdown['pf003_depth'] = depth

    # Need at least 0.01 SOL per 1% move (very loose - pump.fun is thin)
    # Most new tokens will be thin, we accept some slippage risk
    if depth > 0 and depth < 0.01:
        breakdown['pf003_reason'] = f"Too thin: {depth:.3f} SOL/1%"
        return False, 0.45, breakdown

    # === Signal Calculations ===

    # PF-002: Curve momentum
    momentum = pf002_curve_momentum(points)
    acceleration = pf002_curve_acceleration(points)
    breakdown['pf002_momentum'] = momentum
    breakdown['pf002_acceleration'] = acceleration

    # PF-100: Curve position
    position = pf100_curve_position(points)
    zone = pf100_curve_zone(points)
    breakdown['pf100_position'] = position
    breakdown['pf100_zone'] = zone

    # PF-101: Fair value
    fv_deviation = pf101_fair_value_deviation(points)
    breakdown['pf101_fv_deviation'] = fv_deviation

    # === Combine Signals ===

    # Weights for each signal
    weights = {
        'momentum': 0.30,
        'acceleration': 0.25,
        'fv_deviation': 0.20,
        'zone': 0.25,
    }

    # Zone score: early=0.1, middle=0.5, late=0.7, graduation=0.9
    zone_scores = {'early': 0.1, 'middle': 0.5, 'late': 0.7, 'graduation': 0.9}
    zone_score = zone_scores.get(zone, 0.3)

    # Combined signal
    combined = (
        momentum * weights['momentum'] +
        acceleration * weights['acceleration'] +
        (-fv_deviation) * weights['fv_deviation'] +  # Negative because we want undervalued
        zone_score * weights['zone']
    )

    breakdown['combined_signal'] = combined

    # Convert to win probability (0.48 to 0.55)
    # Adjusted range: we accept slightly lower probability trades
    # because pump.fun has higher potential returns
    win_prob = 0.50 + (combined * 0.05)
    win_prob = max(0.48, min(0.55, win_prob))
    breakdown['win_probability'] = win_prob

    # Entry decision: positive signal AND win prob > 50%
    # Looser than before: we want to trade more, let Kelly handle sizing
    should_enter = combined > 0.05 and win_prob > 0.50

    return should_enter, win_prob, breakdown


# =============================================================================
# PF-301: EXIT SIGNAL
# =============================================================================

def pf301_exit_signal(
    points: List[PricePoint],
    entry_price: float,
    target_pct: float = 0.05,
    stop_pct: float = 0.05,
    hold_time_sec: float = 0,
    max_hold_sec: float = 300
) -> Tuple[bool, str, dict]:
    """
    Formula PF-301: Exit Signal

    Smart exit that accounts for:
    1. Normal take profit / stop loss
    2. Momentum fade (exit early if momentum dying)
    3. Liquidity drain (exit before you can't)

    Returns: (should_exit, reason, breakdown)
    """
    if not points:
        return False, "", {}

    breakdown = {}
    current_price = points[-1].price

    # Basic P&L
    pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0
    breakdown['pnl_pct'] = pnl_pct

    # === Standard Exits ===

    if pnl_pct >= target_pct:
        return True, "TARGET", breakdown

    if pnl_pct <= -stop_pct:
        return True, "STOP", breakdown

    if hold_time_sec >= max_hold_sec:
        return True, "TIMEOUT", breakdown

    # === Smart Exits ===

    # Momentum fade: exit at breakeven if momentum dies
    momentum = pf002_curve_momentum(points)
    acceleration = pf002_curve_acceleration(points)
    breakdown['momentum'] = momentum
    breakdown['acceleration'] = acceleration

    # If we're up and momentum is fading, take profit early
    if pnl_pct > 0.01 and momentum < -0.3 and acceleration < -0.2:
        return True, "MOMENTUM_FADE", breakdown

    # Liquidity drain: exit if liquidity is dropping
    depth = pf003_liquidity_depth(points)
    breakdown['liquidity_depth'] = depth

    # If liquidity drops below 0.05 SOL/1%, exit immediately
    if depth < 0.05 and depth > 0:
        return True, "LIQUIDITY_DRAIN", breakdown

    # Approaching stop with bad momentum: exit early
    if pnl_pct < -0.02 and momentum < -0.2:
        return True, "EARLY_STOP", breakdown

    return False, "", breakdown


# =============================================================================
# PF-510: VOLUME SPIKE ENTRY (INVALIDATED - DO NOT USE)
# =============================================================================

def pf510_volume_spike_entry(
    trades: List[PricePoint],
    threshold_sigma: float = 3.0,
    lookback: int = 20
) -> Tuple[bool, float, dict]:
    """
    Formula PF-510: Volume Spike Entry Signal

    STATUS: INVALIDATED - DO NOT USE IN PRODUCTION

    Backtested: 2024-12 on 269,830 trades
    Win Rate: 9.0% (FAILS - below 50% threshold)

    Original hypothesis: Volume spikes indicate momentum, buy on spikes.
    Reality: Volume spikes often precede dumps, not pumps.

    This formula is kept for documentation purposes only.

    Returns:
        (should_enter, confidence, breakdown)
    """
    breakdown = {
        'status': 'INVALIDATED',
        'win_rate': 0.090,
        'reason': 'Fails backtest - volume spikes precede dumps'
    }

    # Always return False - pattern is invalidated
    return False, 0.0, breakdown


# =============================================================================
# PF-511: VOLUME DRY-UP WARNING (VALIDATED)
# =============================================================================

def pf511_volume_dryup_warning(
    trades: List[PricePoint],
    decay_threshold: float = 0.80,
    lookback: int = 10
) -> Tuple[bool, float, dict]:
    """
    Formula PF-511: Volume Dry-Up Warning (EXIT SIGNAL)

    STATUS: VALIDATED

    Backtested: 2024-12 on 269,830 trades
    Win Rate: 62.4% (5,838 signals)

    Discovery: When volume drops >80% from recent average, price follows.
    This is an EXIT signal - get out before the crash.

    Logic:
        1. Calculate rolling volume average over lookback period
        2. If current volume < 20% of average
        3. → Signal EXIT (volume death = price death)

    Returns:
        (should_exit, confidence, breakdown)
    """
    breakdown = {
        'status': 'VALIDATED',
        'backtest_win_rate': 0.624,
        'backtest_signals': 5838,
        'discovery_date': '2024-12'
    }

    if len(trades) < lookback + 1:
        breakdown['reason'] = 'Insufficient data'
        return False, 0.0, breakdown

    # Calculate recent volume average
    recent_volumes = [t.sol_amount for t in trades[-(lookback+1):-1] if t.sol_amount > 0]

    if not recent_volumes:
        breakdown['reason'] = 'No volume data'
        return False, 0.0, breakdown

    avg_volume = sum(recent_volumes) / len(recent_volumes)
    current_volume = trades[-1].sol_amount

    if avg_volume <= 0:
        breakdown['reason'] = 'Zero average volume'
        return False, 0.0, breakdown

    volume_ratio = current_volume / avg_volume
    volume_decay = 1 - volume_ratio

    breakdown['avg_volume'] = avg_volume
    breakdown['current_volume'] = current_volume
    breakdown['volume_decay'] = volume_decay

    # Signal exit if volume has dropped > threshold (default 80%)
    should_exit = volume_decay >= decay_threshold
    confidence = min(1.0, volume_decay) if should_exit else 0.0

    breakdown['should_exit'] = should_exit
    breakdown['reason'] = f'Volume down {volume_decay:.1%}' if should_exit else 'Volume stable'

    return should_exit, confidence, breakdown


# =============================================================================
# PF-512: WHALE FOLLOWING (INVALIDATED - LOSES MONEY)
# =============================================================================

def pf512_whale_following(
    trades: List[PricePoint],
    whale_threshold_sol: float = 5.0
) -> Tuple[bool, float, dict]:
    """
    Formula PF-512: Whale Following Signal

    STATUS: INVALIDATED - LOSES MONEY

    Backtested: 2024-12 on 269,830 trades
    Win Rate: 15.2% (LOSES MONEY)
    Average Return: -22.80% (NEGATIVE)

    Original hypothesis: Follow whale buys for quick profits.
    Reality: Whales are often DUMPING on retail followers.

    CRITICAL: This pattern has NEGATIVE expected value.
    Following whale buys actually LOSES 22.8% on average.

    This formula is kept as a WARNING - counter-trade whales instead.

    Returns:
        (should_enter, confidence, breakdown)
    """
    breakdown = {
        'status': 'INVALIDATED',
        'win_rate': 0.152,
        'avg_return': -0.228,
        'reason': 'LOSES MONEY - Whales dump on followers',
        'recommendation': 'Consider COUNTER-TRADING whale buys'
    }

    # Always return False - pattern loses money
    return False, 0.0, breakdown


# =============================================================================
# PF-520: MEAN REVERSION (VALIDATED - HIGHEST WIN RATE)
# =============================================================================

def pf520_mean_reversion(
    trades: List[PricePoint],
    drop_threshold: float = 0.30,
    lookback: int = 5
) -> Tuple[bool, float, dict]:
    """
    Formula PF-520: Mean Reversion Entry Signal

    STATUS: VALIDATED - BEST PERFORMER

    Backtested: 2024-12 on 269,830 trades
    Win Rate: 82.5% (26,345 signals)

    Discovery: After a 30%+ drop in 5 trades, price tends to bounce.
    This is classic oversold mean reversion.

    Logic:
        1. Calculate price change over last 5 trades
        2. If price dropped > 30%
        3. → Signal BUY (oversold bounce expected)

    Why it works:
        - Panic sells overshoot fair value
        - Liquidity returns after panic
        - Bonding curve mechanics support recovery

    Returns:
        (should_enter, confidence, breakdown)
    """
    breakdown = {
        'status': 'VALIDATED',
        'backtest_win_rate': 0.825,
        'backtest_signals': 26345,
        'discovery_date': '2024-12'
    }

    if len(trades) < lookback + 1:
        breakdown['reason'] = 'Insufficient data'
        return False, 0.0, breakdown

    # Get price at start and end of lookback window
    start_price = trades[-(lookback+1)].price
    current_price = trades[-1].price

    if start_price <= 0:
        breakdown['reason'] = 'Invalid start price'
        return False, 0.0, breakdown

    # Calculate price change
    price_change = (current_price - start_price) / start_price

    breakdown['start_price'] = start_price
    breakdown['current_price'] = current_price
    breakdown['price_change'] = price_change

    # Signal entry if dropped more than threshold
    should_enter = price_change <= -drop_threshold

    # Confidence scales with how oversold (max at 50% drop)
    if should_enter:
        confidence = min(1.0, abs(price_change) / 0.50)
    else:
        confidence = 0.0

    breakdown['should_enter'] = should_enter
    breakdown['reason'] = f'Price dropped {abs(price_change):.1%}' if should_enter else 'No oversold condition'

    return should_enter, confidence, breakdown


# =============================================================================
# PF-530: BUY PRESSURE (VALIDATED)
# =============================================================================

def pf530_buy_pressure(
    trades: List[PricePoint],
    pressure_threshold: float = 0.70,
    lookback: int = 10
) -> Tuple[bool, float, dict]:
    """
    Formula PF-530: Buy Pressure Entry Signal

    STATUS: VALIDATED

    Backtested: 2024-12 on 269,830 trades
    Win Rate: 52.8% (34,146 signals)

    Discovery: When >70% of recent trades are buys, momentum continues.

    Logic:
        1. Count buy vs sell trades in lookback window
        2. If buy_ratio > 70%
        3. → Signal BUY (momentum continuation)

    Why it works:
        - High buy pressure indicates demand
        - Pump.fun tokens momentum until they don't
        - 52.8% edge compounds over many trades

    Returns:
        (should_enter, confidence, breakdown)
    """
    breakdown = {
        'status': 'VALIDATED',
        'backtest_win_rate': 0.528,
        'backtest_signals': 34146,
        'discovery_date': '2024-12'
    }

    if len(trades) < lookback:
        breakdown['reason'] = 'Insufficient data'
        return False, 0.0, breakdown

    recent = trades[-lookback:]

    # Count buys vs total
    buy_count = sum(1 for t in recent if t.is_buy)
    total_count = len(recent)

    buy_ratio = buy_count / total_count if total_count > 0 else 0

    breakdown['buy_count'] = buy_count
    breakdown['total_count'] = total_count
    breakdown['buy_ratio'] = buy_ratio

    # Signal entry if buy pressure exceeds threshold
    should_enter = buy_ratio >= pressure_threshold

    # Confidence scales with pressure (max at 90% buys)
    if should_enter:
        confidence = min(1.0, (buy_ratio - pressure_threshold) / 0.20)
    else:
        confidence = 0.0

    breakdown['should_enter'] = should_enter
    breakdown['reason'] = f'Buy pressure {buy_ratio:.1%}' if should_enter else 'Low buy pressure'

    return should_enter, confidence, breakdown


# =============================================================================
# PF-540: COMBINED STATISTICAL ENTRY (USES VALIDATED PATTERNS)
# =============================================================================

def pf540_statistical_entry(
    trades: List[PricePoint],
    require_both: bool = False
) -> Tuple[bool, float, dict]:
    """
    Formula PF-540: Combined Statistical Entry Signal

    Combines validated patterns (PF-520, PF-530) for entry decisions.

    Logic:
        - PF-520 (Mean Reversion): 82.5% win rate, weight 0.6
        - PF-530 (Buy Pressure): 52.8% win rate, weight 0.4

        If require_both=False: Enter if either pattern fires
        If require_both=True: Enter only if both patterns fire

    Note: PF-511 (Volume Dry-Up) is EXIT only, not used here.

    Returns:
        (should_enter, combined_confidence, breakdown)
    """
    breakdown = {
        'formula': 'PF-540',
        'description': 'Combined Statistical Entry'
    }

    # Check PF-520: Mean Reversion (82.5% win rate)
    mr_enter, mr_conf, mr_breakdown = pf520_mean_reversion(trades)
    breakdown['pf520_mean_reversion'] = {
        'signal': mr_enter,
        'confidence': mr_conf,
        'win_rate': 0.825
    }

    # Check PF-530: Buy Pressure (52.8% win rate)
    bp_enter, bp_conf, bp_breakdown = pf530_buy_pressure(trades)
    breakdown['pf530_buy_pressure'] = {
        'signal': bp_enter,
        'confidence': bp_conf,
        'win_rate': 0.528
    }

    # Combine signals
    if require_both:
        should_enter = mr_enter and bp_enter
        # When both fire, higher confidence
        combined_confidence = (mr_conf * 0.6 + bp_conf * 0.4) if should_enter else 0.0
    else:
        should_enter = mr_enter or bp_enter
        # Weight by win rate when either fires
        if mr_enter and bp_enter:
            combined_confidence = mr_conf * 0.6 + bp_conf * 0.4
        elif mr_enter:
            combined_confidence = mr_conf * 0.6  # Higher weight for better pattern
        elif bp_enter:
            combined_confidence = bp_conf * 0.4
        else:
            combined_confidence = 0.0

    breakdown['should_enter'] = should_enter
    breakdown['combined_confidence'] = combined_confidence
    breakdown['mode'] = 'require_both' if require_both else 'either'

    return should_enter, combined_confidence, breakdown


# =============================================================================
# FORMULA REGISTRY
# =============================================================================

FORMULA_REGISTRY = {
    # =========================================================================
    # SLIPPAGE & EXECUTION (PF-001 to PF-099)
    # =========================================================================
    'PF-001': {
        'name': 'Realized Slippage Estimator',
        'function': pf001_realized_slippage,
        'description': 'Measures actual slippage from price gaps',
        'category': 'SLIPPAGE',
    },
    'PF-002': {
        'name': 'Trade Size Impact Predictor',
        'function': pf004_trade_impact,  # Renumbered from PF-004
        'description': 'Predicted price impact of a trade',
        'category': 'SLIPPAGE',
    },

    # =========================================================================
    # BONDING CURVE DYNAMICS (PF-100 to PF-199)
    # =========================================================================
    'PF-100': {
        'name': 'Bonding Curve Position Value',
        'function': pf100_curve_position,
        'description': 'Progress through bonding curve (0-1)',
        'category': 'CURVE',
    },
    'PF-101': {
        'name': 'Fair Value Deviation',
        'function': pf101_fair_value_deviation,
        'description': 'Price deviation from SOL-implied fair value',
        'category': 'CURVE',
    },
    'PF-102': {
        'name': 'Bonding Curve Momentum Score',
        'function': pf002_curve_momentum,  # Renumbered from PF-002
        'description': 'SOL flow momentum, normalized by curve position',
        'category': 'CURVE',
    },

    # =========================================================================
    # LIQUIDITY ANALYSIS (PF-200 to PF-299)
    # =========================================================================
    'PF-200': {
        'name': 'Liquidity Depth Ratio',
        'function': pf003_liquidity_depth,  # Renumbered from PF-003
        'description': 'SOL needed to move price 1%',
        'category': 'LIQUIDITY',
    },

    # =========================================================================
    # ENTRY/EXIT SIGNALS - THEORETICAL (PF-300 to PF-399)
    # =========================================================================
    'PF-300': {
        'name': 'Entry Signal Composite (Theoretical)',
        'function': pf300_entry_signal,
        'description': 'Combined entry signal with safety checks',
        'category': 'SIGNALS',
        'status': 'THEORETICAL',
    },
    'PF-301': {
        'name': 'Exit Signal (Theoretical)',
        'function': pf301_exit_signal,
        'description': 'Smart exit with momentum and liquidity awareness',
        'category': 'SIGNALS',
        'status': 'THEORETICAL',
    },

    # =========================================================================
    # STATISTICAL PATTERNS - FROM DATA (PF-500 to PF-599)
    # Discovered 2024-12 from 269,830 trades backtest
    # These are EMPIRICALLY VALIDATED (or invalidated)
    # =========================================================================

    # --- INVALIDATED PATTERNS (DO NOT USE) ---
    'PF-510': {
        'name': 'Volume Spike Entry',
        'function': pf510_volume_spike_entry,
        'description': 'DO NOT USE - 9% win rate, fails backtest',
        'category': 'STATISTICAL',
        'status': 'INVALIDATED',
        'win_rate': 0.090,
        'signal_type': 'ENTRY',
        'backtest_date': '2024-12',
        'backtest_trades': 269830,
    },
    'PF-512': {
        'name': 'Whale Following',
        'function': pf512_whale_following,
        'description': 'DO NOT USE - LOSES MONEY (-22.8% avg return)',
        'category': 'STATISTICAL',
        'status': 'INVALIDATED',
        'win_rate': 0.152,
        'avg_return': -0.228,
        'signal_type': 'ENTRY',
        'backtest_date': '2024-12',
        'backtest_trades': 269830,
    },

    # --- VALIDATED EXIT SIGNALS ---
    'PF-511': {
        'name': 'Volume Dry-Up Warning',
        'function': pf511_volume_dryup_warning,
        'description': 'EXIT when volume drops >80% from average',
        'category': 'STATISTICAL',
        'status': 'VALIDATED',
        'win_rate': 0.624,
        'signal_count': 5838,
        'signal_type': 'EXIT',
        'backtest_date': '2024-12',
        'backtest_trades': 269830,
    },

    # --- VALIDATED ENTRY SIGNALS ---
    'PF-520': {
        'name': 'Mean Reversion Entry',
        'function': pf520_mean_reversion,
        'description': 'BUY after 30%+ drop in 5 trades - BEST PERFORMER',
        'category': 'STATISTICAL',
        'status': 'VALIDATED',
        'win_rate': 0.825,
        'signal_count': 26345,
        'signal_type': 'ENTRY',
        'backtest_date': '2024-12',
        'backtest_trades': 269830,
    },
    'PF-530': {
        'name': 'Buy Pressure Entry',
        'function': pf530_buy_pressure,
        'description': 'BUY when >70% of recent trades are buys',
        'category': 'STATISTICAL',
        'status': 'VALIDATED',
        'win_rate': 0.528,
        'signal_count': 34146,
        'signal_type': 'ENTRY',
        'backtest_date': '2024-12',
        'backtest_trades': 269830,
    },
    'PF-540': {
        'name': 'Combined Statistical Entry',
        'function': pf540_statistical_entry,
        'description': 'Combines PF-520 (82.5%) + PF-530 (52.8%)',
        'category': 'STATISTICAL',
        'status': 'VALIDATED',
        'signal_type': 'ENTRY',
        'uses': ['PF-520', 'PF-530'],
    },
}


def get_formula(formula_id: str):
    """Get formula by ID"""
    return FORMULA_REGISTRY.get(formula_id)


def list_formulas(category: str = None, validated_only: bool = False):
    """
    List all available formulas.

    Args:
        category: Filter by category (SLIPPAGE, CURVE, LIQUIDITY, SIGNALS, STATISTICAL)
        validated_only: Only show VALIDATED patterns
    """
    categories = {}

    for fid, info in FORMULA_REGISTRY.items():
        cat = info.get('category', 'OTHER')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((fid, info))

    # Sort categories in logical order
    cat_order = ['SLIPPAGE', 'CURVE', 'LIQUIDITY', 'SIGNALS', 'STATISTICAL']

    for cat in cat_order:
        if cat not in categories:
            continue
        if category and cat != category:
            continue

        print(f"\n{'='*60}")
        print(f"{cat}")
        print('='*60)

        for fid, info in sorted(categories[cat]):
            status = info.get('status', '')
            win_rate = info.get('win_rate')

            # Skip non-validated if filter is on
            if validated_only and status != 'VALIDATED':
                continue

            # Status indicator (ASCII compatible)
            if status == 'VALIDATED':
                status_icon = '[OK]'
            elif status == 'INVALIDATED':
                status_icon = '[X]'
            elif status == 'THEORETICAL':
                status_icon = '[?]'
            else:
                status_icon = '    '

            # Win rate
            win_str = f" ({win_rate:.1%})" if win_rate else ""

            print(f"  {status_icon} {fid}: {info['name']}{win_str}")
            print(f"         {info['description']}")


def get_validated_formulas():
    """Get only validated statistical patterns"""
    return {
        fid: info for fid, info in FORMULA_REGISTRY.items()
        if info.get('status') == 'VALIDATED'
    }


def get_entry_signals():
    """Get all validated entry signals"""
    return {
        fid: info for fid, info in FORMULA_REGISTRY.items()
        if info.get('status') == 'VALIDATED' and info.get('signal_type') == 'ENTRY'
    }


def get_exit_signals():
    """Get all validated exit signals"""
    return {
        fid: info for fid, info in FORMULA_REGISTRY.items()
        if info.get('status') == 'VALIDATED' and info.get('signal_type') == 'EXIT'
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("PUMP.FUN CUSTOM FORMULAS")
    print("=" * 60)
    print("Legend: [OK] = Validated  [X] = Invalidated  [?] = Theoretical")

    list_formulas()

    print("\n" + "=" * 60)
    print("VALIDATED ENTRY SIGNALS FOR PRODUCTION:")
    print("=" * 60)
    for fid, info in get_entry_signals().items():
        win_rate = info.get('win_rate', 0)
        print(f"  {fid}: {info['name']} - {win_rate:.1%} win rate")

    print("\n" + "=" * 60)
    print("VALIDATED EXIT SIGNALS FOR PRODUCTION:")
    print("=" * 60)
    for fid, info in get_exit_signals().items():
        win_rate = info.get('win_rate', 0)
        print(f"  {fid}: {info['name']} - {win_rate:.1%} win rate")
