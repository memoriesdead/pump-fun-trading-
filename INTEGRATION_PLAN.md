# Integration Plan: Validated Formulas into RenTech V3

## Current State Analysis

### rentech_v3.py Signal System (Lines 343-413)
The current `_calculate_signals()` method uses 6 theoretical signals:

| Signal | Weight | Current Logic | Backtest Status |
|--------|--------|---------------|-----------------|
| momentum | 25% | SOL flow in 1m window | NOT VALIDATED |
| volume | 20% | Trade size vs average | **PF-510 INVALIDATED (9%)** |
| whale | 20% | Whale trades in 5m | **PF-512 INVALIDATED (-22.8%)** |
| curve | 15% | Curve position 10-70% | NOT VALIDATED |
| pressure | 10% | Buy/sell ratio | **PF-530 VALIDATED (52.8%)** |
| direction | 10% | Is this trade a buy? | NOT VALIDATED |

**CRITICAL FINDING**: 40% of current signal weight (volume + whale) comes from INVALIDATED patterns!

### Current Exit Logic (Lines 474-505)
- Target: +5%
- Stop: -5%
- Timeout: 300s
- Momentum flip: sol_flow_1m < -2.0
- Graduation risk: curve > 80%

**Missing**: PF-511 Volume Dry-Up Warning (62.4% validated)

---

## Validated Formulas to Integrate

### ENTRY SIGNALS (Replace/Add)
| Formula | Win Rate | Current Status | Action |
|---------|----------|----------------|--------|
| PF-520 | 82.5% | NOT USED | **ADD - PRIMARY** |
| PF-530 | 52.8% | Partially used (10% weight) | **INCREASE WEIGHT** |
| PF-540 | Combined | NOT USED | **ADD - COMBINES ABOVE** |

### EXIT SIGNALS (Add)
| Formula | Win Rate | Current Status | Action |
|---------|----------|----------------|--------|
| PF-511 | 62.4% | NOT USED | **ADD** |

### REMOVE/DISABLE
| Formula | Win Rate | Current Status | Action |
|---------|----------|----------------|--------|
| PF-510 | 9.0% | Used (volume signal 20%) | **REMOVE** |
| PF-512 | 15.2% | Used (whale signal 20%) | **REMOVE** |

---

## Integration Steps

### Step 1: Import Formulas
Add import at top of rentech_v3.py:

```python
from formulas.pumpfun_formulas import (
    PricePoint,
    pf520_mean_reversion,
    pf530_buy_pressure,
    pf540_statistical_entry,
    pf511_volume_dryup_warning,
)
```

### Step 2: Add Conversion Helper
TokenState needs to convert trades to PricePoint format:

```python
def to_price_points(self) -> List[PricePoint]:
    """Convert trade history to PricePoint format for formulas"""
    return [
        PricePoint(
            timestamp=t.timestamp.timestamp(),
            price=t.virtual_sol / max(1, t.virtual_tokens),
            virtual_sol=t.virtual_sol,
            virtual_tokens=t.virtual_tokens,
            sol_amount=t.sol_amount,
            is_buy=t.is_buy
        )
        for t in self.trades
    ]
```

### Step 3: Replace _calculate_signals()
Replace the current theoretical signals with validated formulas:

```python
def _calculate_signals(
    self,
    state: TokenState,
    trade: PumpTrade
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate entry signals using VALIDATED formulas only.

    Based on backtest of 269,830 historical trades (Dec 2024).
    """
    signals = {}
    points = state.to_price_points()

    # ========================================
    # VALIDATED ENTRY SIGNALS
    # ========================================

    # PF-520: Mean Reversion (82.5% win rate) - HIGHEST PRIORITY
    # Buy after 30%+ drop in 5 trades
    mr_enter, mr_conf, mr_breakdown = pf520_mean_reversion(points)
    signals['pf520_mean_reversion'] = mr_conf if mr_enter else 0.0

    # PF-530: Buy Pressure (52.8% win rate)
    # Buy when >70% of recent trades are buys
    bp_enter, bp_conf, bp_breakdown = pf530_buy_pressure(points)
    signals['pf530_buy_pressure'] = bp_conf if bp_enter else 0.0

    # ========================================
    # SUPPORTING SIGNALS (not validated but logical)
    # ========================================

    # Curve position (not backtested but mathematically sound)
    # Sweet spot is 10-50% (enough liquidity, room to grow)
    curve_signal = 0.0
    progress = state.curve_progress
    if 10 <= progress <= 50:
        curve_signal = 0.7
    elif 50 < progress <= 70:
        curve_signal = 0.5
    signals['curve_position'] = curve_signal

    # Current trade is a buy (confirming signal)
    signals['is_buy'] = 0.6 if trade.is_buy else 0.0

    # ========================================
    # WEIGHTED COMBINATION
    # ========================================

    # Weights based on backtest win rates
    weights = {
        'pf520_mean_reversion': 0.50,  # 82.5% win rate - PRIMARY
        'pf530_buy_pressure': 0.30,    # 52.8% win rate
        'curve_position': 0.10,        # Supporting
        'is_buy': 0.10,                # Confirming
    }

    # Calculate weighted probability
    win_prob = sum(signals[k] * weights[k] for k in weights)

    # Boost if PF-520 fires (it's our best signal)
    if mr_enter:
        win_prob = max(win_prob, 0.55)  # Minimum 55% if mean reversion fires

    return win_prob, signals
```

### Step 4: Add Volume Dry-Up Exit
Add to `_check_exit()` method:

```python
async def _check_exit(self, position: Position) -> Optional[str]:
    """Check exit conditions using validated formulas."""
    pnl = position.pnl_pct / 100

    # Standard exits
    if pnl >= self.target_pct:
        return "TARGET"
    if pnl <= -self.stop_pct:
        return "STOP"
    if position.hold_time_sec >= self.max_hold_sec:
        return "TIMEOUT"

    # Get token state
    state = self.tokens.get(position.mint)
    if not state:
        return None

    points = state.to_price_points()

    # ========================================
    # PF-511: Volume Dry-Up Warning (62.4% win rate)
    # Exit when volume drops >80% from average
    # ========================================
    should_exit, confidence, breakdown = pf511_volume_dryup_warning(points)
    if should_exit:
        return "VOLUME_DRYUP"

    # Momentum reversal (keep existing)
    if state.sol_flow_1m < -2.0:
        return "MOMENTUM_FLIP"

    # Near graduation (keep existing)
    if state.curve_progress > 80:
        return "GRADUATION_RISK"

    return None
```

### Step 5: Update Signal Logging
Improve logging to show formula IDs:

```python
logger.info(f"ENTRY SIGNAL: {state.mint[:8]}...")
logger.info(f"  PF-520 Mean Reversion: {'FIRE' if mr_enter else 'no'} ({mr_conf:.2f})")
logger.info(f"  PF-530 Buy Pressure: {'FIRE' if bp_enter else 'no'} ({bp_conf:.2f})")
logger.info(f"  Combined Win Prob: {win_prob*100:.1f}%")
```

---

## New Signal Weights (After Integration)

| Signal | Weight | Win Rate | Source |
|--------|--------|----------|--------|
| PF-520 Mean Reversion | 50% | 82.5% | VALIDATED |
| PF-530 Buy Pressure | 30% | 52.8% | VALIDATED |
| Curve Position | 10% | - | Theoretical |
| Is Buy (confirming) | 10% | - | Logical |

**Expected improvement**:
- Current: ~50% win rate (40% weight on invalidated signals)
- After: ~55-60% win rate (80% weight on validated signals)

---

## REMOVED Signals (Critical)

These were causing losses:

1. **Volume Spike (20% weight)** - PF-510 invalidated at 9% win rate
   - Was: "Big trade = buy signal"
   - Reality: Big trades often precede dumps

2. **Whale Following (20% weight)** - PF-512 invalidated at 15.2% win rate
   - Was: "Follow whale buys"
   - Reality: Whales dump on followers, -22.8% avg return

---

## Testing Plan

### Phase 1: Paper Trading Validation
```bash
# Run with new signals for 4 hours
python trading/rentech_v3.py --paper --duration 14400 --capital 100
```

Expected metrics:
- Win rate > 52%
- More PF-520 entries (mean reversion)
- Volume dry-up exits working

### Phase 2: A/B Comparison
Run old vs new in parallel (paper mode):
- Old: rentech_v3_old.py
- New: rentech_v3.py with validated formulas

Compare after 1000 signals each.

### Phase 3: Live Deployment
Only after Phase 1 & 2 show improvement:
```bash
python trading/rentech_v3.py --live --capital 50 --duration 0
```

---

## File Changes Summary

| File | Changes |
|------|---------|
| trading/rentech_v3.py | Replace `_calculate_signals()`, update `_check_exit()`, add import |
| formulas/pumpfun_formulas.py | Already complete |

---

## Risk Mitigation

1. **Keep old signals commented** - Easy rollback if needed
2. **Lower min_win_prob initially** - 0.50 instead of 0.52 to get more signals
3. **Monitor PF-520 specifically** - It's 50% of our signal weight
4. **Log formula breakdowns** - Track which formulas are firing

---

## Success Criteria

| Metric | Current (Est.) | Target |
|--------|----------------|--------|
| Win Rate | ~48% | >52% |
| Avg Return | Negative | Positive |
| Signals/Hour | Unknown | Track |
| PF-520 Fire Rate | 0% | >10% of entries |
| PF-511 Exits | 0 | Some |

---

## Implementation Order

1. [ ] Backup current rentech_v3.py
2. [ ] Add formula imports
3. [ ] Add TokenState.to_price_points() method
4. [ ] Replace _calculate_signals() with validated formulas
5. [ ] Update _check_exit() with PF-511
6. [ ] Update logging
7. [ ] Test paper trading (4 hours minimum)
8. [ ] Compare results
9. [ ] Deploy if improved
