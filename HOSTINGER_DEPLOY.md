# Deploy to Hostinger VPS

## Files to Copy

```
formulas/pumpfun_formulas.py    <- Validated formulas (PF-520, PF-530, PF-511)
trading/rentech_validated.py    <- Engine using validated formulas
```

## Quick Deploy Commands

### 1. SSH into Hostinger
```bash
ssh root@your-hostinger-ip
```

### 2. Navigate to project
```bash
cd /root/pump-fun-trading
# or wherever your project is
```

### 3. Create formulas directory if needed
```bash
mkdir -p formulas
```

### 4. Copy files (run from your local machine)
```bash
scp formulas/pumpfun_formulas.py root@your-hostinger-ip:/root/pump-fun-trading/formulas/
scp trading/rentech_validated.py root@your-hostinger-ip:/root/pump-fun-trading/trading/
```

### 5. Install dependencies on Hostinger
```bash
pip3 install aiohttp
```

### 6. Run the validated engine
```bash
# Test for 1 hour
python3 trading/rentech_validated.py --capital 100 --duration 3600

# Run indefinitely (use screen or tmux)
screen -S rentech
python3 trading/rentech_validated.py --capital 100 --duration 0

# Detach: Ctrl+A, D
# Reattach: screen -r rentech
```

## Run Options

```bash
python3 trading/rentech_validated.py \
    --capital 100 \      # Starting capital in USD
    --duration 3600 \    # Run time in seconds (0 = forever)
    --target 0.05 \      # Take profit at 5%
    --stop 0.05 \        # Stop loss at 5%
    --max-hold 300 \     # Max hold 5 minutes
    --max-positions 3    # Max 3 concurrent positions
```

## What It Does

Uses VALIDATED formulas from 269,830 historical trades:

**Entry Signals:**
- PF-520: Mean Reversion (82.5% win rate) - Buy after 30%+ drops
- PF-530: Buy Pressure (52.8% win rate) - Buy when >70% are buys

**Exit Signals:**
- PF-511: Volume Dry-Up (62.4% win rate) - Exit when volume dies

**Removed (INVALIDATED):**
- Volume Spike: 9% win rate - FAILS
- Whale Following: -22.8% return - LOSES MONEY

## Expected Output

```
============================================================
RENTECH VALIDATED ENGINE v1.0
============================================================
Capital: $100.00
Target: +5% / Stop: -5%
Max Hold: 300s | Max Positions: 3
------------------------------------------------------------
VALIDATED ENTRY SIGNALS:
  PF-520: Mean Reversion - 82.5% win rate
  PF-530: Buy Pressure   - 52.8% win rate
VALIDATED EXIT SIGNALS:
  PF-511: Volume Dry-Up  - 62.4% win rate
============================================================
Connected to PumpPortal WebSocket

>> ENTRY abc12345... | PF520_MEAN_REVERSION | Price: 0.0000001234 | Size: 10.0000 SOL | Curve: 25%
<< EXIT WIN abc12345... | TARGET | PnL: +5.12% (+0.5120 SOL) | Hold: 45s | WR: 100.0%
```

## Monitoring

```bash
# Check if running
ps aux | grep rentech

# View logs (if using screen)
screen -r rentech

# Kill if needed
pkill -f rentech_validated
```
