# STRICT AUTOMATION RULES FOR RAPID V8 OPTIMIZER

## 🔄 CONTEXT REFRESH SYSTEM

Every 10 iterations, the system will:
1. Save all state
2. Exit RAPID_V8_OPTIMIZER.py
3. Clear context (prevents buildup and errors)
4. Auto-restart with fresh context
5. Resume from saved state

This happens automatically. Your job continues seamlessly.

**Why?** Long context causes errors. Fresh context = reliable operation.

## CRITICAL: What You MUST DO

### 1. EDIT IN-PLACE ONLY
✓ **ALWAYS** edit `officialtesting/V8_PROGRESSIVE.py` directly
✓ **NEVER** create new versions like V9, V10, V11, etc.
✓ **NEVER** create V8_PROGRESSIVE_v2.py or similar
✓ Only modify the existing V8_PROGRESSIVE.py file

### 2. ERROR TRACKING
✓ **ALWAYS** check `ERROR_HISTORY.json` before making changes
✓ **NEVER** repeat the same error twice
✓ **ALWAYS** log what you tried and what failed
✓ Learn from previous iteration failures

### 3. NO EXTRA FILES
✓ **NEVER** create new markdown files
✓ **NEVER** create documentation during iteration
✓ **NEVER** create README, NOTES, or similar files
✓ Only edit code, only log to JSON

### 4. ZERO HUMAN INTERVENTION
✓ **ALWAYS** make decisions autonomously
✓ **NEVER** ask for user input during iterations
✓ **ALWAYS** proceed with best judgment
✓ Use error history to guide decisions

### 5. INCREMENTAL CHANGES
✓ **ALWAYS** make 1-3 small changes per iteration
✓ **NEVER** rewrite entire sections
✓ **ALWAYS** preserve existing working code
✓ Only change what's needed

## CRITICAL: What You MUST NOT DO

### ❌ NEVER CREATE NEW FILES
- ❌ No V9_PROGRESSIVE.py
- ❌ No V10_PROGRESSIVE.py
- ❌ No V8_PROGRESSIVE_v2.py
- ❌ No V8_improved.py
- ❌ No V8_backup.py (system handles backups)
- ❌ No new markdown files
- ❌ No new documentation

### ❌ NEVER REPEAT ERRORS
- ❌ Check ERROR_HISTORY.json first
- ❌ If "increased threshold to 0.15 failed", don't try 0.15 again
- ❌ If "removed HMM caused crash", don't remove HMM again
- ❌ If "VPIN calculation error", check what was tried before

### ❌ NEVER ASK HUMANS
- ❌ No "should I try X?"
- ❌ No "which approach?"
- ❌ No "please confirm"
- ❌ Make the decision and proceed

### ❌ NEVER BREAK EXISTING CODE
- ❌ Don't remove imports that are used
- ❌ Don't delete working functions
- ❌ Don't break existing logic flow
- ❌ Only improve, never break

## WORKFLOW FOR EACH ITERATION

### Step 1: Read Error History
```python
# Check what failed before
errors = read_error_history()
avoid_these_changes = errors['failed_attempts']
```

### Step 2: Analyze Current Results
```python
# What's the problem?
if win_rate < 55%:
    problem = "signal quality"
if stop_losses > take_profits:
    problem = "position sizing"
```

### Step 3: Cross-Reference Renaissance
```python
# Check RENAISSANCE_MATH_TOOLKIT.py
# Find formula that addresses the problem
# Make sure it wasn't tried and failed before
```

### Step 4: Edit V8 In-Place
```python
# Edit officialtesting/V8_PROGRESSIVE.py
# Change 1-3 specific lines
# Add comment: # ITER N: Changed X to Y for reason Z
```

### Step 5: Run Test
```python
# Run the modified V8
# Collect results
# If fails, log to ERROR_HISTORY.json
```

### Step 6: Log Results
```python
# Log to iteration_logs/iteration_XXX.json
# Include what changed, why, and results
# Update ERROR_HISTORY.json if failed
```

## ERROR HISTORY FORMAT

```json
{
  "failed_attempts": [
    {
      "iteration": 5,
      "change": "Increased entry_threshold from 0.10 to 0.20",
      "reason": "Win rate dropped to 35%",
      "never_try_again": "threshold > 0.18"
    },
    {
      "iteration": 12,
      "change": "Removed HMM regime detection",
      "reason": "System crashed - HMM required by other components",
      "never_try_again": "removing HMM"
    },
    {
      "iteration": 18,
      "change": "Set fractional_kelly to 1.0",
      "reason": "Excessive risk - lost 90% of capital",
      "never_try_again": "kelly > 0.5"
    }
  ],
  "successful_changes": [
    {
      "iteration": 3,
      "change": "Increased confidence_threshold from 0.15 to 0.18",
      "result": "Win rate improved 58% -> 63%"
    }
  ]
}
```

## DECISION MAKING LOGIC

### If Win Rate < 55%
1. Check ERROR_HISTORY: Have we tried signal filtering before?
2. If not tried: Add more signal filters
3. If tried and failed: Try different filters from Renaissance toolkit
4. Never try same filter twice

### If Stop Losses > Take Profits
1. Check ERROR_HISTORY: Have we adjusted stops before?
2. If not tried: Widen stop loss slightly
3. If tried: Try position sizing reduction instead
4. Never make same adjustment twice

### If Avg Loss > Avg Win
1. Check ERROR_HISTORY: Have we changed TP/SL ratio?
2. If not tried: Adjust take profit higher
3. If tried: Try different entry criteria
4. Track what works

### If No Improvement for 3 Iterations
1. Review ALL error history
2. Try completely different approach
3. Use Renaissance formula not tried yet
4. Be more aggressive with changes

## EXAMPLE ITERATION

### Iteration 15

**1. Read Error History:**
```
- Iteration 8: entry_threshold=0.20 failed (too high)
- Iteration 11: Removed VPIN failed (needed)
- Iteration 13: kelly=0.8 failed (too aggressive)
```

**2. Current Results:**
```
- Win rate: 54% (below 55% target)
- Stop losses: 45, Take profits: 55 (acceptable)
- Problem: Need better signal quality
```

**3. Decision:**
```
- Need: Better signal filtering
- Tried before: entry_threshold increase (failed)
- New approach: Add regime confirmation to signal_combiner
- From Renaissance: Use HMM confidence level
```

**4. Edit V8:**
```python
# Line 410 in V8_PROGRESSIVE.py
# OLD:
if master_signal > entry_threshold:

# NEW: (ITER 15: Added HMM regime confidence check)
regime_confidence = hmm.get_confidence()  # From Renaissance toolkit
if master_signal > entry_threshold and regime_confidence > 0.70:
```

**5. Run & Log:**
```json
{
  "iteration": 15,
  "change": "Added HMM regime confidence check (>0.70) to entry logic",
  "rationale": "Win rate low, need better filtering. entry_threshold adjustment failed before (iter 8), trying different approach",
  "result": {
    "win_rate": 59.2,
    "improved": true
  }
}
```

## AUTONOMOUS DECISION TREE

```
Start Iteration
    ↓
Read ERROR_HISTORY.json
    ↓
Analyze current metrics
    ↓
Is problem known? ───Yes──→ Check if solution tried before
    │                              ↓
    No                        Tried? ─Yes─→ Try different solution
    ↓                              ↓
Identify new problem            No
    ↓                              ↓
Check Renaissance toolkit      Apply solution
    ↓                              ↓
Select best formula            Edit V8 in-place
    ↓                              ↓
Edit V8 in-place              Run test
    ↓                              ↓
Run test                      Success? ─Yes─→ Log success
    ↓                              ↓
Success? ────────────────────────No
    ↓                              ↓
  No ←───────────────────── Log to ERROR_HISTORY
    ↓
Log iteration
    ↓
Next iteration
```

## SAFETY NETS

### Before Every Change
1. ✓ Check ERROR_HISTORY.json
2. ✓ Verify not repeating failed attempt
3. ✓ Confirm V8_PROGRESSIVE.py exists
4. ✓ System will auto-backup before change

### After Every Change
1. ✓ Test runs successfully
2. ✓ Log results to iteration_logs/
3. ✓ Update ERROR_HISTORY.json if failed
4. ✓ Continue to next iteration

### If Anything Fails
1. ✓ Log detailed error to ERROR_HISTORY.json
2. ✓ System will restore backup
3. ✓ Next iteration will try different approach
4. ✓ Never repeat same error

## REMEMBER

**This is PURE AUTOMATION. No human in the loop.**

- Make decisions confidently
- Learn from error history
- Edit V8 in-place only
- Never create new files
- Never repeat errors
- Keep iterating until goal achieved

**Trust the process. The system will guide you.**
