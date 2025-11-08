# GLM-HMM Task Understanding and Corrections

**Date:** November 8, 2025
**Status:** Awaiting user confirmation before proceeding

---

## What I Initially Misunderstood

### My Incorrect Assumptions:
1. **Position → Side mapping was static:**
   - Position 8 = LEFT correct (always)
   - Position 11 = RIGHT correct (always)

2. **One reversal type:**
   - Assumed simple reversal where correct side switches once per task phase

### The Reality (Based on User Clarification):

**LD 1 choice reversal v3 uses a 7/8 rolling criterion:**
- Animal must get **7 out of 8 consecutive correct** responses
- Upon achieving this → **correct side SWITCHES**
- Can switch multiple times within a single session
- **Dynamic reversals** based on performance, not pre-programmed

**Example scenario:**
```
Trials 1-10:   Position 8 (left) is correct
Trial 10:      Achieves 7/8 criterion → REVERSAL
Trials 11-20:  Position 11 (right) is now correct
Trial 20:      Achieves 7/8 again → REVERSAL
Trials 21-30:  Position 8 (left) is correct again
```

---

## The Critical Problem for GLM-HMM

### What GLM-HMM Needs:

For each trial, the model needs to know:
1. **STIMULUS**: Which side is correct (-1 = left, +1 = right)
2. **CHOICE**: Which side the animal chose (0 = left, 1 = right)

### What We Currently Know:

From the data files, we have:
- **Position**: Where stimulus appeared on screen (8 = left location, 11 = right location)
- **Correct**: Whether animal was correct (1) or incorrect (0)
- **Criterion counts**: How many times 7/8 was achieved per session

### What We DON'T Know:

- **Which trial the reversals occurred on** (only session-level counts)
- **Which side was correct for each specific trial**

---

## Proposed Solutions

### Option 1: Infer Correct Side from Animal Performance ⭐ RECOMMENDED

**Logic:**
```python
# For each trial:
if correct == 1:
    # Animal got it right, so whatever they chose was correct
    if position == 8:
        stimulus = -1  # Left was correct
    elif position == 11:
        stimulus = +1  # Right was correct

# Track reversals by detecting 7/8 rolling correct
rolling_window = last_8_trials
if sum(rolling_window == 1) >= 7:
    # Reversal occurred - correct side flips
    current_correct_side *= -1
```

**Advantages:**
- Uses actual trial-by-trial data
- Can detect exact reversal points
- Works for all reversal patterns

**Disadvantages:**
- More complex implementation
- Assumes animal choices reveal correct side (usually true)

### Option 2: Use Only Non-Reversal Sessions First

**Focus on simpler schedules:**
1. `A_Mouse LD 1 choice v2` - No reversals, position 8 only (left correct)
2. `A_Mouse LD Punish Incorrect Training v2` - Training phase
3. Skip reversal sessions initially

**Advantages:**
- Cleaner, unambiguous stimulus coding
- Easier to validate model is working correctly

**Disadvantages:**
- Doesn't analyze the interesting reversal learning data

### Option 3: Request Trial-Level Reversal Markers

**Ask user to provide:**
- File indicating which trial number each reversal occurred on
- OR a column in the data showing current correct side per trial

**Advantages:**
- Most accurate
- No inference needed

**Disadvantages:**
- Requires additional data processing by user

---

## Current Data Status

### Files Examined:

**W LD Data 11.08 All_processed.csv:**
- 13,463 trials from 19 animals
- Includes all schedule types
- Has position data but no reversal markers per trial

**LDR 2025 data1_processed_withSecondCriterion.csv:**
- 192 sessions from 19 animals (reversal sessions only)
- Has `Second_Criterion_Count` (0, 1, or 2 reversals per session)
- **Does NOT have trial-by-trial reversal markers**

### Schedule Distribution:

```
Schedule Name                                    Count
A_Mouse LD 1 choice reversal v3                  192  (Reversal task)
A_Mouse Pairwise Discrimination v3 - Reversal    146  (PD reversal)
A_Mouse LD Punish Incorrect Training v2          119  (LD training)
A_Mouse Pairwise Discrimination v3                89  (PD task)
A_Mouse LD 1 choice v2                            81  (Main LD, no reversal)
A_Mouse LD Must Touch Training v2                 57  (Training)
```

---

## Recommended Approach

### Phase 1: Validate GLM-HMM on Simple Tasks ✓

**Focus on:**
- `A_Mouse LD 1 choice v2` (position 8 only, no reversals)
- `A_Mouse LD Punish Incorrect Training v2`

**Stimulus coding:**
```python
# Simple, unambiguous
if schedule == "LD 1 choice v2":
    stimulus = -1  # Always left correct (position 8)
```

**Purpose:**
- Verify model is working correctly
- Validate psychometric curves make sense
- Check state interpretations

### Phase 2: Add Reversal Analysis with Inference

**Implement rolling 7/8 criterion detector:**
```python
def infer_correct_side_with_reversals(trial_df):
    correct_side = -1  # Start with left
    rolling_window = deque(maxlen=8)

    for trial in trials:
        rolling_window.append(trial['correct'])

        # Check if reversal criterion met
        if len(rolling_window) == 8 and sum(rolling_window) >= 7:
            correct_side *= -1  # FLIP

        trial['stimulus'] = correct_side
```

**Validate against:**
- `Second_Criterion_Count` should match detected reversals
- State transitions should align with reversal points

### Phase 3: Add State Labels to All Plots

**Current states (unnamed):**
- State 1, State 2, State 3

**Proposed labels based on characteristics:**
- **Engaged State**: High accuracy (>60%), responsive
- **Lapse State**: ~Chance accuracy (~50%), disengaged
- **Learning/Transition State**: Intermediate accuracy

**Implementation:**
- Automatically label based on accuracy + latency
- Add labels to all figure titles and legends
- Color-code consistently across all plots

---

## Questions for User

1. **Should I proceed with Phase 1 (simple tasks only) first?**
   - This would give clean results to validate the approach
   - We can add reversals in Phase 2

2. **For reversal inference, should I:**
   - A) Implement the rolling 7/8 detector (Option 1)
   - B) Request trial-level reversal markers from you
   - C) Skip reversals for now

3. **State naming preferences:**
   - Would you like specific names for the states?
   - Or should I auto-label based on behavioral metrics?

4. **Schedule filtering:**
   - Confirm I should use only:
     - `A_Mouse LD 1 choice v2`
     - `A_Mouse LD Punish Incorrect Training v2`
     - `A_Mouse LD 1 choice reversal v3`
   - From W cohort only initially?

---

## Next Steps (Pending Your Approval)

1. ✅ Fix stimulus coding for simple LD tasks (no reversals)
2. ✅ Filter to specified schedules only
3. ⏸️ Implement reversal inference (pending your choice)
4. ⏸️ Add state labels to all plots
5. ⏸️ Re-run analysis with corrected approach
6. ⏸️ Generate comprehensive documentation

---

## Technical Summary

### GLM-HMM Inputs (Current):

```python
X = Design Matrix (n_trials × 5):
    [1] stimulus_correct_side:    -1 (left) or +1 (right) ← NEEDS FIX FOR REVERSALS
    [2] bias:                     1 (constant)
    [3] prev_choice:              -1 (left) or +1 (right)
    [4] wsls:                     prev_choice × prev_correct
    [5] session_progression:      0.0 to 1.0

y = Choices (n_trials):
    0 = chose LEFT
    1 = chose RIGHT
```

### Current Limitation:

**Stimulus feature assumes static position→side mapping**, which is incorrect for reversal sessions with dynamic 7/8 criterion.

**Impact:**
- Reversal sessions will have incorrect stimulus coding
- Model will learn wrong associations
- State interpretations may be biased

**Solution:**
Start with non-reversal sessions, then add sophisticated reversal inference.

---

**Awaiting your guidance on how to proceed!**
