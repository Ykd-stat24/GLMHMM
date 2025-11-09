# GLM-HMM Implementation Plan - Phase 1 & 2

**Date:** November 9, 2025
**Status:** Ready to implement based on user feedback

---

## 📊 Cohort Structure

### W Cohort (19 animals)
- **Genotypes:** 2 (+, -)
  - WT (+): 9 animals
  - KO (-): 10 animals
- **Sex:** 9 Male, 10 Female
- **Reversal task:** `A_Mouse LD 1 choice reversal v3` (rolling 7/8 criterion)
- **Special:** Some animals achieve 2 reversals per session

### F Cohort (36 animals)
- **Genotypes:** 4 (+, +/+, +/-, -/-)
  - + (WT): 13 animals
  - +/+: 8 animals
  - +/-: 7 animals
  - -/-: 7 animals
- **Sex:** 18 Male, 17 Female
- **Reversal task:** `A_Mouse LD 1 Reversal 9 task` (single reversal)
- **Special:** Different reversal schedule than W cohort

---

## 🎯 Enhanced Feature Design

### Current Features (Baseline):
1. **stimulus_correct_side**: Which side is correct (-1=left, +1=right)
2. **bias**: Constant term (1.0)
3. **prev_choice**: Previous choice (-1=left, +1=right)
4. **wsls**: Win-stay/lose-shift (prev_choice × prev_correct)
5. **session_progression**: Progress within session (0 to 1)

### NEW Features to Add:

#### For ALL Tasks:
6. **recent_side_bias**: Proportion of RIGHT choices in last 10 trials
   ```python
   recent_side_bias = sum(last_10_choices == 'right') / 10
   # Range: 0.0 (all left) to 1.0 (all right)
   ```

7. **task_stage**: Numerical encoding of training progression
   ```python
   task_stage_map = {
       'LD Initial Touch': 0,
       'LD Must Touch': 1,
       'LD Punish Incorrect': 2,
       'LD 1 choice v2': 3,
       'LD 1 choice reversal v3': 4,  # or LD 1 Reversal 9
       'Pairwise Must Touch': 5,
       'Pairwise Punish Incorrect': 6,
       'Pairwise Discrimination v3': 7,
       'Pairwise Discrimination v3 - Reversal': 8
   }
   # Normalized to 0-1 range
   ```

8. **cumulative_trials**: Total trials experienced (captures overall training)
   ```python
   cumulative_trials = trial_count_so_far / total_trials_in_dataset
   ```

#### For Reversal Tasks ONLY:
9. **trials_since_reversal**: Trials since last reversal occurred
   ```python
   trials_since_reversal = trial_num - last_reversal_trial_num
   # Helps capture adaptation dynamics
   # Normalized to 0-1 based on max trials
   ```

#### For PD Tasks ONLY:
10. **stimulus_identity**: Which image is S+ (0 or 1, binary feature)
    ```python
    # Track which specific image is rewarded
    # Important because PD has 2 different images unlike LD
    ```

---

## 🔬 State Labeling Strategy (Process-Focused)

Based on user preference for **Option B**:

### Automatic State Classification Algorithm:

```python
def classify_state(accuracy, cv_latency, mean_latency, dwell_time):
    """
    Classify behavioral state based on metrics.

    Returns one of:
    - "Deliberative High-Performance"
    - "Deliberative Moderate-Performance"
    - "Disengaged Lapse"
    - "Procedural Fast-Performance" (low CV, fast latency)
    """

    # Lapse: near-chance accuracy
    if accuracy < 0.55:
        return "Disengaged Lapse"

    # High deliberation (CV > 0.6)
    if cv_latency > 0.6:
        if accuracy > 0.65:
            return "Deliberative High-Performance"
        else:
            return "Deliberative Moderate-Performance"

    # Low deliberation (procedural/automatic)
    else:
        return "Procedural Fast-Performance"
```

### State Validation:
For each identified state, we'll measure and report:
- Mean accuracy ± SEM
- Mean latency ± SEM
- Latency CV (coefficient of variation)
- Mean dwell time (bout duration)
- Occupancy (% of trials)
- GLM weights (stimulus, bias, prev_choice, wsls)

---

## 🔄 Reversal Detection Logic

### W Cohort (A_Mouse LD 1 choice reversal v3):

**Using rolling 7/8 and 5/6 criteria:**

```python
def detect_reversals_w_cohort(trial_correctness, ldr_criterion_data):
    """
    Detect reversal points for W cohort.

    Uses:
    - First reversal: 7/8 correct in 8-trial window
    - Second reversal: 5/6 correct in 6-trial blocks (after first)
    """

    # Get criterion trial numbers from LDR file
    first_reversal_trial = ldr_criterion_data['No trials to criterion - Generic Evaluation (1)']
    second_reversal_trial = ldr_criterion_data['No trials to criterion - Generic Evaluation (2)']

    # Initialize: position 8 (left) is correct
    correct_side = -1  # Left
    stimulus_values = []

    for trial_num in range(1, 31):
        # Check if reversal occurred at this trial
        if trial_num == first_reversal_trial:
            correct_side = +1  # FLIP to right (position 11)
        elif not pd.isna(second_reversal_trial) and trial_num == second_reversal_trial:
            correct_side = -1  # FLIP back to left (position 8)

        stimulus_values.append(correct_side)

    return stimulus_values
```

### F Cohort (A_Mouse LD 1 Reversal 9):

**Simpler - usually single reversal:**

```python
def detect_reversals_f_cohort(ldr_criterion_data):
    """
    Detect reversal points for F cohort.

    Usually single reversal from position 8 → 11.
    """

    first_reversal_trial = ldr_criterion_data['No trials to criterion - Generic Evaluation (1)']

    correct_side = -1  # Start with left (position 8)
    stimulus_values = []

    for trial_num in range(1, 31):
        if trial_num == first_reversal_trial:
            correct_side = +1  # FLIP to right (position 11)

        stimulus_values.append(correct_side)

    return stimulus_values
```

---

## 📈 Analysis Pipeline

### Phase 1: Non-Reversal Tasks

**Tasks to analyze:**
1. `A_Mouse LD 1 choice v2` (position 8 only, no reversals)
2. `A_Mouse LD Punish Incorrect Training v2`
3. `A_Mouse Pairwise Discrimination v3` (no reversal)

**Separate analyses for:**
- W cohort (19 animals)
- F cohort (36 animals)

**Outputs:**
- State identification and characterization
- State-specific psychometric curves
- GLM weight interpretations
- Learning curves by genotype and sex

### Phase 2: Reversal Tasks

**Tasks to analyze:**
1. `A_Mouse LD 1 choice reversal v3` (W cohort - rolling criterion)
2. `A_Mouse LD 1 Reversal 9` (F cohort - single reversal)

**Additional analyses:**
- State transitions around reversal points
- Adaptation speed (trials to criterion after reversal)
- Reversal-specific state occupancy
- Genotype effects on reversal learning

---

## 📊 Genotype Comparison Strategy

### W Cohort (2 genotypes):

**Simple WT vs KO comparison:**
```python
genotype_comparison = {
    'WT (+)': 9 animals,
    'KO (-)': 10 animals
}

# Statistical tests:
- Final accuracy: t-test
- State occupancy: ANOVA
- Reversal speed: t-test (trials to criterion)
- Learning rate: linear mixed model
```

### F Cohort (4 genotypes):

**Multi-level comparison:**
```python
genotype_comparison = {
    'WT (+)': 13 animals,
    'Homozygous WT (+/+)': 8 animals,
    'Heterozygous (+/-)': 7 animals,
    'Homozygous KO (-/-)': 7 animals
}

# Statistical tests:
- Overall genotype effect: One-way ANOVA
- Post-hoc pairwise: Tukey HSD
- Dose-response: Linear trend test (+/+ → +/- → -/-)
```

### Cross-Cohort Comparison:

**Collapse F genotypes to match W:**
```python
# For W vs F comparison:
F_collapsed = {
    'WT': ['+', '+/+'],      # Combine
    'KO': ['+/-', '-/-']      # Combine hetero + homo KO
}

# Then compare W_WT vs F_WT and W_KO vs F_KO
```

---

## 🎨 Figure Generation Plan

### Core Figures (Per Cohort):
1. GLM-HMM summary (weights, transitions, states)
2. State-specific psychometric curves
3. State occupancy over sessions
4. Learning curves by genotype
5. State transition heatmap

### Genotype Comparison Figures:
6. Final accuracy by genotype (bar plot with error bars)
7. State occupancy by genotype (stacked bar)
8. Learning curves overlaid by genotype
9. Latency distributions by genotype

### Sex Comparison Figures:
10. Learning curves by sex
11. State occupancy by sex
12. Sex × genotype interaction

### Reversal-Specific Figures:
13. State transitions around reversal points
14. Trials to criterion by genotype
15. Adaptation curves (post-reversal learning)

### Cross-Cohort Comparison:
16. W vs F learning curves
17. W vs F state characterization
18. W vs F reversal performance

**Total: ~25-30 figures per cohort = 50-60 figures total**

---

## 🔧 Implementation Steps

### Step 1: Update Feature Engineering
- [x] Add `recent_side_bias` feature
- [ ] Add `task_stage` feature
- [ ] Add `cumulative_trials` feature
- [ ] Add `trials_since_reversal` for reversal tasks
- [ ] Add `stimulus_identity` for PD tasks

### Step 2: Implement Reversal Detection
- [ ] Create `detect_reversals_w_cohort()` function
- [ ] Create `detect_reversals_f_cohort()` function
- [ ] Integrate with LDR criterion file
- [ ] Validate against `Second_Criterion_Count`

### Step 3: Implement State Labeling
- [ ] Create `classify_state()` function
- [ ] Add state validation metrics
- [ ] Auto-label all states in plots

### Step 4: Run Phase 1 Analysis
- [ ] W cohort - non-reversal tasks
- [ ] F cohort - non-reversal tasks
- [ ] Generate all Phase 1 figures
- [ ] Statistical comparisons (genotype, sex)

### Step 5: Run Phase 2 Analysis
- [ ] W cohort - reversal tasks
- [ ] F cohort - reversal tasks
- [ ] Generate all Phase 2 figures
- [ ] Reversal-specific analyses

### Step 6: Cross-Cohort Analysis
- [ ] W vs F comparison figures
- [ ] Combined statistical tests
- [ ] Final summary document

---

## 📝 Deliverables

1. **Code:**
   - Updated `glmhmm_utils.py` with new features
   - New `reversal_detection.py` module
   - Updated `run_complete_analysis.py`

2. **Figures:**
   - 25-30 figures per cohort
   - Organized by analysis type
   - Publication-ready (300 DPI)

3. **Documentation:**
   - `RESULTS_W_COHORT.md`
   - `RESULTS_F_COHORT.md`
   - `CROSS_COHORT_COMPARISON.md`
   - Statistical test summary tables (CSV)

4. **Summary:**
   - Key findings by genotype
   - State characterizations
   - Reversal learning metrics
   - Recommendations for follow-up

---

## ❓ Questions Answered

✅ **Q1: PD side bias?**
**A:** Yes, adding `recent_side_bias` feature (proportion right choices in last 10 trials)

✅ **Q2: State labels?**
**A:** Option B - Process-focused ("Deliberative High-Performance", etc.)

✅ **Q3: Reversal logic?**
**A:** Confirmed correct

✅ **Q4: W vs F differences?**
**A:** Both start position 8 correct, but different reversal task schedules

✅ **Q5: What is stimulus?**
**A:** Stimulus = which side is CORRECT (-1=left, +1=right)

✅ **Q6: Training effects?**
**A:** Adding `task_stage` and `cumulative_trials` features

✅ **Q7: Genotype comparisons?**
**A:** Multi-level for F (4 genotypes), simple for W (2 genotypes), collapsed for cross-cohort

---

**Ready to implement! 🚀**
