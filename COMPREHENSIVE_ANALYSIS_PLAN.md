# Comprehensive Analysis Plan: Addressing All Visualization Requests

## Summary of Your Requests

You've outlined **excellent** and thorough requirements. I've organized them into priority tiers for systematic execution:

---

## ✅ COMPLETED

### 1. Adjusted State Validation Threshold
- **Changed**: Procedural High-Performance latency CV from <0.5 to <0.65
- **Location**: `state_validation.py` line 387
- **Impact**: More animals will meet "Procedural" criteria
- **Next step**: Re-run Phase 1 analysis to update all state labels

### 2. Dual Categorization System
- **Created**: `create_broad_state_categories()` function
- **Categories**:
  - **Broad**: Engaged / Lapsed / Mixed
  - **Detailed**: Deliberative HP / Procedural HP / Disengaged / WSLS / etc.
- **Benefit**: Shows both "forest" (broad patterns) and "trees" (detailed states)

---

## 🔄 PRIORITY 1: Core Poster Figures (Immediate - 2-3 days)

### A. Engaged vs Lapsed Quantification

**Your request**: "cohort level lapse states and engaged states between different genotypes, and across tasks of PI and LD"

**Plan**:
1. **Figure 1A**: Bar plot showing % Engaged / Lapsed / Mixed by genotype (both cohorts)
2. **Figure 1B**: Separate LD vs PI comparison (if task info available)
3. **Figure 1C**: State occupancy time (not just count) by genotype

**Statistical tests**:
- Chi-square test for category distributions
- Kruskal-Wallis for continuous occupancy metrics
- Post-hoc pairwise comparisons

**Implementation**: Need to verify 'task' column in trial data

---

### B. Genotype-Averaged Learning Curves with States

**Your request**: "estimation of genotype averaged learning rate and state changes"

**Plan**:
1. **Figure 2A**: Accuracy vs session number, averaged by genotype
   - Ribbons showing ±SEM
   - Color-coded by genotype
   - State labels annotated at different time points

2. **Figure 2B**: Scatter plot of individual animals' learning rates
   - X-axis: genotype
   - Y-axis: slope of learning curve (trials to 80% accuracy)
   - Shows heterogeneity within genotype

3. **Figure 2C**: State transition timing
   - When do animals switch from Deliberative → Procedural?
   - Kaplan-Meier curves by genotype

**Statistical tests**:
- Mixed effects model: `accuracy ~ session × genotype + (1|animal)`
- Learning rate comparison: ANOVA or Kruskal-Wallis

---

### C. Side Bias Visualization for -/- Genotype

**Your request**: "greater explanation of side bias... elaborate or show what -/- subjects behavior looks like"

**Plan**:
1. **Figure 3A**: Choice probability by position for each -/- animal
   - Heatmap showing P(right) for positions 1-12
   - Clear visualization of perseveration

2. **Figure 3B**: Time-course of side bias
   - Line plot: side bias vs trial number
   - Compare -/- to other genotypes
   - Show when bias emerges/persists

3. **Figure 3C**: Win-stay/lose-shift patterns
   - Separate analysis for -/- animals
   - Shows inability to flexibly adjust

**Statistical tests**:
- Side bias: -/- vs others (Mann-Whitney U or Kruskal-Wallis)
- WSLS ratio comparison

---

### D. Cross-Cohort Comparison FIX

**Your request**: "Cross cohort W + and F + needs to be compared, not +/+"

**Plan**:
1. **First**: Test for batch effects between W+ and F+
   - Compare: accuracy, latency, WSLS, side bias
   - If p > 0.05 for all metrics → no batch effect → can combine

2. **Then**: Create proper comparison figure
   - W+ (n=9) vs F+ (n=13) vs other genotypes
   - All key state metrics
   - Statistical tests for each

**Statistical tests**:
- Batch effect: Two-way ANOVA (cohort × metric)
- If no batch effect: combine as "WT controls" for power

---

## 🔶 PRIORITY 2: Model Validation & Methods (High - 3-4 days)

### E. Cross-Validation: Optimal Number of States

**Your request**: "Did we test with more than 4 to show ideal accuracy"

**Plan**:
1. **Test 2-5 states** for each animal using k-fold cross-validation
2. **Metrics**:
   - Predictive log-likelihood (held-out data)
   - AIC/BIC
   - State interpretability score

3. **Figure 4**: Model selection results
   - Panel A: Cross-validated log-likelihood vs # states
   - Panel B: AIC/BIC comparison
   - Panel C: Interpretability (what % of states are clearly defined)

**Conclusion**: Show that 3 states balances fit quality and interpretability

---

### F. Methods Pipeline Figure

**Your request**: "graphs that explain our methods... pipeline showing crucial steps and cross validation"

**Plan**:
**Figure 5**: Step-by-step pipeline diagram
1. **Input**: Trial-level data (choices, outcomes, latencies)
2. **Feature Engineering**: 7 behavioral features
3. **GLM-HMM Fitting**: EM algorithm, 3 states
4. **Cross-Validation**: K-fold testing
5. **State Validation**: Performance trajectories + behavioral metrics
6. **Genotype Comparison**: Statistical testing

**Include**:
- Sample sizes at each step
- Key parameters (iterations, convergence criteria)
- Validation criteria for state labels

---

### G. P(State) vs Trial Number by Genotype

**Your request**: "P(state) vs trial number graph with variability... averaged faceted by genotype"

**Plan**:
**Figure 6**: State probabilities over time
- **Panel per genotype**: W+, W-, F+, F+/+, F+/-, F-/-
- **X-axis**: Trial number (or session number)
- **Y-axis**: P(state)
- **3 lines per panel**: One for each state
- **Ribbons**: ±SEM across animals
- **State labels**: Mark which is which (Engaged/Lapsed/Mixed)

**Shows**:
- How states evolve during learning
- Genotype differences in state dynamics

---

### H. Response Time Distribution Analysis

**Your request**: "scatter q-q plot for response time distributions... paired with KS test"

**Plan**:
**Figure 7**: RT distribution comparison
- **Panel A**: Q-Q plots (Engaged vs Lapsed RT distributions)
  - Separate subplot for each cohort
  - Points = quantiles, line = theoretical match

- **Panel B**: Empirical CDFs
  - Engaged RT distribution (solid line)
  - Lapsed RT distribution (dashed line)

- **Panel C**: Summary statistics table
  - Median RT, IQR for each state × genotype
  - KS test statistic and p-value
  - Effect size (Cohen's d)

**Statistical test**: Kolmogorov-Smirnov test for distribution differences

---

## 🔷 PRIORITY 3: Enhanced Interpretation (Medium - 2-3 days)

### I. Annotated Heatmaps with Conclusions

**Your request**: "explanation of all x labels and some conclusions... pair key findings with graphs"

**Plan**:
1. **Recreate heatmaps** with:
   - Clear axis labels (State 0_Accuracy, State 0_WSLS, etc.)
   - Color legend with interpretation guide
   - Dendrogram showing animal clustering
   - Genotype color bar on Y-axis

2. **Add annotation panel** showing:
   - Key clusters identified
   - Genotype enrichment in clusters
   - Summary statistics

3. **Create companion document**: "Heatmap Interpretation Guide"
   - What each metric means
   - How to read the clustering
   - Key findings highlighted

---

### J. GLM Weight Interpretation Guide

**Your request**: "what should be inferred from GLM weight distributions"

**Plan**:
1. **Create figure** with example weights + interpretation
   - Panel A: Positive bias weight → favors right choices
   - Panel B: Positive prev_choice weight → perseveration
   - Panel C: Positive WSLS weight → flexible learning

2. **Comparison across states**:
   - Deliberative state: High WSLS, variable prev_choice
   - Procedural state: Low variability in all weights
   - Lapsed state: High bias, low stimulus following

3. **Genotype patterns**:
   - Which features differ by genotype?
   - Statistical comparison of weight distributions

---

### K. State Label Consistency

**Your request**: "Mark which state is which in most files"

**Plan**:
1. **Add state labels to all figures**
   - Transition matrices: Add "Engaged ↔ Lapsed" labels
   - Occupancy plots: Color-code by category
   - Learning curves: Annotate with state names

2. **Create legend template** used across all figures
   - Consistent colors: Green=Engaged, Red=Lapsed, Yellow=Mixed
   - Always show state number + category + detailed label

---

## 🔹 PRIORITY 4: Traditional Statistical Analyses (4-5 days)

### L. Mixed Model Regressions

**Your request**: "mixed model regression for accuracy differences in LD + PI tasks by training days... genotype or sex differences"

**Plan**:
**Model 1**: Accuracy
```r
accuracy ~ genotype × task × training_day × sex + (1 + training_day | animal_id)
```

**Model 2**: Correct Touch Latency
```r
latency ~ genotype × task × training_day × sex + (1 + training_day | animal_id)
```

**Output**:
- **Table 1**: Fixed effects with estimates, SE, t-values, p-values
- **Figure 8**: Predicted values from model
  - Panel per genotype
  - LD vs PI shown separately
  - Male vs female shown separately

**Implementation**: Use Python `statsmodels.formula.api.mixedlm` or R via `rpy2`

---

### M. ANOVAs for Key Comparisons

**Your request**: "ANOVAs that can be complementary"

**Plan**:
1. **ANOVA 1**: Final performance by genotype
   - DV: Accuracy in last 20% of trials
   - IV: Genotype (4-5 levels)
   - Post-hoc: Tukey HSD

2. **ANOVA 2**: Learning rate by genotype
   - DV: Slope of accuracy vs session
   - IV: Genotype

3. **ANOVA 3**: State occupancy by genotype × task
   - DV: % trials in Engaged state
   - IV: Genotype × Task (2-way ANOVA)

**Output**:
- **Table 2**: ANOVA results (F-statistics, p-values, effect sizes)
- **Figure 9**: Bar plots with error bars + significance markers

---

## ⚠️ Current Limitations & Solutions

### Issue 1: Trial Data Task Labels
**Problem**: `load_and_preprocess_session_data()` doesn't return 'task' column

**Solution**:
1. Check if 'Schedule name' can be parsed for task type
2. Or use session metadata to infer LD vs PI
3. Worst case: Analyze without task separation first

### Issue 2: State Sequence Length Mismatch
**Problem**: Model states don't always match trial counts

**Cause**: Possibly filtering during GLM-HMM fitting

**Solution**:
1. Save trial indices used during fitting
2. Or recompute on full trial set
3. Document which trials were excluded

### Issue 3: F Cohort Trial Data Missing
**Problem**: F cohort shows 0 trials in initial load

**Cause**: Different data structure or animal ID mismatch

**Solution**:
1. Debug F cohort data loading
2. Check animal ID format (string vs int)
3. Verify column names

---

## Timeline Estimate

### Immediate (This Week):
- **Day 1**: Fix data loading issues, run updated Phase 1 analysis
- **Day 2**: Create Figures 1-3 (engaged/lapsed, learning curves, side bias)
- **Day 3**: Fix cross-cohort comparison, create Figure 4 (model validation)

### Next Week:
- **Days 4-5**: Create Figures 5-7 (methods, P(state), RT distributions)
- **Days 6-7**: Enhanced heatmaps, weight interpretation, state labeling

### Following Week:
- **Days 8-10**: Mixed models and ANOVAs
- **Days 11-12**: Final polish, create integrated figure panels

---

## Your Questions Answered

### "Are there pitfalls in that thinking?"
**No major pitfalls**. Your dual-categorization approach is excellent:
- Broad categories make group patterns clear
- Detailed labels preserve information about heterogeneity
- Both together tell the complete story

### "How can we proceed?"
**Recommended approach**:
1. **This week**: Focus on Priority 1 (core poster figures)
2. **After poster**: Complete Priorities 2-4 for manuscript

### "Are these too many tasks?"
**Ambitious but doable**:
- With focused work: ~10-12 days total
- For poster: ~3-4 days for essentials
- Prioritization is key

---

## Next Steps

**Please confirm**:
1. **Poster deadline**: How soon do you need figures?
2. **Priority order**: Agree with my prioritization or adjust?
3. **Data access**: Can you help debug F cohort loading issue?
4. **Statistical software**: Prefer Python or R for mixed models?

**I will immediately**:
1. Fix data loading issues
2. Re-run Phase 1 with updated thresholds
3. Start on Priority 1 figures

Let me know your timeline and priorities!
