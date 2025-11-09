# GLM-HMM Analysis Status and Poster Recommendations

## Current Status

### ✅ Phase 1 Analysis (COMPLETE)
**Non-reversal tasks: LD, PI, PD**

#### Completed Analyses:
1. **Individual animal GLM-HMM fits** (54 animals total)
   - Cohort W: 19 animals (genotypes: +, -)
   - Cohort F: 35 animals (genotypes: +, +/+, +/-, -/-)
   - 3 states per animal
   - 7 behavioral features (excludes stimulus)

2. **Within-cohort genotype comparisons:**
   - GLM weight distributions by genotype
   - State metrics heatmaps (hierarchically clustered)
   - Behavioral space PCA
   - Population transition matrices
   - State occupancy distributions
   - State label prevalence

3. **Cross-cohort comparisons:**
   - WT comparison (W+ vs F+/+)
   - Statistical testing across key metrics

4. **Lapse state analysis:**
   - Lapse characteristics by genotype
   - Accuracy, occupancy, dwell time comparisons

#### Key Findings (Phase 1):
- **F cohort -/- genotype** shows significantly impaired performance:
  - State accuracies: 35.6%, 29.1%, 59.4% (vs 58-67% for other genotypes)
  - Higher side bias: 62.2%, 71.3% (vs 38-47% for others)

- **F cohort +/- genotype** shows longer state dwell times:
  - 19.6, 45.4, 43.0 trials (vs 14-38 for others)

- **State heterogeneity**: Not all animals show clear "engaged" vs "lapse" states
  - ~40-48% of states are "undefined" (don't meet strict classification criteria)

#### Available for Poster:
- 23 high-quality figures (PNG + PDF)
- 2 summary statistics CSV tables
- Clear genotype-specific patterns

---

### ⚠️ Phase 2 Analysis (IN PROGRESS - DATA ISSUES)
**Reversal tasks: LD reversal**

#### Current Limitations:
1. **Data preprocessing incomplete**:
   - Reversal detection functions reference missing utilities
   - Trial-level reversal data needs proper formatting
   - Genotype information needs to be merged with reversal trials

2. **Available but not yet processed**:
   - `F_LD_LDR_triallevel.csv` (17,437 trials)
   - `LDR 2025 data1_processed_withSecondCriterion.csv`
   - Need to merge with genotype/animal metadata

#### What CAN be done (with additional data prep):
1. State transitions at reversal points
2. Psychometric curves by state
3. Reversal adaptation speed by genotype
4. Pre/post-reversal state changes

---

## Recommendations for Your Poster

### Option 1: Focus on Phase 1 (RECOMMENDED for immediate poster)
**Advantages:**
- All data processed and validated
- Clear genotype effects
- Publication-quality figures ready
- Strong statistical comparisons

**Suggested Poster Sections:**
1. **Overview**: GLM-HMM identifies 3 behavioral states during learning
2. **Genotype Effects**: -/- shows impaired performance, +/- shows longer engagement
3. **State Characteristics**: Heterogeneity in lapse vs deliberative strategies
4. **Cross-Cohort Validation**: WT animals show consistent patterns

### Option 2: Phase 1 + Phase 2 Framework
**Requires:**
- 1-2 days additional data preprocessing
- Proper reversal trial identification and labeling
- Merging genotype information

**Would add:**
- Reversal-specific state analysis
- Adaptation speed comparisons
- State flexibility metrics

---

## Addressing Your Specific Questions

### 1. "State transitions at reversal points"
**Status**: Requires Phase 2 data preprocessing
**Alternative**: We have state transition matrices for Phase 1 by genotype

### 2. "Psychometric curves for state-specific stimulus following"
**Status**: Requires Phase 2 (stimulus varies in reversals)
**Alternative**: In Phase 1, stimulus was excluded (no variance)

### 3. "Genotype × state × time/training analysis"
**Status**: ✅ Available in Phase 1
**Files**:
- `cohort_X_summary_statistics.csv`
- `cohort_X_behavioral_pca.png`
- Session progression is a feature in the model

### 4. "Phase 1 vs Phase 2 state label comparison"
**Status**: Requires Phase 2 completion
**Can show**: Deliberative vs procedural prevalence in Phase 1 by genotype

### 5. "Lapse differences between phases and cohorts"
**Status**: ✅ Phase 1 lapse analysis complete
**Files**: `lapse_state_analysis.png`

### 6. "Reversal-specific state analysis (pre/post/adaptation)"
**Status**: Requires Phase 2 data
**Timeline**: 1-2 days with proper data formatting

---

## Immediate Next Steps

### For Poster (This Week):
1. ✅ Use all Phase 1 visualizations
2. ✅ Highlight genotype-specific findings
3. ✅ Emphasize state heterogeneity
4. Add text describing:
   - Clear -/- impairment
   - +/- engagement pattern
   - Individual variability

### For Phase 2 (After Poster):
1. Format reversal trial data:
   - Merge genotype information
   - Identify reversal points
   - Create proper trial-level dataframe

2. Run Phase 2 GLM-HMM analysis
3. Generate Phase 1 vs Phase 2 comparisons

---

## Questions for You

1. **Poster deadline**: How soon do you need final figures?

2. **Phase 2 priority**: Is Phase 2 analysis critical for this poster, or can it be follow-up work?

3. **Data access**: Do you have properly formatted reversal trial data with:
   - Animal IDs
   - Genotypes
   - Reversal point markers
   - Trial-level outcomes

4. **State stringency**: You mentioned potentially adjusting "stringency of procedural high performance" - what changes would you like?

---

## Current Deliverables

### Figures (all in `results/phase1_non_reversal/summary_figures/`):
1. `cohort_W_weight_distributions.png/pdf` - Feature weighting by genotype
2. `cohort_W_metrics_heatmap.png/pdf` - Animal clustering
3. `cohort_W_behavioral_pca.png/pdf` - Behavioral space
4. `cohort_W_transition_matrices.png/pdf` - State dynamics
5. `cohort_W_occupancy_distributions.png/pdf` - State usage
6-10. Same for Cohort F
11. `cross_cohort_WT_comparison.png/pdf` - Cross-cohort validation
12. `state_label_distribution_by_genotype.png/pdf` - Label prevalence
13. `lapse_state_analysis.png/pdf` - Lapse characteristics

### Data Tables:
1. `cohort_W_summary_statistics.csv` - All metrics by genotype
2. `cohort_F_summary_statistics.csv` - All metrics by genotype

---

## Bottom Line

**For your poster RIGHT NOW**: You have excellent Phase 1 data showing clear genotype effects on behavioral states during initial learning. The analyses are complete, statistically sound, and publication-ready.

**For Phase 2**: We need to resolve data formatting issues first. This is doable but requires 1-2 days of focused data preprocessing work.

**My recommendation**: Use the comprehensive Phase 1 analyses for your poster, then develop Phase 2 as follow-up work for the manuscript.

What would you like me to prioritize?
