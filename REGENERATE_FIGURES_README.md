# How to Regenerate All Figures with New Labels

## Quick Start

### 1. Pull the latest code
```bash
git pull
```

### 2. Run the regeneration script
```bash
python regenerate_all_figures_locally.py
```

That's it! The script will regenerate all Phase 1 and Phase 2 figures with the new genotype labels and colors.

---

## What Gets Regenerated

### New Genotype Labels & Colors
- `+/+` → **A1D_Wt** (yellow/gold)
- `+/-` → **A1D_Het** (blue)
- `-/-` → **A1D_KO** (maroon)
- `+` → **B6** (red)
- `-` → **C3H x B6** (black)

### State Labels (Improved)
- **State 0** → **Engaged** (high accuracy) - Green
- **State 1** → **Biased** (side preference) - Orange
- **State 2** → **Lapsed** (disengaged) - Red

---

## Output Location

All regenerated figures are saved to: `results/regenerated_figures/`

### Phase 1 Figures (`phase1/`)
1. **fig1_state_occupancy** - % time in each state by genotype
2. **fig2_glm_weights** - GLM weights for all features by state and genotype
3. **fig3_transition_matrices** - State transition probabilities (F cohort only)
4. **fig4_state_characteristics** - Accuracy, latency, and variability by state

### Phase 2 Figures (`phase2/`)
1. **fig1_state_occupancy_reversal** - State occupancy during reversal learning
2. **fig2_reversal_performance** - Overall reversal learning performance

### Combined Figures (`combined/`)
1. **fig1_cross_cohort_comparison** - Cross-cohort and genotype comparisons

**Total**: 7 main figures × 2 formats = **14 files** (PNG + PDF)

---

## File Formats

Each figure is saved in two formats:
- **PNG** (300 DPI) - High-resolution for presentations
- **PDF** (vector) - Publication-ready, scalable

---

## Runtime

- **Phase 1 figures**: ~5-10 seconds
- **Phase 2 figures**: ~3-5 seconds
- **Combined figures**: ~2-3 seconds
- **Total**: ~15-20 seconds

---

## Troubleshooting

### "Module not found" error
```bash
# Make sure you're in the GLMHMM directory
cd path/to/GLMHMM
python regenerate_all_figures_locally.py
```

### "No module named genotype_labels"
```bash
# Ensure genotype_labels.py is in the same directory
ls genotype_labels.py  # Should exist
```

### Phase 2 data missing
If you see "No Phase 2 data available", this is normal if you haven't run Phase 2 analyses yet. The script will:
- Generate all Phase 1 figures (4 figures)
- Skip Phase 2 figures
- Generate combined figures (1 figure)

### Old figures still showing
The script creates NEW figures in `results/regenerated_figures/`. Old figures remain in:
- `results/phase1_non_reversal/`
- `results/phase2_reversal/`
- `results/updated_figures/`

You can safely delete old figure directories if you want to clean up.

---

## Customization

### Change Colors

Edit `genotype_labels.py` and modify the `GENOTYPE_COLORS` dictionary:

```python
GENOTYPE_COLORS = {
    'B6': '#e74c3c',           # Red
    'C3H x B6': '#000000',     # Black
    'A1D_Wt': '#FFD700',       # Yellow
    'A1D_Het': '#4169E1',      # Blue
    'A1D_KO': '#800000'        # Maroon
}
```

Then rerun the script.

### Change State Colors

Edit `genotype_labels.py` and modify the `STATE_COLORS` dictionary:

```python
STATE_COLORS = {
    0: '#2ecc71',  # Green - Engaged
    1: '#f39c12',  # Orange - Biased
    2: '#e74c3c',  # Red - Lapsed
}
```

---

## Alternative: Regenerate Individual Figures

If you only want specific figures, you can:

1. Open `regenerate_all_figures_locally.py`
2. Comment out figure functions you don't need in the `run_all()` method
3. Run the script

Example to only generate Phase 1 figures:
```python
# In run_all() method, comment out Phase 2:
# self.phase2_fig1_state_occupancy_reversal(data_W_p2, data_F_p2)
# self.phase2_fig2_reversal_performance(data_W_p2, data_F_p2)
```

---

## What's Different from Previous Scripts

This script (`regenerate_all_figures_locally.py`) vs `simple_figure_regenerator.py`:

- ✅ Uses NEW color scheme (red, black, yellow, blue, maroon)
- ✅ Includes Phase 2 figures
- ✅ Better error handling for missing data
- ✅ Cleaner output organization
- ✅ Bolder lines and better styling
- ✅ Sample sizes displayed on all plots

---

## Need More Figures?

To add additional figure types:

1. Add a new method to the `ComprehensiveFigureRegenerator` class
2. Call it from `run_all()`
3. Use the helper functions:
   - `get_genotype_color(genotype)` - Get consistent colors
   - `get_state_color(state)` - Get state colors
   - `get_state_label(state)` - Get state names
   - `relabel_genotype(old_label)` - Convert old to new labels

Example:
```python
def your_new_figure(self, data_W, data_F):
    # Your plotting code here
    # Use get_genotype_color() for consistency
    self.save_figure('phase1/your_figure_name')
```

---

## Questions?

- Check `genotype_labels.py` for available helper functions
- Look at existing figure methods for examples
- All figures use consistent styling and colors automatically
