# Cross-Validation Instructions

## Quick Start

### 1. File is Already Configured
The script `run_cross_validation_local.py` is already set up with your data paths:
- W Cohort: `C:/Users/yashodad/OneDrive - Michigan Medicine/Documents/GitHub/GLMHMM/W LD Data 11.08 All_processed.csv`
- F Cohort: `C:/Users/yashodad/OneDrive - Michigan Medicine/Documents/GitHub/GLMHMM/F LD Data 11.08 All_processed.csv`

### 2. Run the Script
Open a terminal/command prompt in your GLMHMM directory and run:

```bash
python run_cross_validation_local.py
```

That's it! The script will:
- Load your data from the OneDrive paths
- Test 2, 3, 4, and 5 state models on 12 animals (balanced across genotypes)
- Run 5-fold cross-validation for each (more robust than 3-fold)
- Save results to `results/phase1_non_reversal/priority2_validation/`

### 3. Expected Runtime
- **Standard test (12 animals, 5-fold CV)**: ~75-90 minutes
- **Each animal takes**: ~6-7 minutes

---

## Customization (Optional)

### Change Which Animals to Test

Open `run_cross_validation_local.py` and find the `ANIMALS_TO_TEST` section (around line 43):

```python
ANIMALS_TO_TEST = [
    # W Cohort
    ('c1m4', 'W'),  # W+
    ('c2m3', 'W'),  # W+
    ('c4m1', 'W'),  # W-
    # ... etc (12 animals total)
]
```

Add or remove animals as needed. Format is `(animal_id, 'cohort')`.

**Current selection**: 12 animals balanced across all genotypes (W+, W-, F+, F+/+, F+/-, F-/-)

### Change Model Settings

Find the settings section (around line 64):

```python
N_STATES_TO_TEST = [2, 3, 4, 5]  # Which model sizes to compare
N_FOLDS = 5  # Number of cross-validation folds
```

- `N_STATES_TO_TEST`: Which model complexities to test
- `N_FOLDS`: More folds = more robust but slower (5 is standard, good balance)

### Update Data File Paths

If your data moves, update the `DATA_FILES` section (around line 32):

```python
DATA_FILES = {
    'W': r"C:/path/to/your/W_data.csv",
    'F': r"C:/path/to/your/F_data.csv"
}
```

Note: The `r` before the quotes ensures Windows paths work correctly.

---

## Output Files

The script creates these files in `results/phase1_non_reversal/priority2_validation/`:

1. **cross_validation_results.csv**: Raw numerical results
   - Columns: animal_id, cohort, n_states, accuracy, AIC, BIC, etc.

2. **model_cross_validation.png**: Visualization with 4 panels:
   - Log-likelihood comparison
   - Prediction accuracy comparison
   - AIC model selection
   - BIC model selection

3. **model_cross_validation.pdf**: Same as PNG, publication-ready

---

## Troubleshooting

### "File not found" error
- Check that your data files are at the paths specified
- Make sure to use raw strings (r"...") for Windows paths
- Verify the files haven't moved in OneDrive

### "Module not found" error
- Make sure you're in the GLMHMM directory when running
- Check that `glmhmm_utils.py` and `glmhmm_ashwood.py` are present

### Very slow execution
- Reduce the number of animals in `ANIMALS_TO_TEST`
- Try with just 2-3 animals first (~15-20 min)
- Consider reducing N_FOLDS from 5 to 3 (faster, still good)
- Each animal takes ~6-7 minutes with 5-fold CV

### "Animal not found" warning
- The animal ID doesn't exist in your data file
- Check the CSV file to see available animal IDs
- Remove that animal from `ANIMALS_TO_TEST`

---

## Understanding Results

The script tests which number of hidden states (2, 3, 4, or 5) works best:

- **Log-Likelihood**: Higher is better (measures model fit)
- **Accuracy**: Higher is better (% of correct predictions)
- **AIC**: Lower is better (balances fit and complexity)
- **BIC**: Lower is better (like AIC but penalizes complexity more)

**Expected result**: 3-state model should be optimal (lowest AIC/BIC, good accuracy)
