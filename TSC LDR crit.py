import pandas as pd
import numpy as np
import os, re

# ------------------------------------------------------------------
# 1.  Paths
# ------------------------------------------------------------------
infile  = r'H:/My Documents/Post Bacc. Data/TSC LD co2/LDR 2025 data1_processed.csv'
folder, base = os.path.split(infile)
name,  ext   = os.path.splitext(base)
outfile = os.path.join(folder, f'{name}_withSecondCriterion{ext}')

# ------------------------------------------------------------------
# 2.  Read file
# ------------------------------------------------------------------
df = pd.read_csv(infile)

# ------------------------------------------------------------------
# 3.  Grab & sort the 30 trial columns (robust, no fancy Index magic)
# ------------------------------------------------------------------
pat = re.compile(r'Trial Analysis\s*-\s*No\.?\s*Correct\s*\(\s*(\d+)\s*\)', re.I)

trial_cols = [
    col for col in df.columns if pat.search(col)          # keep only matching cols
]
# sort by the integer inside (...)
trial_cols.sort(key=lambda c: int(pat.search(c).group(1)))

# ------------------------------------------------------------------
# 4.  Ensure clean 0/1 ints (missing → 0)
# ------------------------------------------------------------------
df[trial_cols] = (
    df[trial_cols]
      .apply(pd.to_numeric, errors='coerce')
      .fillna(0)
      .astype(int)
)

# ------------------------------------------------------------------
# 5.  Helper: count 5-of-6 blocks AFTER the first 7-of-8 block
# ------------------------------------------------------------------
def second_criterion_count(trials: np.ndarray) -> int:
    # -------------- first 7-of-8 --------------------
    first_end = -1
    for i in range(len(trials) - 7):
        if trials[i:i+8].sum() >= 7:
            first_end = i + 7      # last index in that 8-trial window
            break
    if first_end < 0:              # never hit original criterion
        return 0

    # -------------- chunk remainder into 6s ----------
    remaining = trials[first_end+1:]
    blocks    = len(remaining) // 6
    return sum(remaining[b*6 : b*6+6].sum() >= 5 for b in range(blocks))

# ------------------------------------------------------------------
# 6.  Apply row-wise
# ------------------------------------------------------------------
df['Second_Criterion_Count'] = df[trial_cols].apply(
    lambda r: second_criterion_count(r.values), axis=1
)

# ------------------------------------------------------------------
# 7.  Save
# ------------------------------------------------------------------
df.to_csv(outfile, index=False)
print(f'✓  Wrote: {outfile}')
