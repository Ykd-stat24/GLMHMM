"""
Genotype Label Mappings and Utilities
======================================

Provides consistent genotype labeling across all visualization and analysis scripts.

Genotype Naming Convention:
- Old → New
- +/+ → A1D_Wt (Wild-type)
- +/- → A1D_Het (Heterozygous)
- -/- → A1D_KO (Knockout)
- +   → B6 (C57BL/6J)
- -   → C3H x B6 (C3H/HeJ × C57BL/6J)
"""

import pandas as pd
import numpy as np

# Main genotype mapping
GENOTYPE_MAP = {
    '+/+': 'A1D_Wt',
    '+/-': 'A1D_Het',
    '-/-': 'A1D_KO',
    '+': 'B6',
    '-': 'C3H x B6'
}

# Reverse mapping for conversion back if needed
GENOTYPE_MAP_REVERSE = {v: k for k, v in GENOTYPE_MAP.items()}

# Display order for plots (preferred ordering)
GENOTYPE_ORDER = ['B6', 'C3H x B6', 'A1D_Wt', 'A1D_Het', 'A1D_KO']

# Color palette for genotypes (consistent across all figures)
GENOTYPE_COLORS = {
    'B6': '#e74c3c',           # Red
    'C3H x B6': '#000000',     # Black
    'A1D_Wt': '#FFD700',       # Yellow/Gold
    'A1D_Het': '#4169E1',      # Blue (Royal Blue)
    'A1D_KO': '#800000'        # Maroon
}

# State names with descriptions
STATE_LABELS = {
    0: 'Engaged',
    1: 'Biased',
    2: 'Lapsed',
    'State 0': 'Engaged',
    'State 1': 'Biased',
    'State 2': 'Lapsed'
}

# State colors (consistent across all figures)
STATE_COLORS = {
    0: '#2ecc71',  # Green - Engaged
    1: '#f39c12',  # Orange - Biased
    2: '#e74c3c',  # Red - Lapsed
    'Engaged': '#2ecc71',
    'Biased': '#f39c12',
    'Lapsed': '#e74c3c'
}

# Long-form state descriptions for figures
STATE_DESCRIPTIONS = {
    0: 'Engaged\n(High accuracy)',
    1: 'Biased\n(Side preference)',
    2: 'Lapsed\n(Disengaged)',
    'Engaged': 'Engaged\n(High accuracy)',
    'Biased': 'Biased\n(Side preference)',
    'Lapsed': 'Lapsed\n(Disengaged)'
}


def relabel_genotype(genotype):
    """
    Convert old genotype label to new label.

    Args:
        genotype: str or array-like of genotype labels

    Returns:
        Relabeled genotype(s)
    """
    if isinstance(genotype, str):
        return GENOTYPE_MAP.get(genotype, genotype)
    elif isinstance(genotype, (list, np.ndarray, pd.Series)):
        return [GENOTYPE_MAP.get(g, g) for g in genotype]
    else:
        return genotype


def relabel_genotype_df(df, column='genotype'):
    """
    Relabel genotype column in a DataFrame.

    Args:
        df: pandas DataFrame
        column: name of genotype column (default: 'genotype')

    Returns:
        DataFrame with relabeled genotypes
    """
    df = df.copy()
    if column in df.columns:
        df[column] = df[column].map(lambda x: GENOTYPE_MAP.get(x, x))
    return df


def get_state_label(state, long_form=False):
    """
    Get human-readable state label.

    Args:
        state: int or str state identifier
        long_form: if True, return description with details

    Returns:
        State label string
    """
    if long_form:
        return STATE_DESCRIPTIONS.get(state, f'State {state}')
    else:
        return STATE_LABELS.get(state, f'State {state}')


def get_state_color(state):
    """
    Get consistent color for a state.

    Args:
        state: int or str state identifier

    Returns:
        Hex color code
    """
    return STATE_COLORS.get(state, '#95a5a6')  # Gray default


def get_genotype_color(genotype):
    """
    Get consistent color for a genotype.

    Args:
        genotype: str genotype label (new format)

    Returns:
        Hex color code
    """
    return GENOTYPE_COLORS.get(genotype, '#95a5a6')  # Gray default


def sort_by_genotype_order(df, column='genotype'):
    """
    Sort DataFrame by preferred genotype order.

    Args:
        df: pandas DataFrame
        column: name of genotype column

    Returns:
        Sorted DataFrame
    """
    df = df.copy()
    if column in df.columns:
        # Create categorical with specified order
        df[column] = pd.Categorical(
            df[column],
            categories=GENOTYPE_ORDER,
            ordered=True
        )
        df = df.sort_values(column)
    return df


def format_genotype_for_filename(genotype):
    """
    Convert genotype label to filename-safe string.

    Args:
        genotype: str genotype label

    Returns:
        Filename-safe string
    """
    return genotype.replace('/', '_').replace(' ', '_').replace('x', 'x')


# Summary function for checking relabeling
def print_genotype_mapping():
    """Print the genotype mapping for reference."""
    print("Genotype Label Mapping:")
    print("=" * 40)
    for old, new in GENOTYPE_MAP.items():
        print(f"  {old:5s} → {new}")
    print("\nDisplay Order:", ' → '.join(GENOTYPE_ORDER))
    print("\nState Labels:")
    for i in range(3):
        print(f"  State {i}: {get_state_label(i)}")


if __name__ == '__main__':
    # Test the module
    print_genotype_mapping()
