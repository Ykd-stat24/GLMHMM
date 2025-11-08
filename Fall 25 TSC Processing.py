import pandas as pd
import os

# 1) Your lookup table for Sex & Genotype
lookup = pd.DataFrame({
    'Animal ID': [
        '11', '12', '13', '14',
        '21', '22', '23', '24', '25',
        '31', '32', '33', '34', '41',
        '42', '51', '52', '61', '62',
        '63', '64', '71', '72', '73',
        '81', '82', '83', '84', '91',
        '92', '93', '101', '102', '103',
        '104'
    ],
    'Sex': [
        'M', 'M', 'M', 'M',
        'F', 'F', 'F', 'F', 'F',
        'M', 'M', 'M', 'M', 'M',
        'M', 'M', 'M', 'F', 'F',
        'F', 'F', 'M', 'M', 'M',
        'F', 'F', 'F', 'F', 'M',
        'M', 'M', 'F', 'F', 'F',
        'F'
    ],
    'Genotype': [
        '+', '+', '+', '+',
        '+', '+', '+', '+', '+',
        '+', '+', '+', '+', '-/-',
        '+/+', '+/+', '+/+', '+/-', '+/-',
        '-/-', '-/-', '+/-', '+/+', '+/-',
        '+/-', '-/-', '+/+', '+/+', '+/+',
        '-/-', '-/-', '+/-', '+/-', '-/-',
        '+/+'
    ]
})

# FIX: Explicitly set the lookup 'Animal ID' to string type for safety.
lookup['Animal ID'] = lookup['Animal ID'].astype(str)

# 2) List all three input files
input_files = [
    r'C:\Users\yashodad\Downloads\LD 2025\Yashoda\F LD Data 10.30 pD.csv',
    r'C:\Users\yashodad\Downloads\LD 2025\Yashoda\F LD Data 10.30 LD.csv',
    r'C:\Users\yashodad\Downloads\LD 2025\Yashoda\F LD Data 10.30 All.csv'
]

print("Processing files...")

for infile in input_files:
    try:
        # 3) Read & parse date
        df = pd.read_csv(infile, parse_dates=['Schedule run date'])

        # FIX: Convert the 'Animal ID' column from the CSV to string to match the lookup table.
        # This is the line that solves the ValueError.
        df['Animal ID'] = df['Animal ID'].astype(str)

        # 4) Sort, add Day, merge lookup
        df = df.sort_values(['Animal ID', 'Schedule run date'])
        df['Day'] = df.groupby('Animal ID').cumcount() + 1

        # This merge should work now
        df = df.merge(lookup, on='Animal ID', how='left')

        # 5) Build output path
        folder, fname = os.path.split(infile)
        name, ext = os.path.splitext(fname)
        outfile = os.path.join(folder, f"{name}_processed{ext}")

        # 6) Write out
        df.to_csv(outfile, index=False)
        print(f"→ Successfully written: {outfile}")

    except Exception as e:
        print(f"Failed to process {infile}. Error: {e}")

print("All files processed.")