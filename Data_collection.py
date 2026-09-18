# ============================================================
# STEP 1: DATA COLLECTION
# ============================================================

from pathlib import Path
import pandas as pd

# 1. SET PROJECT PATH

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "Mobile Reviews Sentiment null.csv"

# 2. CHECK WHETHER THE INPUT FILE EXISTS

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{INPUT_FILE}\n\n"
        "Please make sure 'Mobile Reviews Sentiment null.csv' "
        "is present in the project folder."
    )

# 3. LOAD THE DATASET

df = pd.read_csv(INPUT_FILE)

# 4. DISPLAY BASIC DATASET INFORMATION

print("=" * 70)
print("STEP 1 - DATA COLLECTION")
print("=" * 70)

print("\nDataset loaded successfully!")
print(f"File: {INPUT_FILE.name}")

# 5. DISPLAY NUMBER OF ROWS AND COLUMNS

print("\n" + "-" * 70)
print("DATASET SHAPE")
print("-" * 70)

print(f"Number of rows    : {df.shape[0]}")
print(f"Number of columns : {df.shape[1]}")

# 6. DISPLAY COLUMN NAMES

print("\n" + "-" * 70)
print("COLUMN NAMES")
print("-" * 70)

for i, column in enumerate(df.columns, start=1):
    print(f"{i:2}. {column}")

# 7. DISPLAY DATA TYPES

print("\n" + "-" * 70)
print("DATA TYPES")
print("-" * 70)

print(df.dtypes)

# 8. DISPLAY FIRST 5 RECORDS

print("\n" + "-" * 70)
print("FIRST 5 RECORDS")
print("-" * 70)

print(df.head())

# 9. DISPLAY LAST 5 RECORDS

print("\n" + "-" * 70)
print("LAST 5 RECORDS")
print("-" * 70)

print(df.tail())

# 10. CHECK MISSING VALUES

print("\n" + "-" * 70)
print("MISSING VALUES")
print("-" * 70)

missing_values = df.isnull().sum()

print(missing_values)

# 11. CHECK DUPLICATE RECORDS

print("\n" + "-" * 70)
print("DUPLICATE RECORDS")
print("-" * 70)

duplicate_count = df.duplicated().sum()

print(f"Number of duplicate rows: {duplicate_count}")

# 12. DISPLAY UNIQUE VALUES FOR IMPORTANT COLUMNS

print("\n" + "-" * 70)
print("UNIQUE VALUE INFORMATION")
print("-" * 70)

if "brand" in df.columns:
    print(f"Number of unique brands  : {df['brand'].nunique()}")

if "model" in df.columns:
    print(f"Number of unique models  : {df['model'].nunique()}")

if "country" in df.columns:
    print(f"Number of unique countries: {df['country'].nunique()}")

# 13. DISPLAY BASIC STATISTICS

print("\n" + "-" * 70)
print("BASIC STATISTICAL SUMMARY")
print("-" * 70)

print(df.describe(include="all").transpose())

# 14. DISPLAY DATASET INFORMATION

print("\n" + "-" * 70)
print("DATASET INFORMATION")
print("-" * 70)

df.info()

# 15. FINAL MESSAGE

print("\n" + "=" * 70)
print("STEP 1 DATA COLLECTION COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nThe dataset has been loaded and inspected.")
print("The following information was checked:")
print("1. Dataset shape")
print("2. Column names")
print("3. Data types")
print("4. First and last records")
print("5. Missing values")
print("6. Duplicate records")
print("7. Unique values")
print("8. Basic statistical summary")
