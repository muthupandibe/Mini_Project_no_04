# ============================================================
# FILE: Step1_Data_Collection.py
# PROJECT: Mobile Product Segmentation and Recommendation System
# ============================================================

import pandas as pd


# ============================================================
# 1. LOAD DATASET
# ============================================================

file_path = "Mobile Reviews Sentiment null.csv"

df = pd.read_csv(file_path)


# ============================================================
# 2. DISPLAY BASIC INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("MOBILE REVIEWS DATA COLLECTION")
print("=" * 60)

print("\nDataset loaded successfully!")


# ============================================================
# 3. DATASET SHAPE
# ============================================================

print("\nDataset Shape:")
print(df.shape)

print(f"Number of Rows    : {df.shape[0]}")
print(f"Number of Columns : {df.shape[1]}")


# ============================================================
# 4. COLUMN NAMES
# ============================================================

print("\n" + "=" * 60)
print("COLUMN NAMES")
print("=" * 60)

for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")


# ============================================================
# 5. DATA TYPES
# ============================================================

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)


# ============================================================
# 6. DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

df.info()


# ============================================================
# 7. FIRST 5 RECORDS
# ============================================================

print("\n" + "=" * 60)
print("FIRST 5 RECORDS")
print("=" * 60)

print(df.head())


# ============================================================
# 8. MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing_values = df.isnull().sum()

print(missing_values)


# ============================================================
# 9. DUPLICATE RECORDS
# ============================================================

print("\n" + "=" * 60)
print("DUPLICATE RECORDS")
print("=" * 60)

duplicate_count = df.duplicated().sum()

print("Number of duplicate records:", duplicate_count)


# ============================================================
# 10. BASIC STATISTICAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)

print(df.describe(include="all").T)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("DATA COLLECTION COMPLETED SUCCESSFULLY")
print("=" * 60)