# ============================================================
# Step1_Data_Collection.py
# ============================================================

import os
import pandas as pd


# ============================================================
# 1. LOAD DATASET
# ============================================================

file_path = "Mobile Reviews Sentiment null.csv"

print("\n" + "=" * 60)
print("MOBILE REVIEWS DATA COLLECTION")
print("=" * 60)

# Check whether file exists
if not os.path.exists(file_path):
    print(f"\nERROR: Dataset file not found!")
    print(f"Expected file: {file_path}")
    raise FileNotFoundError(file_path)

# Load dataset
df = pd.read_csv(file_path, low_memory=False)

print("\nDataset loaded successfully!")


# ============================================================
# 2. DATASET SHAPE
# ============================================================

print("\n" + "=" * 60)
print("DATASET SHAPE")
print("=" * 60)

print(f"Number of Rows    : {df.shape[0]}")
print(f"Number of Columns : {df.shape[1]}")


# ============================================================
# 3. COLUMN NAMES
# ============================================================

print("\n" + "=" * 60)
print("COLUMN NAMES")
print("=" * 60)

for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")


# ============================================================
# 4. DATA TYPES
# ============================================================

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)


# ============================================================
# 5. DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

df.info()


# ============================================================
# 6. FIRST 5 RECORDS
# ============================================================

print("\n" + "=" * 60)
print("FIRST 5 RECORDS")
print("=" * 60)

print(df.head())


# ============================================================
# 7. MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing_values = df.isnull().sum()

print(missing_values)

total_missing = missing_values.sum()

print(f"\nTotal Missing Values: {total_missing}")


# ============================================================
# 8. DUPLICATE RECORDS
# ============================================================

print("\n" + "=" * 60)
print("DUPLICATE RECORDS")
print("=" * 60)

duplicate_count = df.duplicated().sum()

print(f"Number of Duplicate Records: {duplicate_count}")

if len(df) > 0:
    duplicate_percentage = (duplicate_count / len(df)) * 100
    print(f"Duplicate Percentage       : {duplicate_percentage:.2f}%")


# ============================================================
# 9. BASIC STATISTICAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)

print(df.describe(include="all").T)


# ============================================================
# 10. UNIQUE VALUES
# ============================================================

print("\n" + "=" * 60)
print("UNIQUE VALUES")
print("=" * 60)

print(df.nunique().sort_values(ascending=False))


# ============================================================
# 11. SAVE RAW DATASET COPY
# ============================================================

output_file = "collected_mobile_reviews.csv"

df.to_csv(output_file, index=False)

print("\n" + "=" * 60)
print("DATA COLLECTION COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"\nRaw dataset saved as: {output_file}")
