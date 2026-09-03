# ============================================================
# FILE: Step2_Data_Preprocessing.py
# PROJECT: Mobile Product Segmentation and Recommendation System
# ============================================================

import os
import pandas as pd


# ============================================================
# 1. FILE PATHS
# ============================================================

input_file = "Mobile Reviews Sentiment null.csv"
output_file = "cleaned_mobile_reviews.csv"


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("MOBILE PRODUCT DATA PREPROCESSING")
print("=" * 70)

if not os.path.exists(input_file):
    raise FileNotFoundError(
        f"\nERROR: Input file not found: {input_file}"
    )

df = pd.read_csv(input_file)

# Remove spaces from column names
df.columns = df.columns.str.strip()

print("\nDataset loaded successfully.")
print("Original Dataset Shape:", df.shape)


# ============================================================
# 3. DISPLAY ORIGINAL COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("ORIGINAL COLUMNS")
print("=" * 70)

for column in df.columns:
    print("-", column)


# ============================================================
# 4. CHECK REQUIRED PRODUCT COLUMNS
# ============================================================

required_columns = [
    "model",
    "price_usd",
    "rating",
    "battery_life_rating",
    "camera_rating",
    "performance_rating",
    "design_rating",
    "display_rating"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "\nERROR: The following required columns are missing:\n"
        + "\n".join(
            f"- {column}"
            for column in missing_columns
        )
    )


# ============================================================
# 5. HANDLE BRAND
# ============================================================
# The current dataset may already contain one-hot encoded
# brand columns such as:
#
# brand_Apple
# brand_Google
# brand_Motorola
# brand_OnePlus
# brand_Realme
# brand_Samsung
# brand_Xiaomi
#
# If a normal "brand" column exists, keep it.
#
# If only brand_* columns exist, reconstruct the brand column.
# This makes the preprocessing compatible with Step 5.
# ============================================================

brand_columns = [
    column
    for column in df.columns
    if column.startswith("brand_")
]

if "brand" not in df.columns:

    if len(brand_columns) > 0:

        print("\n" + "=" * 70)
        print("RECONSTRUCTING BRAND COLUMN")
        print("=" * 70)

        print(
            "\nOne-hot encoded brand columns detected:"
        )

        for column in brand_columns:
            print("-", column)

        def get_brand(row):

            for column in brand_columns:

                value = row[column]

                if pd.notna(value):

                    try:
                        if float(value) == 1:
                            return column.replace(
                                "brand_",
                                ""
                            )
                    except (ValueError, TypeError):
                        pass

            return "Unknown"

        df["brand"] = df.apply(
            get_brand,
            axis=1
        )

        print(
            "\nOriginal 'brand' column reconstructed successfully."
        )

        # Remove one-hot encoded brand columns
        df = df.drop(
            columns=brand_columns
        )

    else:

        # If no brand information exists
        print(
            "\nWARNING: No brand column or brand_* columns found."
        )

        df["brand"] = "Unknown"


# ============================================================
# 6. CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "price_usd",
    "rating",
    "battery_life_rating",
    "camera_rating",
    "performance_rating",
    "design_rating",
    "display_rating"
]

# Add engagement_score only if it exists
if "engagement_score" in df.columns:
    numeric_columns.append(
        "engagement_score"
    )

print("\n" + "=" * 70)
print("CONVERTING NUMERIC COLUMNS")
print("=" * 70)

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    print(f"{column} -> numeric")


# ============================================================
# 7. MISSING VALUES BEFORE CLEANING
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES BEFORE CLEANING")
print("=" * 70)

missing_before = df.isnull().sum()

print(
    missing_before[
        missing_before > 0
    ]
)


# ============================================================
# 8. FILL NUMERIC MISSING VALUES
# ============================================================

for column in numeric_columns:

    if df[column].isnull().any():

        median_value = df[column].median()

        if pd.isna(median_value):
            median_value = 0

        df[column] = df[column].fillna(
            median_value
        )


# ============================================================
# 9. FILL CATEGORICAL MISSING VALUES
# ============================================================

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

for column in categorical_columns:

    if df[column].isnull().any():

        mode_value = df[column].mode()

        if not mode_value.empty:

            df[column] = df[column].fillna(
                mode_value.iloc[0]
            )

        else:

            df[column] = df[column].fillna(
                "Unknown"
            )


# ============================================================
# 10. REMOVE DUPLICATES
# ============================================================

duplicates_before = df.duplicated().sum()

print("\n" + "=" * 70)
print("DUPLICATE RECORDS")
print("=" * 70)

print(
    "Duplicates before removal:",
    duplicates_before
)

df = df.drop_duplicates().reset_index(
    drop=True
)

duplicates_after = df.duplicated().sum()

print(
    "Duplicates after removal:",
    duplicates_after
)


# ============================================================
# 11. HANDLE INFINITE VALUES
# ============================================================

df = df.replace(
    [float("inf"), float("-inf")],
    pd.NA
)

# Refill numeric values if infinite values
# created missing values
for column in numeric_columns:

    if df[column].isnull().any():

        median_value = df[column].median()

        if pd.isna(median_value):
            median_value = 0

        df[column] = df[column].fillna(
            median_value
        )


# ============================================================
# 12. SELECT RELEVANT COLUMNS
# ============================================================

selected_columns = [
    "brand",
    "model",
    "price_usd",
    "rating",
    "battery_life_rating",
    "camera_rating",
    "performance_rating",
    "design_rating",
    "display_rating"
]

# Add engagement_score only when available
if "engagement_score" in df.columns:

    selected_columns.append(
        "engagement_score"
    )

df_cleaned = df[
    selected_columns
].copy()


# ============================================================
# 13. FINAL MISSING VALUE CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL MISSING VALUE CHECK")
print("=" * 70)

print(
    df_cleaned.isnull().sum()
)

total_missing = (
    df_cleaned.isnull().sum().sum()
)

print(
    "\nTotal remaining missing values:",
    total_missing
)


# ============================================================
# 14. FINAL DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATA TYPES")
print("=" * 70)

print(
    df_cleaned.dtypes
)


# ============================================================
# 15. FINAL DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET")
print("=" * 70)

print(
    "\nFinal Dataset Shape:",
    df_cleaned.shape
)

print("\nFinal Columns:")

for column in df_cleaned.columns:
    print("-", column)


# ============================================================
# 16. DISPLAY SAMPLE DATA
# ============================================================

print("\n" + "=" * 70)
print("FIRST 10 CLEANED RECORDS")
print("=" * 70)

print(
    df_cleaned.head(10).to_string(
        index=False
    )
)


# ============================================================
# 17. BRAND DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("BRAND DISTRIBUTION")
print("=" * 70)

print(
    df_cleaned["brand"]
    .value_counts()
)


# ============================================================
# 18. SAVE CLEANED DATASET
# ============================================================

df_cleaned.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    f"\nCleaned dataset saved as:"
    f"\n{output_file}"
)

print("\nImportant:")
print(
    "Brand and model columns were preserved "
    "for EDA, clustering and recommendation."
)

print(
    "\nScaling and one-hot encoding are NOT performed "
    "in Step 2."
)

print(
    "Scaling will be performed in the ML stages."
)