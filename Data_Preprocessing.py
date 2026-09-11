# ============================================================
# Step2_Data_Preprocessing.py
# ============================================================

import os
import numpy as np
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

df = pd.read_csv(
    input_file,
    low_memory=False
)

# Remove leading/trailing spaces from column names
df.columns = df.columns.str.strip()

print("\nDataset loaded successfully.")
print("Original Dataset Shape:", df.shape)


# ============================================================
# 3. DISPLAY ORIGINAL COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("ORIGINAL COLUMNS")
print("=" * 70)

for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")


# ============================================================
# 4. CHECK REQUIRED COLUMNS
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
        "\nERROR: Required columns are missing:\n"
        + "\n".join(
            f"- {column}"
            for column in missing_columns
        )
    )

print("\nRequired columns verified successfully.")


# ============================================================
# 5. HANDLE BRAND
# ============================================================

if "brand" not in df.columns:

    brand_columns = [
        column
        for column in df.columns
        if column.lower().startswith("brand_")
    ]

    if brand_columns:

        print("\nBrand one-hot columns detected.")

        # Vectorized reconstruction instead of apply()
        brand_values = df[brand_columns].apply(
            pd.to_numeric,
            errors="coerce"
        )

        brand_name = brand_values.idxmax(axis=1)

        max_value = brand_values.max(axis=1)

        df["brand"] = np.where(
            max_value > 0,
            brand_name.str.replace(
                "brand_",
                "",
                regex=False
            ),
            "Unknown"
        )

        # Remove original one-hot columns
        df.drop(
            columns=brand_columns,
            inplace=True
        )

        print("Brand column reconstructed successfully.")

    else:

        print(
            "\nWARNING: 'brand' column not found."
        )

        df["brand"] = "Unknown"


# ============================================================
# 6. HANDLE OPTIONAL COUNTRY COLUMN
# ============================================================

if "country" in df.columns:

    print("\nCountry column detected.")

    df["country"] = (
        df["country"]
        .astype("string")
        .str.strip()
    )

    df["country"] = df["country"].fillna(
        "Unknown"
    )

else:

    print(
        "\nNOTE: Country column is not available "
        "in the dataset."
    )


# ============================================================
# 7. HANDLE MODEL COLUMN
# ============================================================

df["model"] = (
    df["model"]
    .astype("string")
    .str.strip()
)

df["model"] = df["model"].fillna(
    "Unknown"
)


# ============================================================
# 8. NUMERIC COLUMNS
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

# Add engagement score if available
if "engagement_score" in df.columns:
    numeric_columns.append(
        "engagement_score"
    )


# ============================================================
# 9. DATA TYPE CONVERSION
# ============================================================

print("\n" + "=" * 70)
print("DATA TYPE CONVERSION")
print("=" * 70)

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    print(
        f"{column:<25} -> numeric"
    )


# ============================================================
# 10. MISSING VALUES BEFORE CLEANING
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES BEFORE CLEANING")
print("=" * 70)

missing_before = df.isnull().sum()

missing_before = (
    missing_before[
        missing_before > 0
    ]
    .sort_values(
        ascending=False
    )
)

if missing_before.empty:

    print("No missing values found.")

else:

    print(missing_before)

print(
    "\nTotal missing values:",
    df.isnull().sum().sum()
)


# ============================================================
# 11. HANDLE INFINITE VALUES
# ============================================================

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)


# ============================================================
# 12. FILL NUMERIC MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("NUMERIC MISSING VALUE HANDLING")
print("=" * 70)

for column in numeric_columns:

    if df[column].isnull().any():

        median_value = df[column].median()

        if pd.isna(median_value):
            median_value = 0

        df[column] = df[column].fillna(
            median_value
        )

        print(
            f"{column:<25} -> filled with median "
            f"{median_value:.2f}"
        )


# ============================================================
# 13. HANDLE CATEGORICAL MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("CATEGORICAL MISSING VALUE HANDLING")
print("=" * 70)

categorical_columns = [
    column
    for column in [
        "brand",
        "country",
        "model"
    ]
    if column in df.columns
]

for column in categorical_columns:

    missing_count = df[column].isnull().sum()

    if missing_count > 0:

        df[column] = df[column].fillna(
            "Unknown"
        )

        print(
            f"{column:<25} -> filled with 'Unknown'"
        )


# ============================================================
# 14. REMOVE DUPLICATE RECORDS
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE RECORD REMOVAL")
print("=" * 70)

duplicates_before = df.duplicated().sum()

print(
    "Duplicates before removal:",
    duplicates_before
)

df.drop_duplicates(
    inplace=True
)

df.reset_index(
    drop=True,
    inplace=True
)

duplicates_after = df.duplicated().sum()

print(
    "Duplicates after removal:",
    duplicates_after
)


# ============================================================
# 15. SELECT RELEVANT FEATURES
# ============================================================

selected_columns = [
    "brand",
    "model"
]

# Keep country when available
if "country" in df.columns:
    selected_columns.append(
        "country"
    )

selected_columns.extend([
    "price_usd",
    "rating",
    "battery_life_rating",
    "camera_rating",
    "performance_rating",
    "design_rating",
    "display_rating"
])

# Keep engagement score when available
if "engagement_score" in df.columns:
    selected_columns.append(
        "engagement_score"
    )


df_cleaned = df[
    selected_columns
].copy()


# ============================================================
# 16. FINAL MISSING VALUE CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL MISSING VALUE CHECK")
print("=" * 70)

final_missing = df_cleaned.isnull().sum()

print(final_missing)

total_missing = (
    final_missing.sum()
)

print(
    "\nTotal remaining missing values:",
    total_missing
)


# ============================================================
# 17. FINAL DATA TYPE CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATA TYPES")
print("=" * 70)

print(
    df_cleaned.dtypes
)


# ============================================================
# 18. FINAL DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET")
print("=" * 70)

print(
    "\nFinal Dataset Shape:",
    df_cleaned.shape
)

print("\nFinal Columns:")

for i, column in enumerate(
    df_cleaned.columns,
    start=1
):
    print(f"{i}. {column}")


# ============================================================
# 19. DISPLAY SAMPLE DATA
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
# 20. BRAND DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("BRAND DISTRIBUTION")
print("=" * 70)

print(
    df_cleaned["brand"]
    .value_counts()
)


# ============================================================
# 21. SAVE CLEANED DATASET
# ============================================================

df_cleaned.to_csv(
    output_file,
    index=False
)


# ============================================================
# 22. COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    f"\nCleaned dataset saved as:"
    f"\n{output_file}"
)

print("\nPreprocessing performed:")
print("1. Missing value handling")
print("2. Duplicate removal")
print("3. Data type conversion")
print("4. Relevant feature selection")
print("5. Brand/model/country preservation")

print("\nNote:")
print(
    "Encoding and scaling will be performed in "
    "the machine-learning/clustering pipeline."
)
