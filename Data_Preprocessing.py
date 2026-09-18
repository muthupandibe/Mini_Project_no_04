# ============================================================
# STEP 2: DATA PREPROCESSING
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd

# 1. PROJECT PATHS

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "Mobile Reviews Sentiment null.csv"
OUTPUT_FILE = BASE_DIR / "cleaned_mobile_reviews.csv"

# 2. REQUIRED COLUMNS

REQUIRED_COLUMNS = [
    "brand",
    "model",
    "country",
    "price_usd",
    "rating",
    "battery_life_rating",
    "camera_rating",
    "performance_rating",
    "design_rating",
    "display_rating",
    "helpful_votes"
]

# 3. COLUMNS USED FOR CLUSTERING

CLUSTERING_FEATURES = [
    "price_usd",
    "rating",
    "battery_life_rating",
    "camera_rating",
    "performance_rating",
    "design_rating",
    "display_rating",
    "engagement_score"
]

# 4. CATEGORICAL COLUMNS

CATEGORICAL_COLUMNS = [
    "brand",
    "country",
    "model"
]

# 5. FINAL OUTPUT COLUMNS

FINAL_COLUMNS = [
    "brand",
    "model",
    "country",
    "price_usd",
    "rating",
    "battery_life_rating",
    "camera_rating",
    "performance_rating",
    "design_rating",
    "display_rating",
    "helpful_votes",
    "engagement_score"
]

# 6. CHECK INPUT FILE

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input dataset not found:\n{INPUT_FILE}"
    )

# 7. LOAD DATASET

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("STEP 2 - DATA PREPROCESSING")
print("=" * 70)

print("\nOriginal dataset shape:")
print(df.shape)

# 8. CHECK REQUIRED COLUMNS

missing_columns = [
    column for column in REQUIRED_COLUMNS
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "The following required columns are missing:\n"
        + "\n".join(missing_columns)
    )

# 9. REMOVE DUPLICATE RECORDS

duplicates_before = df.duplicated().sum()

print("\nDuplicate records before removal:", duplicates_before)

df = df.drop_duplicates().reset_index(drop=True)

duplicates_after = df.duplicated().sum()

print("Duplicate records after removal :", duplicates_after)

# 10. CONVERT NUMERIC COLUMNS

NUMERIC_COLUMNS = [
    "price_usd",
    "rating",
    "battery_life_rating",
    "camera_rating",
    "performance_rating",
    "design_rating",
    "display_rating",
    "helpful_votes"
]

for column in NUMERIC_COLUMNS:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# 11. HANDLE INVALID INFINITE VALUES

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

# 12. HANDLE MISSING VALUES IN PRICE

if df["price_usd"].isna().any():

    price_median = df["price_usd"].median()

    print(
        f"\nMissing price_usd values: "
        f"{df['price_usd'].isna().sum()}"
    )

    print(
        f"Replacing missing price_usd with median: "
        f"{price_median:.2f}"
    )

    df["price_usd"] = df["price_usd"].fillna(
        price_median
    )

# 13. HANDLE MISSING VALUES IN RATING

if df["rating"].isna().any():

    rating_median = df["rating"].median()

    print(
        f"\nMissing rating values: "
        f"{df['rating'].isna().sum()}"
    )

    print(
        f"Replacing missing rating with median: "
        f"{rating_median:.2f}"
    )

    df["rating"] = df["rating"].fillna(
        rating_median
    )

# 14. HANDLE MISSING SPECIFICATION RATINGS

SPECIFICATION_COLUMNS = [
    "battery_life_rating",
    "camera_rating",
    "performance_rating",
    "design_rating",
    "display_rating"
]

for column in SPECIFICATION_COLUMNS:

    if df[column].isna().any():

        median_value = df[column].median()

        print(
            f"Missing {column}: "
            f"{df[column].isna().sum()} "
            f"-> filling with median {median_value:.2f}"
        )

        df[column] = df[column].fillna(
            median_value
        )

# 15. HANDLE HELPFUL VOTES

missing_votes = df["helpful_votes"].isna().sum()

negative_votes = (
    df["helpful_votes"] < 0
).sum()

print("\nHelpful votes:")
print("Missing values :", missing_votes)
print("Negative values:", negative_votes)


# Missing helpful votes are treated as zero engagement
df["helpful_votes"] = df["helpful_votes"].fillna(0)

# Negative votes are invalid, so convert them to zero
df["helpful_votes"] = df["helpful_votes"].clip(
    lower=0
)

# 16. CREATE ENGAGEMENT SCORE

# log1p reduces the effect of very large helpful-vote values
df["engagement_score"] = np.log1p(
    df["helpful_votes"]
)

print("\nEngagement score created successfully.")

# 17. HANDLE CATEGORICAL COLUMNS

for column in CATEGORICAL_COLUMNS:

    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
    )

    df[column] = df[column].fillna("Unknown")

    df[column] = df[column].replace(
        "",
        "Unknown"
    )

# 18. SELECT RELEVANT FEATURES

df = df[FINAL_COLUMNS].copy()

# 19. FINAL NUMERIC VALIDATION

for column in CLUSTERING_FEATURES:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

# 20. FINAL MISSING VALUE CHECK

remaining_missing = df.isna().sum()

print("\n" + "-" * 70)
print("FINAL MISSING VALUE CHECK")
print("-" * 70)

print(
    remaining_missing[
        remaining_missing > 0
    ]
)

if remaining_missing.sum() > 0:
    raise ValueError(
        "Missing values still exist after preprocessing."
    )

# 21. FINAL DUPLICATE CHECK

final_duplicates = df.duplicated().sum()

print("\nFinal duplicate records:", final_duplicates)

# 22. DISPLAY FINAL DATASET

print("\n" + "-" * 70)
print("FINAL PREPROCESSED DATASET")
print("-" * 70)

print("Shape:", df.shape)

print("\nColumns:")
for column in df.columns:
    print("-", column)

# 23. DISPLAY CLUSTERING FEATURES

print("\n" + "-" * 70)
print("CLUSTERING FEATURES")
print("-" * 70)

for i, feature in enumerate(
    CLUSTERING_FEATURES,
    start=1
):
    print(f"{i}. {feature}")

# 24. DISPLAY SAMPLE DATA

print("\n" + "-" * 70)
print("SAMPLE PREPROCESSED DATA")
print("-" * 70)

print(df.head())

# 25. SAVE PREPROCESSED DATA

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nPreprocessed dataset saved to:")
print(OUTPUT_FILE)

# 26. IMPORTANT NOTE ABOUT ENCODING

print("\n" + "=" * 70)
print("CATEGORICAL VARIABLE HANDLING")
print("=" * 70)

print(
    "Categorical columns identified:"
)

for column in CATEGORICAL_COLUMNS:
    print("-", column)

print(
    "\nThese categorical columns are retained in cleaned_mobile_reviews.csv."
)

print(
    "Product-level encoding is not required for the current "
    "K-Means feature set because clustering uses the 8 numeric "
    "features listed above."
)

# 27. IMPORTANT NOTE ABOUT SCALING

print("\n" + "=" * 70)
print("STANDARDIZATION")
print("=" * 70)

print(
    "The 8 clustering features are prepared as numeric values "
    "in Step 2."
)

print(
    "StandardScaler will be FIT and SAVED in Step 4 using "
    "the product-level aggregated data."
)

print(
    "This keeps the scaler consistent with the K-Means model "
    "and prevents feature-count/scaler mismatch errors."
)

# 28. COMPLETION MESSAGE

print("\n" + "=" * 70)
print("STEP 2 DATA PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nOutput file:")
print(OUTPUT_FILE.name)

print("\nFinal shape:")
print(df.shape)

print("\nFinal columns:")
print(list(df.columns))
