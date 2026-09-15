# ============================================================
# Step2_Data_Preprocessing.py
# MOBILE PRODUCT DATA PREPROCESSING
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
# 2. CHECK INPUT FILE
# ============================================================

if not os.path.exists(input_file):

    raise FileNotFoundError(

        f"\nERROR: {input_file} not found.\n"
        "Please place the original dataset in the project folder."

    )


# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv(

    input_file,

    low_memory=False

)


# Clean column names
df.columns = (

    df.columns
    .str.strip()

)


print("\n" + "=" * 70)
print("MOBILE PRODUCT DATA PREPROCESSING")
print("=" * 70)


print(

    "\nOriginal Dataset Shape:",

    df.shape

)


print("\nOriginal Columns:")


for column in df.columns:

    print(
        "-",
        column
    )


# ============================================================
# 4. REQUIRED RAW COLUMNS
# ============================================================

required_raw_columns = [

    "brand",
    "model",

    "price_usd",
    "rating",

    "battery_life_rating",
    "camera_rating",
    "performance_rating",
    "design_rating",
    "display_rating",

    "helpful_votes",

    "country"

]


missing_raw_columns = [

    column

    for column in required_raw_columns

    if column not in df.columns

]


if missing_raw_columns:

    raise ValueError(

        "\nERROR: Required columns are missing "
        "from the ORIGINAL dataset:\n\n"

        +

        "\n".join(

            f"- {column}"

            for column in missing_raw_columns

        )

        +

        "\n\nPlease check that you are using "
        "the original Mobile Reviews Sentiment dataset."

    )


# ============================================================
# 5. REMOVE EXACT DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)


duplicate_count = df.duplicated().sum()


print(

    "Duplicate rows found:",

    duplicate_count

)


if duplicate_count > 0:

    df = (

        df
        .drop_duplicates()
        .reset_index(drop=True)

    )


print(

    "Dataset shape after duplicate removal:",

    df.shape

)


# ============================================================
# 6. NUMERIC COLUMNS
# ============================================================

numeric_columns = [

    "age",

    "price_usd",
    "price_local",

    "exchange_rate_to_usd",

    "rating",

    "battery_life_rating",
    "camera_rating",
    "performance_rating",
    "design_rating",
    "display_rating",

    "helpful_votes"

]


# Only convert columns that actually exist
numeric_columns = [

    column

    for column in numeric_columns

    if column in df.columns

]


print("\n" + "=" * 70)
print("NUMERIC TYPE CONVERSION")
print("=" * 70)


for column in numeric_columns:

    df[column] = pd.to_numeric(

        df[column],

        errors="coerce"

    )


print(

    "\nNumeric columns converted successfully."

)


# ============================================================
# 7. HANDLE INFINITE VALUES
# ============================================================

df = df.replace(

    [np.inf, -np.inf],

    np.nan

)


# ============================================================
# 8. CREATE ENGAGEMENT SCORE
# ============================================================

print("\n" + "=" * 70)
print("ENGAGEMENT SCORE FEATURE ENGINEERING")
print("=" * 70)


# helpful_votes represents user engagement.
# log1p reduces the effect of extremely large vote counts.

df["helpful_votes"] = (

    pd.to_numeric(

        df["helpful_votes"],

        errors="coerce"

    )

    .fillna(0)

)


# Make sure negative values do not occur
df["helpful_votes"] = (

    df["helpful_votes"]
    .clip(lower=0)

)


df["engagement_score"] = (

    np.log1p(
        df["helpful_votes"]
    )

)


print(

    "engagement_score created successfully."

)


print(

    "\nEngagement Score Statistics:"

)


print(

    df[
        "engagement_score"
    ]
    .describe()

)


# ============================================================
# 9. HANDLE MISSING NUMERIC VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING NUMERIC VALUES")
print("=" * 70)


numeric_features_for_model = [

    "price_usd",
    "rating",

    "battery_life_rating",
    "camera_rating",
    "performance_rating",
    "design_rating",
    "display_rating",

    "engagement_score"

]


for column in numeric_features_for_model:

    median_value = (

        df[column]
        .median()

    )


    if pd.isna(median_value):

        median_value = 0


    df[column] = (

        df[column]
        .fillna(
            median_value
        )

    )


print(

    "\nMissing numeric values handled successfully."

)


print(

    "\nRemaining missing values:"

)


print(

    df[
        numeric_features_for_model
    ]
    .isnull()
    .sum()

)


# ============================================================
# 10. HANDLE CATEGORICAL COLUMNS
# ============================================================

categorical_columns = [

    "brand",
    "model",
    "country"

]


for column in categorical_columns:

    df[column] = (

        df[column]
        .astype(str)
        .str.strip()

    )


    df[column] = (

        df[column]
        .replace(
            ["", "nan", "None"],
            "Unknown"
        )

    )


print("\n" + "=" * 70)
print("CATEGORICAL VALUES HANDLED")
print("=" * 70)


for column in categorical_columns:

    print(

        f"{column}: "
        f"{df[column].nunique()} unique values"

    )


# ============================================================
# 11. SELECT FINAL COLUMNS
# ============================================================

final_columns = [

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


# Keep only columns that exist
final_columns = [

    column

    for column in final_columns

    if column in df.columns

]


df_cleaned = (

    df[
        final_columns
    ]
    .copy()

)


# ============================================================
# 12. FINAL MISSING VALUE CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATA QUALITY CHECK")
print("=" * 70)


print(

    "\nFinal Dataset Shape:",

    df_cleaned.shape

)


print(

    "\nFinal Columns:"

)


for column in df_cleaned.columns:

    print(
        "-",
        column
    )


print(

    "\nMissing Values:"

)


print(

    df_cleaned
    .isnull()
    .sum()

)


# ============================================================
# 13. VERIFY ENGAGEMENT SCORE
# ============================================================

if "engagement_score" not in df_cleaned.columns:

    raise ValueError(

        "\nERROR: engagement_score was not created.\n"
        "Please check the helpful_votes column."

    )


# ============================================================
# 14. VERIFY CLUSTERING FEATURES
# ============================================================

clustering_features = [

    "price_usd",
    "rating",

    "battery_life_rating",
    "camera_rating",
    "performance_rating",
    "design_rating",
    "display_rating",

    "engagement_score"

]


missing_clustering_features = [

    column

    for column in clustering_features

    if column not in df_cleaned.columns

]


if missing_clustering_features:

    raise ValueError(

        "\nERROR: Clustering features are missing:\n"

        +

        "\n".join(

            f"- {column}"

            for column in missing_clustering_features

        )

    )


print("\n" + "=" * 70)
print("CLUSTERING FEATURE VALIDATION")
print("=" * 70)


print(

    "\nAll 8 clustering features are available."

)


for number, feature in enumerate(

    clustering_features,

    start=1

):

    print(

        f"{number}. {feature}"

    )


# ============================================================
# 15. SAVE CLEANED DATASET
# ============================================================

df_cleaned.to_csv(

    output_file,

    index=False

)


print("\n" + "=" * 70)
print("CLEANED DATASET SAVED")
print("=" * 70)


print(

    "\nOutput File:",

    output_file

)


print(

    "Final Dataset Shape:",

    df_cleaned.shape

)


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 70)


print("\nFeatures prepared for clustering:")


for feature in clustering_features:

    print(
        "-",
        feature
    )


print(

    "\nOutput:",

    "cleaned_mobile_reviews.csv"

)


print("\n" + "=" * 70)
