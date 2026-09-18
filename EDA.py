# ============================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. PROJECT PATHS

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "cleaned_mobile_reviews.csv"
OUTPUT_FOLDER = BASE_DIR / "eda_outputs"

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

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
    "helpful_votes",
    "engagement_score"
]

# 3. CHECK INPUT FILE

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Cleaned dataset not found:\n{INPUT_FILE}\n\n"
        "Please run Step 2 first."
    )

# 4. LOAD CLEANED DATA

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("STEP 3 - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print("\nDataset loaded successfully.")
print("Shape:", df.shape)

# 5. CHECK REQUIRED COLUMNS

missing_columns = [
    column
    for column in REQUIRED_COLUMNS
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Required columns are missing from "
        "cleaned_mobile_reviews.csv:\n"
        + "\n".join(missing_columns)
    )

# 6. CONVERT NUMERIC COLUMNS

NUMERIC_COLUMNS = [
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

for column in NUMERIC_COLUMNS:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# 7. HANDLE INFINITE VALUES

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

# 8. VALIDATE MISSING VALUES

missing_values = df.isna().sum()

print("\n" + "-" * 70)
print("MISSING VALUE CHECK")
print("-" * 70)

print(
    missing_values[
        missing_values > 0
    ]
)

if missing_values.sum() > 0:
    raise ValueError(
        "Missing values are present in the cleaned dataset. "
        "Please rerun Step 2."
    )

# REQUIREMENT 1
# PRODUCT DISTRIBUTION ACROSS BRANDS AND COUNTRIES

print("\n" + "=" * 70)
print("1. PRODUCT DISTRIBUTION")
print("=" * 70)

# 9. BRAND DISTRIBUTION

brand_distribution = (
    df["brand"]
    .value_counts()
    .reset_index()
)

brand_distribution.columns = [
    "Brand",
    "Review_Count"
]

print("\nBrand distribution:")
print(brand_distribution)

brand_distribution.to_csv(
    OUTPUT_FOLDER / "brand_distribution.csv",
    index=False
)

# 10. COUNTRY DISTRIBUTION

country_distribution = (
    df["country"]
    .value_counts()
    .reset_index()
)

country_distribution.columns = [
    "Country",
    "Review_Count"
]

print("\nCountry distribution:")
print(country_distribution)


country_distribution.to_csv(
    OUTPUT_FOLDER / "country_distribution.csv",
    index=False
)

# 11. BRAND DISTRIBUTION PLOT

plt.figure(figsize=(10, 6))

plt.bar(
    brand_distribution["Brand"],
    brand_distribution["Review_Count"]
)

plt.title("Mobile Reviews Distribution by Brand")
plt.xlabel("Brand")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    OUTPUT_FOLDER / "brand_distribution.png",
    dpi=300
)

plt.show()
plt.close()

# 12. COUNTRY DISTRIBUTION PLOT

plt.figure(figsize=(10, 6))

plt.bar(
    country_distribution["Country"],
    country_distribution["Review_Count"]
)

plt.title("Mobile Reviews Distribution by Country")
plt.xlabel("Country")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    OUTPUT_FOLDER / "country_distribution.png",
    dpi=300
)

plt.show()
plt.close()

# REQUIREMENT 2
# TOP-RATED AND LOW-RATED PRODUCTS

print("\n" + "=" * 70)
print("2. TOP-RATED AND LOW-RATED PRODUCTS")
print("=" * 70)

# 13. PRODUCT-LEVEL SUMMARY

product_summary = (
    df.groupby(
        ["brand", "model"],
        as_index=False
    )
    .agg(
        Average_Price_USD=("price_usd", "mean"),
        Average_Rating=("rating", "mean"),
        Average_Battery=("battery_life_rating", "mean"),
        Average_Camera=("camera_rating", "mean"),
        Average_Performance=("performance_rating", "mean"),
        Average_Design=("design_rating", "mean"),
        Average_Display=("display_rating", "mean"),
        Average_Engagement=("engagement_score", "mean"),
        Review_Count=("rating", "count")
    )
)

# 14. TOP-RATED PRODUCTS

top_rated_products = (
    product_summary
    .sort_values(
        by=["Average_Rating", "Review_Count"],
        ascending=[False, False]
    )
    .head(10)
    .reset_index(drop=True)
)

print("\nTop-rated products:")
print(top_rated_products)


top_rated_products.to_csv(
    OUTPUT_FOLDER / "top_rated_products.csv",
    index=False
)

# 15. LOW-RATED PRODUCTS

low_rated_products = (
    product_summary
    .sort_values(
        by=["Average_Rating", "Review_Count"],
        ascending=[True, False]
    )
    .head(10)
    .reset_index(drop=True)
)

print("\nLow-rated products:")
print(low_rated_products)


low_rated_products.to_csv(
    OUTPUT_FOLDER / "low_rated_products.csv",
    index=False
)

# REQUIREMENT 3
# RELATIONSHIP BETWEEN PRICE, RATINGS AND SPECIFICATIONS

print("\n" + "=" * 70)
print("3. PRICE, RATING AND SPECIFICATION RELATIONSHIPS")
print("=" * 70)

# 16. CORRELATION ANALYSIS

CORRELATION_COLUMNS = [
    "price_usd",
    "rating",
    "battery_life_rating",
    "camera_rating",
    "performance_rating",
    "design_rating",
    "display_rating",
    "engagement_score"
]

correlation_matrix = (
    df[CORRELATION_COLUMNS]
    .corr()
)


print("\nCorrelation matrix:")
print(
    correlation_matrix.round(3)
)


correlation_matrix.to_csv(
    OUTPUT_FOLDER / "correlation_matrix.csv"
)

# 17. PRICE VS RATING

price_rating_correlation = (
    df["price_usd"]
    .corr(df["rating"])
)

print(
    f"\nPrice vs Rating correlation: "
    f"{price_rating_correlation:.4f}"
)

# 18. PRICE VS PERFORMANCE

price_performance_score = (
    df[
        [
            "rating",
            "battery_life_rating",
            "camera_rating",
            "performance_rating",
            "design_rating",
            "display_rating"
        ]
    ]
    .mean(axis=1)
)

df["overall_performance_score"] = (
    price_performance_score
)

price_performance_correlation = (
    df["price_usd"]
    .corr(df["overall_performance_score"])
)

print(
    f"Price vs Overall Performance correlation: "
    f"{price_performance_correlation:.4f}"
)

# 19. PRICE VS RATING SCATTER PLOT

plt.figure(figsize=(10, 6))

plt.scatter(
    df["price_usd"],
    df["rating"],
    alpha=0.3
)

plt.title("Price vs Rating")
plt.xlabel("Price (USD)")
plt.ylabel("Rating")
plt.tight_layout()

plt.savefig(
    OUTPUT_FOLDER / "price_vs_rating.png",
    dpi=300
)

plt.show()
plt.close()

# 20. PRICE VS PERFORMANCE SCATTER PLOT

plt.figure(figsize=(10, 6))

plt.scatter(
    df["price_usd"],
    df["overall_performance_score"],
    alpha=0.3
)

plt.title("Price vs Overall Performance")
plt.xlabel("Price (USD)")
plt.ylabel("Overall Performance Score")
plt.tight_layout()

plt.savefig(
    OUTPUT_FOLDER / "price_vs_performance.png",
    dpi=300
)

plt.show()
plt.close()

# REQUIREMENT 4
# PATTERNS, TRENDS AND CORRELATIONS

print("\n" + "=" * 70)
print("4. PATTERNS, TRENDS AND CORRELATIONS")
print("=" * 70)

# 21. SPECIFICATION CORRELATIONS WITH RATING

SPECIFICATION_RATING_CORRELATIONS = {}

for column in [
    "battery_life_rating",
    "camera_rating",
    "performance_rating",
    "design_rating",
    "display_rating"
]:

    correlation = df[column].corr(
        df["rating"]
    )

    SPECIFICATION_RATING_CORRELATIONS[
        column
    ] = correlation


specification_correlation_df = (
    pd.DataFrame(
        list(
            SPECIFICATION_RATING_CORRELATIONS.items()
        ),
        columns=[
            "Specification",
            "Correlation_With_Rating"
        ]
    )
    .sort_values(
        "Correlation_With_Rating",
        ascending=False
    )
    .reset_index(drop=True)
)


print("\nSpecification vs Rating correlations:")
print(
    specification_correlation_df.round(4)
)


specification_correlation_df.to_csv(
    OUTPUT_FOLDER /
    "specification_rating_correlations.csv",
    index=False
)

# 22. BRAND-WISE SUMMARY

brand_summary = (
    df.groupby("brand")
    .agg(
        Review_Count=("rating", "count"),
        Average_Price=("price_usd", "mean"),
        Average_Rating=("rating", "mean"),
        Average_Battery=(
            "battery_life_rating",
            "mean"
        ),
        Average_Camera=(
            "camera_rating",
            "mean"
        ),
        Average_Performance=(
            "performance_rating",
            "mean"
        ),
        Average_Design=(
            "design_rating",
            "mean"
        ),
        Average_Display=(
            "display_rating",
            "mean"
        ),
        Average_Engagement=(
            "engagement_score",
            "mean"
        )
    )
    .reset_index()
)


print("\nBrand-wise summary:")
print(
    brand_summary.round(2)
)


brand_summary.to_csv(
    OUTPUT_FOLDER / "brand_summary.csv",
    index=False
)

# 23. COUNTRY-WISE SUMMARY

country_summary = (
    df.groupby("country")
    .agg(
        Review_Count=("rating", "count"),
        Average_Price=("price_usd", "mean"),
        Average_Rating=("rating", "mean"),
        Average_Engagement=(
            "engagement_score",
            "mean"
        )
    )
    .reset_index()
)


print("\nCountry-wise summary:")
print(
    country_summary.round(2)
)


country_summary.to_csv(
    OUTPUT_FOLDER / "country_summary.csv",
    index=False
)

# REQUIREMENT 5
# STATISTICAL SUMMARIES AND COMPARISONS

print("\n" + "=" * 70)
print("5. STATISTICAL SUMMARY")
print("=" * 70)

# 24. NUMERICAL STATISTICS

statistical_summary = (
    df[CORRELATION_COLUMNS]
    .describe()
    .transpose()
)


print("\nStatistical summary:")
print(
    statistical_summary.round(3)
)


statistical_summary.to_csv(
    OUTPUT_FOLDER / "statistical_summary.csv"
)

# 25. BRAND COMPARISON

brand_comparison = (
    brand_summary[
        [
            "brand",
            "Average_Price",
            "Average_Rating",
            "Average_Performance",
            "Average_Engagement"
        ]
    ]
    .sort_values(
        "Average_Rating",
        ascending=False
    )
)


print("\nBrand comparison:")
print(
    brand_comparison.round(3)
)


brand_comparison.to_csv(
    OUTPUT_FOLDER / "brand_comparison.csv",
    index=False
)

# 26. PRICE STATISTICS

print("\n" + "-" * 70)
print("PRICE STATISTICS")
print("-" * 70)

print(
    f"Mean price   : "
    f"${df['price_usd'].mean():.2f}"
)

print(
    f"Median price : "
    f"${df['price_usd'].median():.2f}"
)

print(
    f"Minimum price: "
    f"${df['price_usd'].min():.2f}"
)

print(
    f"Maximum price: "
    f"${df['price_usd'].max():.2f}"
)

# 27. RATING STATISTICS

print("\n" + "-" * 70)
print("RATING STATISTICS")
print("-" * 70)

print(
    f"Mean rating   : "
    f"{df['rating'].mean():.2f}"
)

print(
    f"Median rating : "
    f"{df['rating'].median():.2f}"
)

print(
    f"Minimum rating: "
    f"{df['rating'].min():.2f}"
)

print(
    f"Maximum rating: "
    f"{df['rating'].max():.2f}"
)

# ADDITIONAL EDA VISUALIZATIONS

# 28. RATING DISTRIBUTION

plt.figure(figsize=(10, 6))

plt.hist(
    df["rating"],
    bins=5
)

plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Reviews")
plt.tight_layout()

plt.savefig(
    OUTPUT_FOLDER / "rating_distribution.png",
    dpi=300
)

plt.show()
plt.close()

# 29. PRICE DISTRIBUTION

plt.figure(figsize=(10, 6))

plt.hist(
    df["price_usd"],
    bins=30
)

plt.title("Price Distribution")
plt.xlabel("Price (USD)")
plt.ylabel("Number of Reviews")
plt.tight_layout()

plt.savefig(
    OUTPUT_FOLDER / "price_distribution.png",
    dpi=300
)

plt.show()
plt.close()

# 30. SPECIFICATION AVERAGES

specification_means = (
    df[
        [
            "battery_life_rating",
            "camera_rating",
            "performance_rating",
            "design_rating",
            "display_rating"
        ]
    ]
    .mean()
    .sort_values(ascending=False)
)


print("\nAverage specification ratings:")
print(
    specification_means.round(3)
)


specification_means.to_csv(
    OUTPUT_FOLDER / "specification_means.csv"
)

# 31. SAVE PRODUCT-LEVEL SUMMARY

product_summary.to_csv(
    OUTPUT_FOLDER / "product_level_summary.csv",
    index=False
)

# 32. SAVE EDA DATASET WITH PERFORMANCE SCORE

df.to_csv(
    OUTPUT_FOLDER / "eda_processed_data.csv",
    index=False
)

# FINAL SUMMARY

print("\n" + "=" * 70)
print("STEP 3 EDA COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nEDA analysis completed for:")
print(f"- {df.shape[0]:,} review records")
print(f"- {df['brand'].nunique()} brands")
print(f"- {df['model'].nunique()} models")
print(f"- {df['country'].nunique()} countries")

print("\nImportant analyses completed:")
print("1. Brand distribution")
print("2. Country distribution")
print("3. Top-rated products")
print("4. Low-rated products")
print("5. Price vs rating relationship")
print("6. Price vs performance relationship")
print("7. Specification correlations")
print("8. Brand-wise comparison")
print("9. Country-wise comparison")
print("10. Statistical summaries")

print("\nEDA output folder:")
print(OUTPUT_FOLDER)
