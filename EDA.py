# ============================================================
# FILE: Step3_EDA.py
# PROJECT: Mobile Product Segmentation and Recommendation System
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. LOAD CLEANED DATASET
# ============================================================

input_file = "cleaned_mobile_reviews.csv"

if not os.path.exists(input_file):
    raise FileNotFoundError(
        f"\nERROR: {input_file} not found.\n"
        "Please run Step2_Data_Preprocessing.py first."
    )

df = pd.read_csv(input_file)

# Remove spaces from column names
df.columns = df.columns.str.strip()


print("\n" + "=" * 70)
print("MOBILE PRODUCT EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print("\nDataset Shape:", df.shape)


# ============================================================
# 2. DISPLAY DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("COLUMN NAMES")
print("=" * 70)

for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")


print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes)


print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

print(df.isnull().sum())


# ============================================================
# 3. STATISTICAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STATISTICAL SUMMARY")
print("=" * 70)

print(
    df.describe(
        include="all"
    ).T
)


# ============================================================
# 4. BRAND DISTRIBUTION
# ============================================================

if "brand" in df.columns:

    brand_distribution = (
        df.groupby("brand")["model"]
        .nunique()
        .sort_values(
            ascending=False
        )
    )

    print("\n" + "=" * 70)
    print("NUMBER OF UNIQUE PRODUCTS BY BRAND")
    print("=" * 70)

    print(brand_distribution)

    plt.figure(figsize=(10, 6))

    brand_distribution.plot(
        kind="bar"
    )

    plt.title(
        "Number of Unique Mobile Products by Brand"
    )

    plt.xlabel("Brand")
    plt.ylabel("Number of Products")

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# ============================================================
# 5. COUNTRY ANALYSIS
# ============================================================
# Run this only if country exists in the dataset.
# ============================================================

if "country" in df.columns:

    country_distribution = (
        df.groupby("country")["model"]
        .nunique()
        .sort_values(
            ascending=False
        )
    )

    print("\n" + "=" * 70)
    print("NUMBER OF UNIQUE PRODUCTS BY COUNTRY")
    print("=" * 70)

    print(country_distribution)

    plt.figure(figsize=(10, 6))

    country_distribution.head(15).plot(
        kind="bar"
    )

    plt.title(
        "Top Countries by Number of Mobile Products"
    )

    plt.xlabel("Country")
    plt.ylabel("Number of Products")

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# ============================================================
# 6. TOP-RATED PRODUCTS
# ============================================================

if all(
    column in df.columns
    for column in ["brand", "model", "rating"]
):

    top_rated_products = (
        df.groupby(
            ["brand", "model"]
        )["rating"]
        .mean()
        .sort_values(
            ascending=False
        )
        .head(10)
    )

    print("\n" + "=" * 70)
    print("TOP 10 RATED PRODUCTS")
    print("=" * 70)

    print(top_rated_products)


    plt.figure(figsize=(10, 6))

    top_rated_products.sort_values().plot(
        kind="barh"
    )

    plt.title(
        "Top 10 Rated Mobile Products"
    )

    plt.xlabel("Average Rating")
    plt.ylabel("Product")

    plt.tight_layout()
    plt.show()


# ============================================================
# 7. LOWEST-RATED PRODUCTS
# ============================================================

if all(
    column in df.columns
    for column in ["brand", "model", "rating"]
):

    low_rated_products = (
        df.groupby(
            ["brand", "model"]
        )["rating"]
        .mean()
        .sort_values(
            ascending=True
        )
        .head(10)
    )

    print("\n" + "=" * 70)
    print("LOWEST 10 RATED PRODUCTS")
    print("=" * 70)

    print(low_rated_products)


    plt.figure(figsize=(10, 6))

    low_rated_products.sort_values().plot(
        kind="barh"
    )

    plt.title(
        "Lowest 10 Rated Mobile Products"
    )

    plt.xlabel("Average Rating")
    plt.ylabel("Product")

    plt.tight_layout()
    plt.show()


# ============================================================
# 8. PRICE VS RATING
# ============================================================

if all(
    column in df.columns
    for column in ["price_usd", "rating"]
):

    correlation = df[
        ["price_usd", "rating"]
    ].corr()

    print("\n" + "=" * 70)
    print("PRICE VS RATING CORRELATION")
    print("=" * 70)

    print(correlation)


    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        data=df,
        x="price_usd",
        y="rating",
        alpha=0.5
    )

    plt.title(
        "Price vs Rating"
    )

    plt.xlabel("Price (USD)")
    plt.ylabel("Rating")

    plt.tight_layout()
    plt.show()


# ============================================================
# 9. PRICE VS PERFORMANCE
# ============================================================

if all(
    column in df.columns
    for column in [
        "price_usd",
        "performance_rating"
    ]
):

    correlation = df[
        [
            "price_usd",
            "performance_rating"
        ]
    ].corr()

    print("\n" + "=" * 70)
    print("PRICE VS PERFORMANCE CORRELATION")
    print("=" * 70)

    print(correlation)


    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        data=df,
        x="price_usd",
        y="performance_rating",
        alpha=0.5
    )

    plt.title(
        "Price vs Performance Rating"
    )

    plt.xlabel("Price (USD)")
    plt.ylabel("Performance Rating")

    plt.tight_layout()
    plt.show()


# ============================================================
# 10. PRICE VS CAMERA RATING
# ============================================================

if all(
    column in df.columns
    for column in [
        "price_usd",
        "camera_rating"
    ]
):

    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        data=df,
        x="price_usd",
        y="camera_rating",
        alpha=0.5
    )

    plt.title(
        "Price vs Camera Rating"
    )

    plt.xlabel("Price (USD)")
    plt.ylabel("Camera Rating")

    plt.tight_layout()
    plt.show()


# ============================================================
# 11. PRICE VS BATTERY RATING
# ============================================================

if all(
    column in df.columns
    for column in [
        "price_usd",
        "battery_life_rating"
    ]
):

    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        data=df,
        x="price_usd",
        y="battery_life_rating",
        alpha=0.5
    )

    plt.title(
        "Price vs Battery Life Rating"
    )

    plt.xlabel("Price (USD)")
    plt.ylabel("Battery Life Rating")

    plt.tight_layout()
    plt.show()


# ============================================================
# 12. RATING DISTRIBUTION
# ============================================================

if "rating" in df.columns:

    plt.figure(figsize=(8, 6))

    sns.histplot(
        df["rating"],
        bins=20,
        kde=True
    )

    plt.title(
        "Distribution of Mobile Product Ratings"
    )

    plt.xlabel("Rating")
    plt.ylabel("Number of Records")

    plt.tight_layout()
    plt.show()


# ============================================================
# 13. CORRELATION ANALYSIS
# ============================================================

correlation_features = [
    "price_usd",
    "rating",
    "battery_life_rating",
    "camera_rating",
    "performance_rating",
    "design_rating",
    "display_rating"
]

if "engagement_score" in df.columns:

    correlation_features.append(
        "engagement_score"
    )


available_correlation_features = [
    column
    for column in correlation_features
    if column in df.columns
]


if len(available_correlation_features) >= 2:

    correlation_matrix = df[
        available_correlation_features
    ].corr()

    print("\n" + "=" * 70)
    print("CORRELATION MATRIX")
    print("=" * 70)

    print(
        correlation_matrix.round(2)
    )


    plt.figure(
        figsize=(10, 8)
    )

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title(
        "Correlation Between Product Features"
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 14. BRAND-WISE AVERAGE RATING
# ============================================================

if all(
    column in df.columns
    for column in ["brand", "rating"]
):

    brand_rating = (
        df.groupby("brand")["rating"]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    print("\n" + "=" * 70)
    print("BRAND-WISE AVERAGE RATING")
    print("=" * 70)

    print(
        brand_rating.round(2)
    )


    plt.figure(figsize=(10, 6))

    brand_rating.plot(
        kind="bar"
    )

    plt.title(
        "Average Rating by Brand"
    )

    plt.xlabel("Brand")
    plt.ylabel("Average Rating")

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# ============================================================
# 15. BRAND-WISE AVERAGE PRICE
# ============================================================

if all(
    column in df.columns
    for column in ["brand", "price_usd"]
):

    brand_price = (
        df.groupby("brand")["price_usd"]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    print("\n" + "=" * 70)
    print("BRAND-WISE AVERAGE PRICE")
    print("=" * 70)

    print(
        brand_price.round(2)
    )


    plt.figure(figsize=(10, 6))

    brand_price.plot(
        kind="bar"
    )

    plt.title(
        "Average Price by Brand"
    )

    plt.xlabel("Brand")
    plt.ylabel("Average Price (USD)")

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# ============================================================
# 16. SPECIFICATION COMPARISON
# ============================================================

specification_features = [
    "battery_life_rating",
    "camera_rating",
    "performance_rating",
    "design_rating",
    "display_rating"
]

available_specifications = [
    column
    for column in specification_features
    if column in df.columns
]


if available_specifications:

    specification_summary = (
        df[available_specifications]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    print("\n" + "=" * 70)
    print("AVERAGE PRODUCT SPECIFICATION RATINGS")
    print("=" * 70)

    print(
        specification_summary.round(2)
    )


    plt.figure(figsize=(10, 6))

    specification_summary.plot(
        kind="bar"
    )

    plt.title(
        "Average Mobile Product Specification Ratings"
    )

    plt.xlabel("Specification")
    plt.ylabel("Average Rating")

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# ============================================================
# 17. SAVE EDA SPECIFICATION SUMMARY
# ============================================================

if available_specifications:

    eda_specification_summary = pd.DataFrame({

        "Feature":
            specification_summary.index,

        "Average_Rating":
            specification_summary.values.round(2)
    })

    eda_specification_summary.to_csv(
        "eda_specification_summary.csv",
        index=False
    )

    print(
        "\nEDA specification summary saved as:"
    )

    print(
        "eda_specification_summary.csv"
    )


# ============================================================
# 18. SAVE BRAND SUMMARY
# ============================================================

if "brand" in df.columns:

    brand_summary = (
        df.groupby("brand")
        .agg(
            Product_Count=(
                "model",
                "nunique"
            )
            if "model" in df.columns
            else (
                "brand",
                "count"
            ),

            Average_Rating=(
                "rating",
                "mean"
            )
            if "rating" in df.columns
            else (
                "brand",
                "count"
            ),

            Average_Price=(
                "price_usd",
                "mean"
            )
            if "price_usd" in df.columns
            else (
                "brand",
                "count"
            )
        )
        .reset_index()
    )

    brand_summary[
        "Average_Rating"
    ] = brand_summary[
        "Average_Rating"
    ].round(2)

    brand_summary[
        "Average_Price"
    ] = brand_summary[
        "Average_Price"
    ].round(2)

    brand_summary.to_csv(
        "eda_brand_summary.csv",
        index=False
    )

    print(
        "\nEDA brand summary saved as:"
    )

    print(
        "eda_brand_summary.csv"
    )


# ============================================================
# 19. COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nEDA outputs:")
print("1. EDA visualizations")
print("2. eda_specification_summary.csv")
print("3. eda_brand_summary.csv")

print(
    "\nThe cleaned dataset remains in original "
    "business units for analysis."
)

print(
    "\nScaling will be performed later during "
    "clustering/recommendation."
)