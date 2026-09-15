# ============================================================
# Step3_EDA.py
# Mobile Product Segmentation and Recommendation System
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. FILE PATHS
# ============================================================

INPUT_FILE = "cleaned_mobile_reviews.csv"

EDA_BRAND_OUTPUT = "eda_brand_summary.csv"
EDA_SPEC_OUTPUT = "eda_specification_summary.csv"
EDA_PRODUCT_OUTPUT = "eda_product_summary.csv"
EDA_COUNTRY_OUTPUT = "eda_country_summary.csv"


# ============================================================
# 2. PROJECT HEADER
# ============================================================

print("\n" + "=" * 70)
print("MOBILE PRODUCT EXPLORATORY DATA ANALYSIS")
print("=" * 70)


# ============================================================
# 3. CHECK INPUT FILE
# ============================================================

if not os.path.isfile(INPUT_FILE):

    raise FileNotFoundError(
        f"\nERROR: {INPUT_FILE} not found.\n"
        "Please run Step2_Data_Preprocessing.py first."
    )


# ============================================================
# 4. LOAD DATASET
# ============================================================

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

df.columns = (
    df.columns
    .str.strip()
)

print("\nDataset loaded successfully.")
print("Dataset Shape:", df.shape)


# ============================================================
# 5. DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("COLUMN NAMES")
print("=" * 70)

for i, column in enumerate(
    df.columns,
    start=1
):
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
# 6. STATISTICAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STATISTICAL SUMMARY")
print("=" * 70)

numeric_columns = df.select_dtypes(
    include="number"
).columns.tolist()

print(
    df[numeric_columns]
    .describe()
    .T
)


# ============================================================
# 7. BRAND DISTRIBUTION
# ============================================================

if all(
    column in df.columns
    for column in ["brand", "model"]
):

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


    plt.figure(
        figsize=(10, 6)
    )

    brand_distribution.plot(
        kind="bar"
    )

    plt.title(
        "Number of Unique Mobile Products by Brand"
    )

    plt.xlabel("Brand")
    plt.ylabel("Number of Unique Products")

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 8. COUNTRY ANALYSIS
# ============================================================

if all(
    column in df.columns
    for column in ["country", "model"]
):

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


    plt.figure(
        figsize=(10, 6)
    )

    country_distribution.head(15).plot(
        kind="bar"
    )

    plt.title(
        "Top Countries by Number of Mobile Products"
    )

    plt.xlabel("Country")
    plt.ylabel("Number of Unique Products")

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()
    plt.show()


    country_summary = pd.DataFrame({
        "Country":
            country_distribution.index,

        "Unique_Product_Count":
            country_distribution.values
    })

    country_summary.to_csv(
        EDA_COUNTRY_OUTPUT,
        index=False
    )


# ============================================================
# 9. PRODUCT-LEVEL SUMMARY
# ============================================================

required_product_columns = [
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

if all(
    column in df.columns
    for column in required_product_columns
):

    product_summary = (
        df.groupby(
            ["brand", "model"]
        )
        .agg(
            Average_Price=(
                "price_usd",
                "mean"
            ),

            Average_Rating=(
                "rating",
                "mean"
            ),

            Battery_Rating=(
                "battery_life_rating",
                "mean"
            ),

            Camera_Rating=(
                "camera_rating",
                "mean"
            ),

            Performance_Rating=(
                "performance_rating",
                "mean"
            ),

            Design_Rating=(
                "design_rating",
                "mean"
            ),

            Display_Rating=(
                "display_rating",
                "mean"
            ),

            Review_Count=(
                "rating",
                "count"
            )
        )
        .reset_index()
    )


    product_summary[
        "Average_Price"
    ] = product_summary[
        "Average_Price"
    ].round(2)

    product_summary[
        "Average_Rating"
    ] = product_summary[
        "Average_Rating"
    ].round(2)

    product_summary.to_csv(
        EDA_PRODUCT_OUTPUT,
        index=False
    )

    print("\n" + "=" * 70)
    print("PRODUCT-LEVEL SUMMARY")
    print("=" * 70)

    print(
        product_summary.head(10)
        .to_string(index=False)
    )


# ============================================================
# 10. TOP-RATED PRODUCTS
# ============================================================

if all(
    column in df.columns
    for column in [
        "brand",
        "model",
        "rating"
    ]
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

    print(
        top_rated_products.round(2)
    )


    plt.figure(
        figsize=(10, 6)
    )

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
# 11. LOWEST-RATED PRODUCTS
# ============================================================

if all(
    column in df.columns
    for column in [
        "brand",
        "model",
        "rating"
    ]
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

    print(
        low_rated_products.round(2)
    )


    plt.figure(
        figsize=(10, 6)
    )

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
# 12. PRICE VS RATING
# ============================================================

if all(
    column in df.columns
    for column in [
        "price_usd",
        "rating"
    ]
):

    correlation = df[
        [
            "price_usd",
            "rating"
        ]
    ].corr().iloc[0, 1]

    print("\n" + "=" * 70)
    print("PRICE VS RATING")
    print("=" * 70)

    print(
        f"Correlation: {correlation:.3f}"
    )


    plt.figure(
        figsize=(8, 6)
    )

    sns.scatterplot(
        data=df,
        x="price_usd",
        y="rating",
        alpha=0.5
    )

    plt.title(
        "Price vs Rating"
    )

    plt.xlabel(
        "Price (USD)"
    )

    plt.ylabel(
        "Rating"
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 13. PRICE VS PERFORMANCE
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
    ].corr().iloc[0, 1]

    print("\n" + "=" * 70)
    print("PRICE VS PERFORMANCE")
    print("=" * 70)

    print(
        f"Correlation: {correlation:.3f}"
    )


    plt.figure(
        figsize=(8, 6)
    )

    sns.scatterplot(
        data=df,
        x="price_usd",
        y="performance_rating",
        alpha=0.5
    )

    plt.title(
        "Price vs Performance Rating"
    )

    plt.xlabel(
        "Price (USD)"
    )

    plt.ylabel(
        "Performance Rating"
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 14. PRICE VS CAMERA
# ============================================================

if all(
    column in df.columns
    for column in [
        "price_usd",
        "camera_rating"
    ]
):

    plt.figure(
        figsize=(8, 6)
    )

    sns.scatterplot(
        data=df,
        x="price_usd",
        y="camera_rating",
        alpha=0.5
    )

    plt.title(
        "Price vs Camera Rating"
    )

    plt.xlabel(
        "Price (USD)"
    )

    plt.ylabel(
        "Camera Rating"
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 15. PRICE VS BATTERY
# ============================================================

if all(
    column in df.columns
    for column in [
        "price_usd",
        "battery_life_rating"
    ]
):

    plt.figure(
        figsize=(8, 6)
    )

    sns.scatterplot(
        data=df,
        x="price_usd",
        y="battery_life_rating",
        alpha=0.5
    )

    plt.title(
        "Price vs Battery Life Rating"
    )

    plt.xlabel(
        "Price (USD)"
    )

    plt.ylabel(
        "Battery Life Rating"
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 16. RATING DISTRIBUTION
# ============================================================

if "rating" in df.columns:

    plt.figure(
        figsize=(8, 6)
    )

    sns.histplot(
        df["rating"],
        bins=20,
        kde=True
    )

    plt.title(
        "Distribution of Mobile Product Ratings"
    )

    plt.xlabel(
        "Rating"
    )

    plt.ylabel(
        "Number of Reviews"
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 17. CORRELATION MATRIX
# ============================================================

correlation_features = [
    "price_usd",
    "rating",
    "battery_life_rating",
    "camera_rating",
    "performance_rating",
    "design_rating",
    "display_rating",
    "engagement_score"
]

available_features = [
    column
    for column in correlation_features
    if column in df.columns
]

if len(available_features) >= 2:

    correlation_matrix = (
        df[available_features]
        .corr()
    )

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
        fmt=".2f",
        linewidths=0.5
    )

    plt.title(
        "Correlation Between Product Features"
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 18. BRAND-WISE AVERAGE RATING
# ============================================================

if all(
    column in df.columns
    for column in [
        "brand",
        "rating"
    ]
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


    plt.figure(
        figsize=(10, 6)
    )

    brand_rating.plot(
        kind="bar"
    )

    plt.title(
        "Average Rating by Brand"
    )

    plt.xlabel(
        "Brand"
    )

    plt.ylabel(
        "Average Rating"
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 19. BRAND-WISE AVERAGE PRICE
# ============================================================

if all(
    column in df.columns
    for column in [
        "brand",
        "price_usd"
    ]
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


    plt.figure(
        figsize=(10, 6)
    )

    brand_price.plot(
        kind="bar"
    )

    plt.title(
        "Average Price by Brand"
    )

    plt.xlabel(
        "Brand"
    )

    plt.ylabel(
        "Average Price (USD)"
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 20. BRAND SUMMARY TABLE
# ============================================================

if "brand" in df.columns:

    aggregation = {}

    if "model" in df.columns:
        aggregation[
            "Unique_Products"
        ] = (
            "model",
            "nunique"
        )

    if "rating" in df.columns:
        aggregation[
            "Average_Rating"
        ] = (
            "rating",
            "mean"
        )

    if "price_usd" in df.columns:
        aggregation[
            "Average_Price_USD"
        ] = (
            "price_usd",
            "mean"
        )

    if "performance_rating" in df.columns:
        aggregation[
            "Average_Performance"
        ] = (
            "performance_rating",
            "mean"
        )

    if "camera_rating" in df.columns:
        aggregation[
            "Average_Camera"
        ] = (
            "camera_rating",
            "mean"
        )

    if "battery_life_rating" in df.columns:
        aggregation[
            "Average_Battery"
        ] = (
            "battery_life_rating",
            "mean"
        )

    if aggregation:

        brand_summary = (
            df.groupby("brand")
            .agg(**aggregation)
            .reset_index()
        )

        numeric_summary_columns = (
            brand_summary
            .select_dtypes(
                include="number"
            )
            .columns
        )

        brand_summary[
            numeric_summary_columns
        ] = brand_summary[
            numeric_summary_columns
        ].round(2)

        brand_summary.to_csv(
            EDA_BRAND_OUTPUT,
            index=False
        )

        print("\n" + "=" * 70)
        print("BRAND ANALYSIS SUMMARY")
        print("=" * 70)

        print(
            brand_summary
            .to_string(index=False)
        )


# ============================================================
# 21. SPECIFICATION ANALYSIS
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


    plt.figure(
        figsize=(10, 6)
    )

    specification_summary.plot(
        kind="bar"
    )

    plt.title(
        "Average Mobile Product Specification Ratings"
    )

    plt.xlabel(
        "Specification"
    )

    plt.ylabel(
        "Average Rating"
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()
    plt.show()


    specification_table = pd.DataFrame({

        "Feature":
            specification_summary.index,

        "Average_Rating":
            specification_summary.values.round(2)
    })


    specification_table.to_csv(
        EDA_SPEC_OUTPUT,
        index=False
    )


# ============================================================
# 22. ENGAGEMENT ANALYSIS
# ============================================================

if all(
    column in df.columns
    for column in [
        "brand",
        "engagement_score"
    ]
):

    brand_engagement = (
        df.groupby("brand")[
            "engagement_score"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    print("\n" + "=" * 70)
    print("BRAND-WISE ENGAGEMENT SCORE")
    print("=" * 70)

    print(
        brand_engagement.round(2)
    )


# ============================================================
# 23. SAVE TOP PRODUCT TABLE
# ============================================================

if all(
    column in df.columns
    for column in [
        "brand",
        "model",
        "rating"
    ]
):

    top_products = (
        df.groupby(
            ["brand", "model"]
        )
        .agg(
            Average_Rating=(
                "rating",
                "mean"
            )
        )
        .reset_index()
        .sort_values(
            "Average_Rating",
            ascending=False
        )
        .head(10)
    )

    top_products[
        "Average_Rating"
    ] = top_products[
        "Average_Rating"
    ].round(2)

    top_products.to_csv(
        "eda_top_rated_products.csv",
        index=False
    )


# ============================================================
# 24. SAVE LOW PRODUCT TABLE
# ============================================================

if all(
    column in df.columns
    for column in [
        "brand",
        "model",
        "rating"
    ]
):

    low_products = (
        df.groupby(
            ["brand", "model"]
        )
        .agg(
            Average_Rating=(
                "rating",
                "mean"
            )
        )
        .reset_index()
        .sort_values(
            "Average_Rating",
            ascending=True
        )
        .head(10)
    )

    low_products[
        "Average_Rating"
    ] = low_products[
        "Average_Rating"
    ].round(2)

    low_products.to_csv(
        "eda_low_rated_products.csv",
        index=False
    )


# ============================================================
# 25. COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nEDA output files created:")

print(
    f"1. {EDA_BRAND_OUTPUT}"
)

print(
    f"2. {EDA_SPEC_OUTPUT}"
)

print(
    f"3. {EDA_PRODUCT_OUTPUT}"
)

if "country" in df.columns:

    print(
        f"4. {EDA_COUNTRY_OUTPUT}"
    )

print(
    "5. eda_top_rated_products.csv"
)

print(
    "6. eda_low_rated_products.csv"
)

print("\nNext Step:")
print(
    "Run Step4_Clustering.py"
)

print("=" * 70)
