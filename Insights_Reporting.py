# ============================================================
# STEP 6: INSIGHTS & REPORTING
# ============================================================

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. PATH SETUP

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "clustered_mobile_products.csv"
RECOMMENDATION_FILE = (
    BASE_DIR / "recommendations" / "all_product_recommendations.csv"
)

OUTPUT_DIR = BASE_DIR / "insights"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("STEP 6: INSIGHTS & REPORTING")
print("=" * 70)

# 2. LOAD CLUSTERED PRODUCT DATA

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Required file not found:\n{INPUT_FILE}\n\n"
        "Please run Step 4 before running Step 6."
    )

df = pd.read_csv(INPUT_FILE)

print("\nClustered product data loaded successfully.")
print("Shape:", df.shape)

# 3. REQUIRED COLUMNS

required_columns = [
    "brand",
    "model",
    "price_usd",
    "rating",
    "battery_life_rating",
    "camera_rating",
    "performance_rating",
    "design_rating",
    "display_rating",
    "engagement_score",
    "review_count",
    "Cluster",
    "Segment",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Required columns are missing from clustered_mobile_products.csv:\n"
        + ", ".join(missing_columns)
        + "\n\nPlease rerun Step 2 and Step 4."
    )

# 4. DATA VALIDATION

numeric_columns = [
    "price_usd",
    "rating",
    "battery_life_rating",
    "camera_rating",
    "performance_rating",
    "design_rating",
    "display_rating",
    "engagement_score",
    "review_count",
    "Cluster",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

if df[numeric_columns].isnull().any().any():
    print("\nMissing numeric values detected.")
    print(df[numeric_columns].isnull().sum())

    raise ValueError(
        "\nNumeric missing values detected.\n"
        "Please rerun Step 2 and Step 4."
    )

if np.isinf(df[numeric_columns].to_numpy()).any():
    raise ValueError(
        "Infinite values detected in the dataset. "
        "Please rerun Step 2 and Step 4."
    )

# 5. CHECK PRODUCT UNIQUENESS

duplicate_products = df.duplicated(
    subset=["brand", "model"]
).sum()

if duplicate_products > 0:
    raise ValueError(
        f"Found {duplicate_products} duplicate brand-model products.\n"
        "Step 4 should contain one row per product."
    )

print("Unique products:", len(df))

# 6. CHECK CLUSTERS

unique_clusters = sorted(df["Cluster"].unique())

print("\nClusters found:", unique_clusters)

if len(unique_clusters) != 3:
    raise ValueError(
        f"Expected exactly 3 clusters, but found {len(unique_clusters)}."
    )

# 7. PERFORMANCE SCORE

performance_features = [
    "rating",
    "battery_life_rating",
    "camera_rating",
    "performance_rating",
    "design_rating",
    "display_rating",
]

df["performance_score"] = df[performance_features].mean(axis=1)

# 8. PRODUCT SEGMENTATION – CLUSTER-WISE ANALYSIS

cluster_summary = (
    df.groupby(["Cluster", "Segment"])
    .agg(
        Product_Count=("model", "count"),
        Average_Price_USD=("price_usd", "mean"),
        Average_Rating=("rating", "mean"),
        Average_Battery=("battery_life_rating", "mean"),
        Average_Camera=("camera_rating", "mean"),
        Average_Performance=("performance_rating", "mean"),
        Average_Design=("design_rating", "mean"),
        Average_Display=("display_rating", "mean"),
        Average_Engagement=("engagement_score", "mean"),
        Total_Reviews=("review_count", "sum"),
        Average_Performance_Score=("performance_score", "mean"),
    )
    .reset_index()
)

cluster_summary["Percentage"] = (
    cluster_summary["Product_Count"]
    / len(df)
    * 100
)

cluster_summary = cluster_summary.sort_values("Cluster")

cluster_summary.to_csv(
    OUTPUT_DIR / "cluster_wise_analysis.csv",
    index=False
)

print("\n" + "=" * 70)
print("CLUSTER-WISE ANALYSIS")
print("=" * 70)

print(
    cluster_summary[
        [
            "Cluster",
            "Segment",
            "Product_Count",
            "Percentage",
            "Average_Price_USD",
            "Average_Rating",
            "Average_Performance_Score",
        ]
    ].round(2)
)

# 9. HIGH-PERFORMING PRODUCTS

high_performing = (
    df.sort_values(
        by="performance_score",
        ascending=False
    )
    [
        [
            "brand",
            "model",
            "Segment",
            "Cluster",
            "price_usd",
            "rating",
            "battery_life_rating",
            "camera_rating",
            "performance_rating",
            "design_rating",
            "display_rating",
            "performance_score",
        ]
    ]
    .head(10)
)

high_performing.to_csv(
    OUTPUT_DIR / "high_performing_products.csv",
    index=False
)

print("\n" + "=" * 70)
print("TOP 10 HIGH-PERFORMING PRODUCTS")
print("=" * 70)

print(high_performing.round(2).to_string(index=False))

# 10. LOW-PERFORMING PRODUCTS

low_performing = (
    df.sort_values(
        by="performance_score",
        ascending=True
    )
    [
        [
            "brand",
            "model",
            "Segment",
            "Cluster",
            "price_usd",
            "rating",
            "battery_life_rating",
            "camera_rating",
            "performance_rating",
            "design_rating",
            "display_rating",
            "performance_score",
        ]
    ]
    .head(10)
)

low_performing.to_csv(
    OUTPUT_DIR / "low_performing_products.csv",
    index=False
)

print("\n" + "=" * 70)
print("BOTTOM 10 LOW-PERFORMING PRODUCTS")
print("=" * 70)

print(low_performing.round(2).to_string(index=False))

# 11. PRICE VS PERFORMANCE ANALYSIS

price_performance_correlation = df[
    ["price_usd", "performance_score"]
].corr().iloc[0, 1]

print("\n" + "=" * 70)
print("PRICE VS PERFORMANCE")
print("=" * 70)

print(
    f"Price vs Performance Correlation: "
    f"{price_performance_correlation:.4f}"
)

# 12. PRICE RANGE ANALYSIS

def price_band(price):
    if price < 400:
        return "Below $400"
    elif price <= 700:
        return "$400 - $700"
    else:
        return "Above $700"


df["Price_Band"] = df["price_usd"].apply(price_band)

price_range_summary = (
    df.groupby("Price_Band", observed=False)
    .agg(
        Product_Count=("model", "count"),
        Average_Price_USD=("price_usd", "mean"),
        Average_Rating=("rating", "mean"),
        Average_Performance_Score=("performance_score", "mean"),
        Average_Engagement=("engagement_score", "mean"),
    )
    .reset_index()
)

price_range_summary.to_csv(
    OUTPUT_DIR / "price_range_performance.csv",
    index=False
)

print("\nPrice range analysis:")
print(price_range_summary.round(2).to_string(index=False))

# 13. CUSTOMER PREFERENCE / ATTRIBUTE PATTERN ANALYSIS

attribute_columns = [
    "price_usd",
    "battery_life_rating",
    "camera_rating",
    "performance_rating",
    "design_rating",
    "display_rating",
    "engagement_score",
]

attribute_patterns = []

for column in attribute_columns:

    rating_corr = df[column].corr(df["rating"])

    performance_corr = df[column].corr(
        df["performance_score"]
    )

    attribute_patterns.append(
        {
            "Attribute": column,
            "Correlation_with_Rating": rating_corr,
            "Correlation_with_Performance": performance_corr,
            "Average_Value": df[column].mean(),
        }
    )

attribute_patterns = pd.DataFrame(attribute_patterns)

attribute_patterns.to_csv(
    OUTPUT_DIR / "product_attribute_patterns.csv",
    index=False
)

print("\n" + "=" * 70)
print("PRODUCT ATTRIBUTE PATTERNS")
print("=" * 70)

print(attribute_patterns.round(4).to_string(index=False))

# 14. BRAND PERFORMANCE ANALYSIS

brand_analysis = (
    df.groupby("brand")
    .agg(
        Product_Count=("model", "count"),
        Average_Price_USD=("price_usd", "mean"),
        Average_Rating=("rating", "mean"),
        Average_Performance_Score=("performance_score", "mean"),
        Average_Engagement=("engagement_score", "mean"),
        Total_Reviews=("review_count", "sum"),
    )
    .reset_index()
)

brand_analysis = brand_analysis.sort_values(
    "Average_Performance_Score",
    ascending=False
)

brand_analysis.to_csv(
    OUTPUT_DIR / "brand_performance_analysis.csv",
    index=False
)

print("\n" + "=" * 70)
print("BRAND PERFORMANCE ANALYSIS")
print("=" * 70)

print(brand_analysis.round(2).to_string(index=False))

# 15. CLUSTER DISTRIBUTION PLOT

plt.figure(figsize=(8, 5))

cluster_plot_data = (
    df.groupby(["Cluster", "Segment"])
    .size()
    .reset_index(name="Product_Count")
)

labels = (
    cluster_plot_data["Segment"]
    + " (C"
    + cluster_plot_data["Cluster"].astype(str)
    + ")"
)

plt.bar(
    labels,
    cluster_plot_data["Product_Count"]
)

plt.title("Product Distribution Across Clusters")
plt.xlabel("Segment")
plt.ylabel("Number of Products")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "cluster_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# 16. PRICE VS PERFORMANCE PLOT

plt.figure(figsize=(9, 6))

for segment in sorted(df["Segment"].unique()):

    segment_data = df[df["Segment"] == segment]

    plt.scatter(
        segment_data["price_usd"],
        segment_data["performance_score"],
        label=segment,
        s=70
    )

    for _, row in segment_data.iterrows():

        plt.annotate(
            row["model"],
            (
                row["price_usd"],
                row["performance_score"]
            ),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points"
        )

plt.title("Price vs Product Performance")
plt.xlabel("Price (USD)")
plt.ylabel("Performance Score")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "price_vs_performance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# 17. RATING BY SEGMENT

segment_rating = (
    df.groupby("Segment")
    .agg(
        Average_Rating=("rating", "mean"),
        Average_Performance=("performance_score", "mean"),
        Product_Count=("model", "count"),
    )
    .reset_index()
)

segment_rating.to_csv(
    OUTPUT_DIR / "segment_rating_analysis.csv",
    index=False
)

# 18. RECOMMENDATION SYSTEM VALIDATION

recommendation_exists = RECOMMENDATION_FILE.exists()

recommendation_summary = ""

if recommendation_exists:

    recommendations = pd.read_csv(
        RECOMMENDATION_FILE
    )

    print("\n" + "=" * 70)
    print("RECOMMENDATION VALIDATION")
    print("=" * 70)

    print(
        "Recommendation rows:",
        len(recommendations)
    )

    expected_columns = [
        "Selected_Brand",
        "Selected_Model",
        "Recommended_Brand",
        "Recommended_Model",
        "Similarity_Score",
        "Recommendation_Rank",
    ]

    missing_recommendation_columns = [
        col
        for col in expected_columns
        if col not in recommendations.columns
    ]

    if missing_recommendation_columns:

        recommendation_summary = (
            "Recommendation file exists, but required "
            "validation columns are missing."
        )

    else:

        # Check self recommendations
        self_recommendations = (
            (
                recommendations["Selected_Brand"]
                == recommendations["Recommended_Brand"]
            )
            &
            (
                recommendations["Selected_Model"]
                == recommendations["Recommended_Model"]
            )
        ).sum()

        # Check similarity scores
        invalid_similarity = (
            recommendations["Similarity_Score"]
            .isnull()
            .sum()
        )

        # Check recommendation ranks
        invalid_ranks = (
            recommendations["Recommendation_Rank"]
            .isnull()
            .sum()
        )

        # Number of recommendations per selected product
        recommendation_counts = (
            recommendations
            .groupby(
                ["Selected_Brand", "Selected_Model"]
            )
            .size()
        )

        max_recommendations = (
            recommendation_counts.max()
            if len(recommendation_counts) > 0
            else 0
        )

        print(
            "Self-recommendations:",
            self_recommendations
        )

        print(
            "Missing similarity scores:",
            invalid_similarity
        )

        print(
            "Missing recommendation ranks:",
            invalid_ranks
        )

        print(
            "Maximum recommendations per product:",
            max_recommendations
        )

        if (
            self_recommendations == 0
            and invalid_similarity == 0
            and invalid_ranks == 0
            and max_recommendations <= 5
        ):
            recommendation_summary = (
                "Recommendation structure passed validation: "
                "no self-recommendations, valid similarity scores, "
                "valid ranks, and maximum 5 recommendations per product."
            )

        else:
            recommendation_summary = (
                "Recommendation file requires further validation."
            )

else:

    recommendation_summary = (
        "Recommendation file was not found. "
        "Run Step 5 before validating recommendations."
    )

print(recommendation_summary)

# 19. AUTOMATIC INSIGHT GENERATION

highest_performance_product = df.loc[
    df["performance_score"].idxmax()
]

lowest_performance_product = df.loc[
    df["performance_score"].idxmin()
]

highest_price_product = df.loc[
    df["price_usd"].idxmax()
]

lowest_price_product = df.loc[
    df["price_usd"].idxmin()
]

largest_cluster = (
    cluster_summary.loc[
        cluster_summary["Product_Count"].idxmax()
    ]
)

smallest_cluster = (
    cluster_summary.loc[
        cluster_summary["Product_Count"].idxmin()
    ]
)

best_price_band = (
    price_range_summary.loc[
        price_range_summary["Average_Performance_Score"].idxmax()
    ]
)

# 20. TEXT REPORT

report_file = OUTPUT_DIR / "mobile_product_insights_report.txt"

with open(report_file, "w", encoding="utf-8") as report:

    report.write("=" * 75 + "\n")
    report.write("MOBILE PRODUCT SEGMENTATION & RECOMMENDATION SYSTEM\n")
    report.write("INSIGHTS & REPORTING\n")
    report.write("=" * 75 + "\n\n")

    # Dataset

    report.write("1. DATASET OVERVIEW\n")
    report.write("-" * 75 + "\n")

    report.write(
        f"Number of products analyzed: {len(df)}\n"
    )

    report.write(
        f"Number of clusters: {len(unique_clusters)}\n"
    )

    report.write(
        f"Average product price: "
        f"${df['price_usd'].mean():.2f}\n"
    )

    report.write(
        f"Average rating: "
        f"{df['rating'].mean():.2f}\n\n"
    )


    # Segmentation

    report.write("2. PRODUCT SEGMENTATION\n")
    report.write("-" * 75 + "\n")

    for _, row in cluster_summary.iterrows():

        report.write(
            f"Cluster {int(row['Cluster'])} - "
            f"{row['Segment']}\n"
        )

        report.write(
            f"  Products: {int(row['Product_Count'])}\n"
        )

        report.write(
            f"  Percentage: {row['Percentage']:.2f}%\n"
        )

        report.write(
            f"  Average Price: "
            f"${row['Average_Price_USD']:.2f}\n"
        )

        report.write(
            f"  Average Rating: "
            f"{row['Average_Rating']:.2f}\n"
        )

        report.write(
            f"  Average Performance Score: "
            f"{row['Average_Performance_Score']:.2f}\n"
        )

        report.write("\n")

    report.write(
        f"The largest segment is "
        f"{largest_cluster['Segment']} "
        f"with {int(largest_cluster['Product_Count'])} products.\n"
    )

    report.write(
        f"The smallest segment is "
        f"{smallest_cluster['Segment']} "
        f"with {int(smallest_cluster['Product_Count'])} products.\n\n"
    )

    # High performance

    report.write("3. HIGH-PERFORMING PRODUCTS\n")
    report.write("-" * 75 + "\n")

    report.write(
        f"Highest performance score: "
        f"{highest_performance_product['brand']} "
        f"{highest_performance_product['model']} "
        f"({highest_performance_product['performance_score']:.2f})\n\n"
    )

    report.write(
        "The high-performing products are identified using "
        "the average of rating and five specification ratings.\n\n"
    )

    # Low performance

    report.write("4. LOW-PERFORMING PRODUCTS\n")
    report.write("-" * 75 + "\n")

    report.write(
        f"Lowest performance score: "
        f"{lowest_performance_product['brand']} "
        f"{lowest_performance_product['model']} "
        f"({lowest_performance_product['performance_score']:.2f})\n\n"
    )

    # Price-performance

    report.write("5. PRICE VS PERFORMANCE\n")
    report.write("-" * 75 + "\n")

    report.write(
        f"Price-performance correlation: "
        f"{price_performance_correlation:.4f}\n\n"
    )

    report.write(
        "Correlation measures statistical association and "
        "does not establish causation.\n\n"
    )

    report.write(
        f"Lowest-priced product: "
        f"{lowest_price_product['brand']} "
        f"{lowest_price_product['model']} "
        f"(${lowest_price_product['price_usd']:.2f})\n"
    )

    report.write(
        f"Highest-priced product: "
        f"{highest_price_product['brand']} "
        f"{highest_price_product['model']} "
        f"(${highest_price_product['price_usd']:.2f})\n\n"
    )

    report.write(
        f"Price band with highest average performance score: "
        f"{best_price_band['Price_Band']} "
        f"({best_price_band['Average_Performance_Score']:.2f})\n\n"
    )

    # Attribute patterns

    report.write("6. PRODUCT ATTRIBUTE PATTERNS\n")
    report.write("-" * 75 + "\n")

    strongest_rating_attribute = (
        attribute_patterns
        .dropna(subset=["Correlation_with_Rating"])
        .loc[
            attribute_patterns[
                "Correlation_with_Rating"
            ].abs().idxmax()
        ]
    )

    report.write(
        f"Attribute with strongest observed association "
        f"with rating: "
        f"{strongest_rating_attribute['Attribute']}\n"
    )

    report.write(
        f"Correlation with rating: "
        f"{strongest_rating_attribute['Correlation_with_Rating']:.4f}\n\n"
    )

    report.write(
        "These are product-level attribute patterns in the "
        "available dataset. They should not be interpreted as "
        "direct proof of customer preferences because the dataset "
        "does not contain explicit purchase-choice or preference labels.\n\n"
    )

    # Recommendations

    report.write("7. RECOMMENDATION SYSTEM VALIDATION\n")
    report.write("-" * 75 + "\n")

    report.write(
        recommendation_summary + "\n\n"
    )

    # Decision support

    report.write("8. DATA-DRIVEN DECISION SUPPORT\n")
    report.write("-" * 75 + "\n")

    report.write(
        "The analysis can support product-level decisions by "
        "identifying price-based segments, comparing product "
        "performance characteristics, examining rating patterns, "
        "and generating similarity-based recommendations.\n\n"
    )

    report.write(
        "Possible uses include:\n"
        "- Product portfolio analysis\n"
        "- Segment-level product comparison\n"
        "- Similar-product recommendation\n"
        "- Product specification analysis\n"
        "- Pricing and performance exploration\n"
        "- Identification of products requiring further review\n\n"
    )

    # Limitations

    report.write("9. LIMITATIONS\n")
    report.write("-" * 75 + "\n")

    report.write(
        "1. The performance score is a simple average of rating "
        "and five specification ratings.\n"
    )

    report.write(
        "2. Correlation does not prove causation.\n"
    )

    report.write(
        "3. Product-level aggregation may hide individual "
        "customer differences.\n"
    )

    report.write(
        "4. Customer preference cannot be directly established "
        "without purchase, click, choice, or preference labels.\n"
    )

    report.write(
        "5. The price bands used for analysis are descriptive "
        "and are separate from the K-Means segments.\n"
    )

    report.write(
        "6. Recommendation relevance is structurally validated, "
        "but there is no ground-truth user feedback dataset "
        "for offline accuracy evaluation.\n"
    )


# 21. FINAL OUTPUT SUMMARY

print("\n" + "=" * 70)
print("STEP 6 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nInsights files created:")

output_files = [
    "cluster_wise_analysis.csv",
    "high_performing_products.csv",
    "low_performing_products.csv",
    "price_range_performance.csv",
    "product_attribute_patterns.csv",
    "brand_performance_analysis.csv",
    "segment_rating_analysis.csv",
    "cluster_distribution.png",
    "price_vs_performance.png",
    "mobile_product_insights_report.txt",
]

for filename in output_files:

    file_path = OUTPUT_DIR / filename

    if file_path.exists():
        print("✓", filename)
    else:
        print("✗", filename)

print("\nOutput folder:")
print(OUTPUT_DIR)

print("\nStep 6 workflow:")
print("Step 2 → Cleaned Data")
print("Step 3 → EDA")
print("Step 4 → Clustering")
print("Step 5 → Recommendation")
print("Step 6 → Insights & Reporting")

print("\n" + "=" * 70)
