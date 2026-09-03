# ============================================================
# FILE: Step7_Insights_Reporting.py
# PROJECT: Mobile Product Segmentation and Recommendation System
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. FILE PATHS
# ============================================================

cleaned_file = "cleaned_mobile_reviews.csv"

clustered_file = "clustered_mobile_reviews.csv"

output_folder = "insights"

os.makedirs(
    output_folder,
    exist_ok=True
)


# ============================================================
# 2. CHECK INPUT FILES
# ============================================================

if not os.path.exists(cleaned_file):

    raise FileNotFoundError(
        f"\nERROR: {cleaned_file} not found.\n"
        "Please run Step2_Data_Preprocessing.py first."
    )


if not os.path.exists(clustered_file):

    raise FileNotFoundError(
        f"\nERROR: {clustered_file} not found.\n"
        "Please run Step4_Clustering.py first."
    )


# ============================================================
# 3. LOAD DATA
# ============================================================

df = pd.read_csv(cleaned_file)

clustered_df = pd.read_csv(clustered_file)


df.columns = df.columns.str.strip()

clustered_df.columns = (
    clustered_df.columns.str.strip()
)


print("\n" + "=" * 75)
print("MOBILE PRODUCT INSIGHTS & REPORTING")
print("=" * 75)


print(
    "\nCleaned Dataset Shape:",
    df.shape
)

print(
    "Clustered Dataset Shape:",
    clustered_df.shape
)


# ============================================================
# 4. REQUIRED COLUMNS
# ============================================================

required_columns = [

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


missing_columns = [

    column

    for column in required_columns

    if column not in clustered_df.columns

]


if missing_columns:

    raise ValueError(

        "\nERROR: Required columns are missing:\n"

        + "\n".join(

            f"- {column}"

            for column in missing_columns

        )

    )


# ============================================================
# 5. NUMERIC CONVERSION
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


for column in numeric_columns:

    clustered_df[column] = pd.to_numeric(

        clustered_df[column],

        errors="coerce"

    )


# ============================================================
# 6. HANDLE MISSING VALUES
# ============================================================

for column in numeric_columns:

    median_value = (
        clustered_df[column].median()
    )

    if pd.isna(median_value):

        median_value = 0

    clustered_df[column] = (

        clustered_df[column]

        .fillna(median_value)

    )


# ============================================================
# INSIGHT 1
# PRODUCT SEGMENTATION / CLUSTER ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("1. PRODUCT SEGMENTATION ANALYSIS")
print("=" * 75)


if "Cluster" not in clustered_df.columns:

    raise ValueError(
        "\nERROR: Cluster column not found."
        "\nPlease run Step4_Clustering.py first."
    )


cluster_profile = (

    clustered_df

    .groupby("Cluster")[numeric_columns]

    .mean()

    .round(2)

)


cluster_counts = (

    clustered_df

    .groupby("Cluster")

    .size()

    .reset_index(

        name="Product_Count"

    )

)


cluster_profile = (

    cluster_profile

    .reset_index()

    .merge(

        cluster_counts,

        on="Cluster"

    )

)


cluster_profile["Percentage"] = (

    cluster_profile["Product_Count"]

    / len(clustered_df)

    * 100

).round(2)


# ------------------------------------------------------------
# Add Segment Names
# ------------------------------------------------------------

if "Segment" in clustered_df.columns:

    segment_mapping = (

        clustered_df[

            ["Cluster", "Segment"]

        ]

        .drop_duplicates()

    )


    cluster_profile = (

        cluster_profile

        .merge(

            segment_mapping,

            on="Cluster",

            how="left"

        )

    )


print("\nCluster Profile:")

print(

    cluster_profile

    .to_string(index=False)

)


# ------------------------------------------------------------
# Save Cluster Analysis
# ------------------------------------------------------------

cluster_profile_file = os.path.join(

    output_folder,

    "cluster_wise_analysis.csv"

)


cluster_profile.to_csv(

    cluster_profile_file,

    index=False

)


# ============================================================
# INSIGHT 2
# HIGH-PERFORMING PRODUCTS
# ============================================================

print("\n" + "=" * 75)
print("2. HIGH-PERFORMING PRODUCTS")
print("=" * 75)


# ------------------------------------------------------------
# Create Overall Performance Score
# ------------------------------------------------------------

performance_columns = [

    "rating",

    "battery_life_rating",

    "camera_rating",

    "performance_rating",

    "design_rating",

    "display_rating"

]


clustered_df["Overall_Performance_Score"] = (

    clustered_df[performance_columns]

    .mean(axis=1)

)


high_performing = (

    clustered_df

    .sort_values(

        "Overall_Performance_Score",

        ascending=False

    )

    .head(10)

)


high_performing_display = high_performing[

    [

        "brand",

        "model",

        "price_usd",

        "rating",

        "camera_rating",

        "performance_rating",

        "battery_life_rating",

        "Overall_Performance_Score"

    ]

].copy()


high_performing_display = (

    high_performing_display

    .round(2)

)


print("\nTop 10 High-Performing Products:")

print(

    high_performing_display

    .to_string(index=False)

)


high_file = os.path.join(

    output_folder,

    "high_performing_products.csv"

)


high_performing_display.to_csv(

    high_file,

    index=False

)


# ============================================================
# INSIGHT 3
# LOW-PERFORMING PRODUCTS
# ============================================================

print("\n" + "=" * 75)
print("3. LOW-PERFORMING PRODUCTS")
print("=" * 75)


low_performing = (

    clustered_df

    .sort_values(

        "Overall_Performance_Score",

        ascending=True

    )

    .head(10)

)


low_performing_display = low_performing[

    [

        "brand",

        "model",

        "price_usd",

        "rating",

        "camera_rating",

        "performance_rating",

        "battery_life_rating",

        "Overall_Performance_Score"

    ]

].copy()


low_performing_display = (

    low_performing_display

    .round(2)

)


print("\nBottom 10 Low-Performing Products:")

print(

    low_performing_display

    .to_string(index=False)

)


low_file = os.path.join(

    output_folder,

    "low_performing_products.csv"

)


low_performing_display.to_csv(

    low_file,

    index=False

)


# ============================================================
# INSIGHT 4
# PRICE VS PERFORMANCE
# ============================================================

print("\n" + "=" * 75)
print("4. PRICE VS PERFORMANCE ANALYSIS")
print("=" * 75)


price_performance_correlation = (

    clustered_df[

        [

            "price_usd",

            "Overall_Performance_Score"

        ]

    ]

    .corr()

    .iloc[0, 1]

)


print(

    "\nPrice vs Overall Performance "
    "Correlation:",

    round(

        price_performance_correlation,

        4

    )

)


# ------------------------------------------------------------
# Price vs Performance Scatter Plot
# ------------------------------------------------------------

plt.figure(

    figsize=(9, 6)

)


if "Segment" in clustered_df.columns:

    sns.scatterplot(

        data=clustered_df,

        x="price_usd",

        y="Overall_Performance_Score",

        hue="Segment",

        alpha=0.7

    )

else:

    sns.scatterplot(

        data=clustered_df,

        x="price_usd",

        y="Overall_Performance_Score",

        hue="Cluster",

        alpha=0.7

    )


plt.title(

    "Price vs Overall Product Performance"

)


plt.xlabel(

    "Price (USD)"

)


plt.ylabel(

    "Overall Performance Score"

)


plt.tight_layout()


price_plot_file = os.path.join(

    output_folder,

    "price_vs_performance.png"

)


plt.savefig(

    price_plot_file,

    dpi=300,

    bbox_inches="tight"

)


plt.show()


# ============================================================
# PRICE RANGE ANALYSIS
# ============================================================

clustered_df["Price_Range"] = pd.cut(

    clustered_df["price_usd"],

    bins=[

        -float("inf"),

        200,

        400,

        700,

        float("inf")

    ],

    labels=[

        "Budget",

        "Mid-Range",

        "Upper Mid-Range",

        "Premium"

    ]

)


price_range_analysis = (

    clustered_df

    .groupby(

        "Price_Range",

        observed=False

    )

    [

        [

            "price_usd",

            "rating",

            "Overall_Performance_Score"

        ]

    ]

    .mean()

    .round(2)

)


print("\nPrice Range Performance:")

print(price_range_analysis)


price_range_file = os.path.join(

    output_folder,

    "price_range_performance.csv"

)


price_range_analysis.to_csv(

    price_range_file

)


# ============================================================
# INSIGHT 5
# CUSTOMER PREFERENCE PATTERNS
# ============================================================

print("\n" + "=" * 75)
print("5. CUSTOMER PREFERENCE PATTERNS")
print("=" * 75)


preference_features = [

    "rating",

    "battery_life_rating",

    "camera_rating",

    "performance_rating",

    "design_rating",

    "display_rating"

]


preference_analysis = (

    clustered_df[

        preference_features

    ]

    .mean()

    .sort_values(

        ascending=False

    )

    .round(2)

)


print(

    "\nAverage Feature Ratings:"

)


print(

    preference_analysis

)


# ------------------------------------------------------------
# Feature Preference Ranking
# ------------------------------------------------------------

preference_ranking = (

    preference_analysis

    .reset_index()

)


preference_ranking.columns = [

    "Feature",

    "Average_Rating"

]


preference_ranking["Rank"] = (

    preference_ranking[

        "Average_Rating"

    ]

    .rank(

        ascending=False,

        method="dense"

    )

    .astype(int)

)


preference_ranking = (

    preference_ranking

    .sort_values("Rank")

)


print(

    "\nFeature Preference Ranking:"

)


print(

    preference_ranking

    .to_string(index=False)

)


preference_file = os.path.join(

    output_folder,

    "customer_preference_patterns.csv"

)


preference_ranking.to_csv(

    preference_file,

    index=False

)


# ============================================================
# INSIGHT 6
# BRAND ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("6. BRAND PERFORMANCE ANALYSIS")
print("=" * 75)


brand_analysis = (

    clustered_df

    .groupby("brand")

    .agg(

        Product_Count=(

            "model",

            "count"

        ),

        Average_Price=(

            "price_usd",

            "mean"

        ),

        Average_Rating=(

            "rating",

            "mean"

        ),

        Average_Performance=(

            "Overall_Performance_Score",

            "mean"

        )

    )

    .round(2)

    .sort_values(

        "Average_Performance",

        ascending=False

    )

)


print(

    "\nBrand Performance Summary:"

)


print(

    brand_analysis

    .to_string()

)


brand_file = os.path.join(

    output_folder,

    "brand_performance_analysis.csv"

)


brand_analysis.to_csv(

    brand_file

)


# ============================================================
# INSIGHT 7
# DATA-DRIVEN BUSINESS INSIGHTS
# ============================================================

print("\n" + "=" * 75)
print("7. DATA-DRIVEN BUSINESS INSIGHTS")
print("=" * 75)


insights = []


# ------------------------------------------------------------
# Segmentation Insight
# ------------------------------------------------------------

largest_cluster = (

    cluster_profile

    .sort_values(

        "Product_Count",

        ascending=False

    )

    .iloc[0]

)


if "Segment" in cluster_profile.columns:

    largest_segment = (

        largest_cluster["Segment"]

    )

else:

    largest_segment = (

        f"Cluster {int(largest_cluster['Cluster'])}"

    )


insight_1 = (

    f"The largest product segment is "

    f"{largest_segment}, containing "

    f"{int(largest_cluster['Product_Count'])} "

    f"products "

    f"({largest_cluster['Percentage']:.2f}% of the dataset)."

)


print("\n1.", insight_1)

insights.append(insight_1)


# ------------------------------------------------------------
# High Performance Insight
# ------------------------------------------------------------

best_product = (

    high_performing_display.iloc[0]

)


insight_2 = (

    f"The highest-performing product in the "

    f"analysis is "

    f"{best_product['brand']} "

    f"{best_product['model']} "

    f"with an overall performance score of "

    f"{best_product['Overall_Performance_Score']:.2f}."

)


print("\n2.", insight_2)

insights.append(insight_2)


# ------------------------------------------------------------
# Low Performance Insight
# ------------------------------------------------------------

worst_product = (

    low_performing_display.iloc[0]

)


insight_3 = (

    f"The lowest-performing product is "

    f"{worst_product['brand']} "

    f"{worst_product['model']} "

    f"with an overall performance score of "

    f"{worst_product['Overall_Performance_Score']:.2f}."

)


print("\n3.", insight_3)

insights.append(insight_3)


# ------------------------------------------------------------
# Price Performance Insight
# ------------------------------------------------------------

if price_performance_correlation > 0.5:

    price_message = (

        "There is a strong positive relationship "

        "between product price and overall performance."

    )


elif price_performance_correlation > 0.2:

    price_message = (

        "There is a moderate positive relationship "

        "between product price and overall performance."

    )


elif price_performance_correlation > -0.2:

    price_message = (

        "There is a weak relationship between "

        "product price and overall performance."

    )


else:

    price_message = (

        "There is a negative relationship between "

        "product price and overall performance."

    )


print("\n4.", price_message)

insights.append(price_message)


# ------------------------------------------------------------
# Preference Insight
# ------------------------------------------------------------

top_preference = (

    preference_ranking.iloc[0]

)


insight_5 = (

    f"Among the measured product attributes, "

    f"{top_preference['Feature']} has the highest "

    f"average rating of "

    f"{top_preference['Average_Rating']:.2f}."

)


print("\n5.", insight_5)

insights.append(insight_5)


# ------------------------------------------------------------
# Brand Insight
# ------------------------------------------------------------

best_brand = (

    brand_analysis

    .sort_values(

        "Average_Performance",

        ascending=False

    )

    .iloc[0]

)


best_brand_name = (

    brand_analysis

    .sort_values(

        "Average_Performance",

        ascending=False

    )

    .index[0]

)


insight_6 = (

    f"{best_brand_name} has the highest average "

    f"overall performance score among the brands "

    f"in the dataset, at "

    f"{best_brand['Average_Performance']:.2f}."

)


print("\n6.", insight_6)

insights.append(insight_6)


# ============================================================
# 8. DATA-DRIVEN DECISION MAKING
# ============================================================

print("\n" + "=" * 75)
print("8. DATA-DRIVEN DECISION MAKING")
print("=" * 75)


decision_recommendations = [

    "Use product segmentation to identify Budget, Mid-Range, Upper Mid-Range and Premium market groups.",

    "Prioritize high-performing product features when evaluating new or existing mobile products.",

    "Use price-versus-performance analysis to identify products offering stronger value for money.",

    "Use rating patterns to understand which product attributes receive stronger customer evaluations.",

    "Use brand-level performance comparisons to support product portfolio and competitive analysis.",

    "Use the recommendation system to help customers discover products with similar characteristics."

]


for number, recommendation in enumerate(

    decision_recommendations,

    start=1

):

    print(

        f"\n{number}. {recommendation}"

    )


# ============================================================
# 9. SAVE TEXT REPORT
# ============================================================

report_file = os.path.join(

    output_folder,

    "mobile_product_insights_report.txt"

)


with open(

    report_file,

    "w",

    encoding="utf-8"

) as file:

    file.write(

        "MOBILE PRODUCT SEGMENTATION & "
        "RECOMMENDATION SYSTEM\n"

    )

    file.write(

        "=" * 75 + "\n\n"

    )


    file.write(

        "1. PRODUCT SEGMENTATION\n"

    )

    file.write(

        "-" * 40 + "\n"

    )


    for insight in insights:

        file.write(

            f"- {insight}\n"

        )


    file.write(

        "\n\n2. DATA-DRIVEN DECISION MAKING\n"

    )

    file.write(

        "-" * 40 + "\n"

    )


    for number, recommendation in enumerate(

        decision_recommendations,

        start=1

    ):

        file.write(

            f"{number}. {recommendation}\n"

        )


    file.write(

        "\n\n3. IMPORTANT NOTE\n"

    )

    file.write(

        "-" * 40 + "\n"

    )


    file.write(

        "Customer preference patterns are inferred "
        "from product ratings and specifications. "
        "The dataset does not contain direct "
        "individual customer purchase-history or "
        "preference data.\n"

    )


# ============================================================
# 10. COMPLETION
# ============================================================

print("\n" + "=" * 75)
print("INSIGHTS & REPORTING COMPLETED SUCCESSFULLY")
print("=" * 75)


print("\nOutput Folder:")

print(

    f"{output_folder}/"

)


print("\nGenerated Files:")

print(

    "1. cluster_wise_analysis.csv"

)

print(

    "2. high_performing_products.csv"

)

print(

    "3. low_performing_products.csv"

)

print(

    "4. price_range_performance.csv"

)

print(

    "5. customer_preference_patterns.csv"

)

print(

    "6. brand_performance_analysis.csv"

)

print(

    "7. price_vs_performance.png"

)

print(

    "8. mobile_product_insights_report.txt"

)


print("\n" + "=" * 75)