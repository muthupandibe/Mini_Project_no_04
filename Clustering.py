# ============================================================
# Step4_Clustering.py
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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

df.columns = df.columns.str.strip()

print("\n" + "=" * 70)
print("MOBILE PRODUCT CLUSTERING / SEGMENTATION")
print("=" * 70)

print("\nOriginal Dataset Shape:", df.shape)


# ============================================================
# 2. CHECK REQUIRED COLUMNS
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


# ============================================================
# 3. SELECT CLUSTERING FEATURES
# ============================================================

clustering_features = [
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
    clustering_features.append("engagement_score")


print("\n" + "=" * 70)
print("CLUSTERING FEATURES")
print("=" * 70)

for feature in clustering_features:
    print("-", feature)


# ============================================================
# 4. CONVERT FEATURES TO NUMERIC
# ============================================================

for column in clustering_features:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# 5. HANDLE INFINITE VALUES
# ============================================================

df[clustering_features] = df[
    clustering_features
].replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# 6. HANDLE MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES BEFORE CLUSTERING")
print("=" * 70)

print(
    df[clustering_features].isnull().sum()
)


for column in clustering_features:

    median_value = df[column].median()

    if pd.isna(median_value):
        median_value = 0

    df[column] = df[column].fillna(
        median_value
    )


print("\nMissing values handled successfully.")


# ============================================================
# 7. PRODUCT-LEVEL AGGREGATION
# ============================================================
# If multiple reviews exist for the same mobile model,
# calculate average values for product-level clustering.

aggregation_dict = {
    feature: "mean"
    for feature in clustering_features
}

product_df = (
    df.groupby(
        ["brand", "model"],
        as_index=False
    )
    .agg(aggregation_dict)
)


print("\n" + "=" * 70)
print("PRODUCT-LEVEL DATA")
print("=" * 70)

print(
    "Original Records :",
    len(df)
)

print(
    "Unique Products   :",
    len(product_df)
)

print(
    "Product Dataset Shape:",
    product_df.shape
)


# ============================================================
# 8. CREATE FEATURE MATRIX
# ============================================================

X = product_df[
    clustering_features
].copy()


# ============================================================
# 9. STANDARDIZE FEATURES
# ============================================================

print("\n" + "=" * 70)
print("FEATURE STANDARDIZATION")
print("=" * 70)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print(
    "\nFeatures standardized successfully."
)


# ============================================================
# 10. SAVE SCALER
# ============================================================

scaler_file = "mobile_scaler.pkl"

joblib.dump(
    scaler,
    scaler_file
)

print(
    "Scaler saved as:",
    scaler_file
)


# ============================================================
# 11. ELBOW METHOD - INERTIA CHECK
# ============================================================

print("\n" + "=" * 70)
print("ELBOW METHOD - INERTIA CHECK")
print("=" * 70)

inertia_values = []

k_values = range(2, 9)


for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertia_values.append(
        model.inertia_
    )


# ============================================================
# 12. PRINT INERTIA VALUES
# ============================================================

print("\nInertia Values:")

for k, inertia in zip(
    k_values,
    inertia_values
):

    print(
        f"K = {k}  -->  "
        f"Inertia = {inertia:.2f}"
    )


# ============================================================
# 13. ELBOW CURVE
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    list(k_values),
    inertia_values,
    marker="o"
)

plt.title(
    "Elbow Method for Optimal Number of Clusters"
)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Inertia"
)

plt.xticks(
    list(k_values)
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()
plt.show()


# ============================================================
# 14. K-MEANS CLUSTERING
# ============================================================

print("\n" + "=" * 70)
print("K-MEANS CLUSTERING")
print("=" * 70)


# Project requirement
NUMBER_OF_CLUSTERS = 4


kmeans = KMeans(
    n_clusters=NUMBER_OF_CLUSTERS,
    random_state=42,
    n_init=10
)


product_df["Cluster"] = kmeans.fit_predict(
    X_scaled
)


print(
    "\nK-Means clustering completed successfully."
)

print(
    "Number of Clusters:",
    NUMBER_OF_CLUSTERS
)


# ============================================================
# 15. SAVE K-MEANS MODEL
# ============================================================

model_file = "mobile_kmeans_model.pkl"

joblib.dump(
    kmeans,
    model_file
)

print(
    "K-Means model saved as:",
    model_file
)


# ============================================================
# 16. SILHOUETTE SCORE
# ============================================================

print("\n" + "=" * 70)
print("SILHOUETTE SCORE")
print("=" * 70)


# Sample data for faster calculation
sample_size = min(
    10000,
    len(X_scaled)
)


silhouette = silhouette_score(
    X_scaled,
    product_df["Cluster"],
    sample_size=sample_size,
    random_state=42
)


print(
    "\nSilhouette Score:",
    round(silhouette, 4)
)

print(
    "Evaluation Sample Size:",
    sample_size
)


# ============================================================
# 17. CLUSTER DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CLUSTER DISTRIBUTION")
print("=" * 70)


cluster_counts = (
    product_df["Cluster"]
    .value_counts()
    .sort_index()
)


print("\nNumber of Products:")

print(
    cluster_counts
)


cluster_percentage = (
    product_df["Cluster"]
    .value_counts(
        normalize=True
    )
    .sort_index()
    .mul(100)
    .round(2)
)


print("\nCluster Percentage:")

print(
    cluster_percentage
)


# ============================================================
# 18. PRICE VS RATING CLUSTER VISUALIZATION
# ============================================================

plt.figure(
    figsize=(9, 6)
)


sns.scatterplot(
    data=product_df,
    x="price_usd",
    y="rating",
    hue="Cluster",
    palette="Set1",
    s=70,
    alpha=0.8
)


plt.title(
    "Mobile Product Segmentation using K-Means"
)

plt.xlabel(
    "Price (USD)"
)

plt.ylabel(
    "Rating"
)

plt.legend(
    title="Cluster"
)

plt.tight_layout()
plt.show()


# ============================================================
# 19. CLUSTER-WISE PRODUCT PROFILE
# ============================================================

print("\n" + "=" * 70)
print("CLUSTER-WISE PRODUCT PROFILE")
print("=" * 70)


cluster_analysis = (
    product_df
    .groupby("Cluster")[
        clustering_features
    ]
    .mean()
    .round(2)
)


print(
    cluster_analysis
)


# ============================================================
# 20. CREATE CLUSTER SUMMARY
# ============================================================

cluster_profile = (
    product_df
    .groupby("Cluster")
    .size()
    .reset_index(
        name="Product_Count"
    )
)


cluster_profile["Percentage"] = (
    cluster_profile["Product_Count"]
    / len(product_df)
    * 100
).round(2)


cluster_profile = cluster_profile.merge(
    cluster_analysis.reset_index(),
    on="Cluster"
)


# ============================================================
# 21. ASSIGN SEGMENT NAMES
# ============================================================

print("\n" + "=" * 70)
print("PRODUCT SEGMENT LABELING")
print("=" * 70)


sorted_clusters = (
    cluster_profile
    .sort_values(
        "price_usd"
    )["Cluster"]
    .tolist()
)


segment_names = {}


if len(sorted_clusters) == 4:

    segment_names = {

        sorted_clusters[0]: "Budget",

        sorted_clusters[1]: "Mid-Range",

        sorted_clusters[2]: "Upper Mid-Range",

        sorted_clusters[3]: "Premium"
    }


else:

    for position, cluster in enumerate(
        sorted_clusters,
        start=1
    ):

        segment_names[
            cluster
        ] = f"Segment {position}"


product_df["Segment"] = (
    product_df["Cluster"]
    .map(segment_names)
)


cluster_profile["Segment"] = (
    cluster_profile["Cluster"]
    .map(segment_names)
)


# ============================================================
# 22. DISPLAY SEGMENT PROFILES
# ============================================================

print("\n" + "=" * 70)
print("SEGMENT PROFILE SUMMARY")
print("=" * 70)


print(
    cluster_profile.to_string(
        index=False
    )
)


# ============================================================
# 23. BUSINESS INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("SEGMENT INTERPRETATION")
print("=" * 70)


for _, row in cluster_profile.iterrows():

    print(
        f"\nCluster {int(row['Cluster'])}"
        f" - {row['Segment']}"
    )

    print(
        f"Average Price       : "
        f"${row['price_usd']:.2f}"
    )

    print(
        f"Average Rating      : "
        f"{row['rating']:.2f}"
    )

    print(
        f"Average Battery     : "
        f"{row['battery_life_rating']:.2f}"
    )

    print(
        f"Average Camera      : "
        f"{row['camera_rating']:.2f}"
    )

    print(
        f"Average Performance : "
        f"{row['performance_rating']:.2f}"
    )

    print(
        f"Average Design      : "
        f"{row['design_rating']:.2f}"
    )

    print(
        f"Average Display     : "
        f"{row['display_rating']:.2f}"
    )

    print(
        f"Product Count       : "
        f"{int(row['Product_Count'])}"
    )

    print(
        f"Cluster Percentage  : "
        f"{row['Percentage']:.2f}%"
    )


# ============================================================
# 24. BRAND DISTRIBUTION BY CLUSTER
# ============================================================

print("\n" + "=" * 70)
print("BRAND DISTRIBUTION BY CLUSTER")
print("=" * 70)


brand_cluster = pd.crosstab(
    product_df["Cluster"],
    product_df["brand"]
)


print(
    brand_cluster
)


# ============================================================
# 25. SEGMENT DISTRIBUTION VISUALIZATION
# ============================================================

segment_counts = (
    product_df["Segment"]
    .value_counts()
)


plt.figure(
    figsize=(9, 6)
)


segment_counts.plot(
    kind="bar"
)


plt.title(
    "Mobile Product Segment Distribution"
)

plt.xlabel(
    "Product Segment"
)

plt.ylabel(
    "Number of Products"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()
plt.show()


# ============================================================
# 26. SAVE CLUSTER PROFILE
# ============================================================

profile_file = (
    "cluster_profile_summary.csv"
)


cluster_profile.to_csv(
    profile_file,
    index=False
)


print(
    "\nCluster profile saved as:",
    profile_file
)


# ============================================================
# 27. SAVE CLUSTERED PRODUCT DATASET
# ============================================================

output_file = (
    "clustered_mobile_products.csv"
)


product_df.to_csv(
    output_file,
    index=False
)


print(
    "Clustered product dataset saved as:",
    output_file
)


# ============================================================
# 28. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CLUSTERING ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)


print("\nOutputs Created:")

print(
    "1. clustered_mobile_products.csv"
)

print(
    "2. cluster_profile_summary.csv"
)

print(
    "3. mobile_scaler.pkl"
)

print(
    "4. mobile_kmeans_model.pkl"
)


print("\nMethod:")
print("K-Means Clustering")


print(
    "\nNumber of Clusters:",
    NUMBER_OF_CLUSTERS
)


print(
    "\nSilhouette Score:",
    round(silhouette, 4)
)


print("\nSegment Labels:")

for cluster, segment in segment_names.items():

    print(
        f"Cluster {cluster} -> {segment}"
    )


# ============================================================
# 29. INERTIA SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL INERTIA SUMMARY")
print("=" * 70)


for k, inertia in zip(
    k_values,
    inertia_values
):

    print(
        f"K = {k} | "
        f"Inertia = {inertia:.2f}"
    )


print("\nCheck the Elbow graph to determine the")
print("most appropriate number of clusters.")
