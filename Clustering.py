# ============================================================
# Step4_Clustering.py
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. LOAD CLEANED DATASET

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

print("\nInput File:", input_file)
print("Original Dataset Shape:", df.shape)

# 2. CHECK REQUIRED COLUMNS

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

    "engagement_score"

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

        + "\n\n"
        "Please run Step2_Data_Preprocessing.py again "
        "and make sure engagement_score is created."

    )

# 3. SELECT CLUSTERING FEATURES

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

print("\n" + "=" * 70)
print("CLUSTERING FEATURES")
print("=" * 70)

for number, feature in enumerate(
    clustering_features,
    start=1
):

    print(
        f"{number}. {feature}"
    )

print(
    "\nTotal Clustering Features:",
    len(clustering_features)
)

# 4. CONVERT CLUSTERING FEATURES TO NUMERIC

for column in clustering_features:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# 5. HANDLE INFINITE VALUES

df[clustering_features] = (

    df[clustering_features]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )

)

# 6. CHECK MISSING VALUES

print("\n" + "=" * 70)
print("MISSING VALUES BEFORE CLUSTERING")
print("=" * 70)

missing_values = (

    df[clustering_features]
    .isnull()
    .sum()

)

print(missing_values)

total_missing = int(
    missing_values.sum()
)

if total_missing > 0:

    raise ValueError(

        "\nERROR: Missing values are still present "
        "in clustering features.\n\n"

        "Please run Step2_Data_Preprocessing.py again "
        "before running Step4.\n\n"

        "Missing values:\n"
        + missing_values[
            missing_values > 0
        ].to_string()

    )

print(
    "\nNo missing values found."
)

# 7. CLEAN PRODUCT IDENTIFIERS

df["brand"] = (

    df["brand"]
    .astype(str)
    .str.strip()

)

df["model"] = (

    df["model"]
    .astype(str)
    .str.strip()

)

if df["brand"].eq("").any():

    raise ValueError(
        "\nERROR: Empty brand values found."
    )

if df["model"].eq("").any():

    raise ValueError(
        "\nERROR: Empty model values found."
    )

# 8. PRODUCT-LEVEL AGGREGATION

aggregation_dict = {

    feature: "mean"
    for feature in clustering_features

}

product_df = (

    df
    .groupby(
        ["brand", "model"],
        as_index=False
    )
    .agg(
        aggregation_dict
    )

)

# Create review count separately

review_counts = (

    df
    .groupby(
        ["brand", "model"]
    )
    .size()
    .reset_index(
        name="review_count"
    )

)

product_df = product_df.merge(

    review_counts,

    on=[
        "brand",
        "model"
    ],

    how="left"

)

print("\n" + "=" * 70)
print("PRODUCT-LEVEL DATA")
print("=" * 70)

print(
    "Original Records      :",
    len(df)
)

print(
    "Unique Products       :",
    len(product_df)
)

print(
    "Product Dataset Shape :",
    product_df.shape
)

# 9. VALIDATE PRODUCT-LEVEL DATA

duplicate_products = (

    product_df
    .duplicated(
        subset=[
            "brand",
            "model"
        ]
    )
    .sum()

)

if duplicate_products > 0:

    raise ValueError(
        "\nERROR: Duplicate product records found."
    )


if product_df["review_count"].isnull().any():

    raise ValueError(
        "\nERROR: review_count contains missing values."
    )

# 10. CREATE FEATURE MATRIX

X = (

    product_df[
        clustering_features
    ]
    .copy()

)

print(
    "\nFeature Matrix Shape:",
    X.shape
)

print(
    "Number of Features:",
    X.shape[1]
)

# 11. STANDARDIZE FEATURES

print("\n" + "=" * 70)
print("FEATURE STANDARDIZATION")
print("=" * 70)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)

print(
    "\nFeatures standardized successfully."
)

print(
    "Scaled Matrix Shape:",
    X_scaled.shape
)

# 12. VALIDATE SCALER FEATURE COUNT

if scaler.n_features_in_ != len(clustering_features):

    raise ValueError(

        "\nERROR: Scaler feature count mismatch.\n"

        f"Expected: {len(clustering_features)}\n"

        f"Actual: {scaler.n_features_in_}"

    )

print(
    "Scaler Feature Count:",
    scaler.n_features_in_
)

# 13. SAVE SCALER

scaler_file = "mobile_scaler.pkl"

joblib.dump(
    scaler,
    scaler_file
)


print(
    "\nScaler saved as:",
    scaler_file
)

# 14. ELBOW METHOD - INERTIA CHECK

print("\n" + "=" * 70)
print("ELBOW METHOD - INERTIA CHECK")
print("=" * 70)

inertia_values = []

max_k = min(
    8,
    len(product_df) - 1
)

k_values = list(
    range(
        2,
        max_k + 1
    )
)

if len(k_values) < 2:

    raise ValueError(
        "\nERROR: Not enough products for elbow analysis."
    )

for k in k_values:

    model = KMeans(

        n_clusters=k,

        random_state=42,

        n_init=10

    )

    model.fit(
        X_scaled
    )

    inertia_values.append(
        model.inertia_
    )

# 15. PRINT INERTIA VALUES

print("\nInertia Values:")

for k, inertia in zip(

    k_values,

    inertia_values

):

    print(

        f"K = {k}  -->  "
        f"Inertia = {inertia:.2f}"

    )

# 16. SAVE ELBOW RESULTS

elbow_results = pd.DataFrame({

    "K": k_values,

    "Inertia": inertia_values

})

elbow_file = "elbow_inertia.csv"

elbow_results.to_csv(

    elbow_file,

    index=False

)

print(
    "\nElbow results saved as:",
    elbow_file
)

# 17. ELBOW CURVE

plt.figure(
    figsize=(9, 6)
)

plt.plot(

    k_values,

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
    k_values
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

elbow_plot_file = "elbow_curve.png"

plt.savefig(

    elbow_plot_file,

    dpi=300,

    bbox_inches="tight"

)

plt.show()

plt.close()

print(
    "Elbow plot saved as:",
    elbow_plot_file
)

# 18. K-MEANS CLUSTERING

print("\n" + "=" * 70)
print("K-MEANS CLUSTERING")
print("=" * 70)

NUMBER_OF_CLUSTERS = 3

if NUMBER_OF_CLUSTERS > len(product_df):

    raise ValueError(
        "\nERROR: Number of clusters cannot exceed "
        "number of products."
    )

kmeans = KMeans(

    n_clusters=NUMBER_OF_CLUSTERS,

    random_state=42,

    n_init=10

)

product_df["Cluster"] = (

    kmeans.fit_predict(
        X_scaled
    )

)

print(
    "\nK-Means clustering completed successfully."
)

print(
    "Number of Clusters:",
    NUMBER_OF_CLUSTERS
)

# 19. SAVE K-MEANS MODEL

model_file = "mobile_kmeans_model.pkl"

joblib.dump(

    kmeans,

    model_file

)

print(
    "K-Means model saved as:",
    model_file
)

# 20. SILHOUETTE SCORE

print("\n" + "=" * 70)
print("SILHOUETTE SCORE")
print("=" * 70)

evaluation_sample_size = len(X_scaled)

silhouette = silhouette_score(

    X_scaled,

    product_df["Cluster"],

    metric="euclidean"

)

print(

    "\nSilhouette Score:",

    round(
        silhouette,
        4
    )

)

print(

    "Evaluation Sample Size:",

    evaluation_sample_size

)

# 21. CLUSTER DISTRIBUTION

print("\n" + "=" * 70)
print("CLUSTER DISTRIBUTION")
print("=" * 70)

cluster_counts = (

    product_df[
        "Cluster"
    ]
    .value_counts()
    .sort_index()

)

print(
    "\nNumber of Products:"
)

print(
    cluster_counts
)

cluster_percentage = (

    product_df[
        "Cluster"
    ]
    .value_counts(
        normalize=True
    )
    .sort_index()
    .mul(100)
    .round(2)

)

print(
    "\nCluster Percentage:"
)


print(
    cluster_percentage
)

# 22. PRICE VS RATING CLUSTER VISUALIZATION

plt.figure(
    figsize=(9, 6)
)

for cluster in sorted(
    product_df["Cluster"].unique()
):

    cluster_data = product_df[
        product_df["Cluster"] == cluster
    ]


    plt.scatter(

        cluster_data["price_usd"],

        cluster_data["rating"],

        s=70,

        alpha=0.8,

        label=f"Cluster {cluster}"

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

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

cluster_plot_file = "cluster_segmentation.png"

plt.savefig(

    cluster_plot_file,

    dpi=300,

    bbox_inches="tight"

)

plt.show()

plt.close()

print(
    "Cluster plot saved as:",
    cluster_plot_file
)

# 23. CLUSTER-WISE PRODUCT PROFILE

print("\n" + "=" * 70)
print("CLUSTER-WISE PRODUCT PROFILE")
print("=" * 70)

cluster_analysis = (

    product_df
    .groupby(
        "Cluster"
    )[
        clustering_features
    ]
    .mean()
    .round(2)

)

print(
    cluster_analysis
)

# 24. CREATE CLUSTER SUMMARY

cluster_profile = (

    product_df
    .groupby(
        "Cluster"
    )
    .agg(

        Product_Count=(
            "model",
            "count"
        ),

        Review_Count=(
            "review_count",
            "sum"
        )

    )
    .reset_index()

)

cluster_profile["Percentage"] = (

    cluster_profile[
        "Product_Count"
    ]

    /

    len(product_df)

    *

    100

).round(2)

cluster_profile = (

    cluster_profile
    .merge(

        cluster_analysis.reset_index(),

        on="Cluster",

        how="left"

    )

)

# 25. ASSIGN BUSINESS SEGMENT NAMES

print("\n" + "=" * 70)
print("PRODUCT SEGMENT LABELING")
print("=" * 70)

sorted_clusters = (

    cluster_profile
    .sort_values(
        "price_usd"
    )[
        "Cluster"
    ]
    .tolist()

)

segment_names = {

    sorted_clusters[0]: "Budget",

    sorted_clusters[1]: "Mid-Range",

    sorted_clusters[2]: "Premium"

}

product_df["Segment"] = (

    product_df[
        "Cluster"
    ]
    .map(
        segment_names
    )

)

cluster_profile["Segment"] = (

    cluster_profile[
        "Cluster"
    ]
    .map(
        segment_names
    )

)

# 26. DISPLAY SEGMENT PROFILES

print("\n" + "=" * 70)
print("SEGMENT PROFILE SUMMARY")
print("=" * 70)

print(

    cluster_profile
    .sort_values(
        "price_usd"
    )
    .to_string(
        index=False
    )

)

# 27. BUSINESS INTERPRETATION

print("\n" + "=" * 70)
print("SEGMENT INTERPRETATION")
print("=" * 70)

for _, row in (

    cluster_profile
    .sort_values(
        "price_usd"
    )
    .iterrows()

):

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

        f"Average Engagement  : "
        f"{row['engagement_score']:.2f}"

    )

    print(

        f"Product Count       : "
        f"{int(row['Product_Count'])}"

    )

    print(

        f"Review Count        : "
        f"{int(row['Review_Count'])}"

    )

    print(

        f"Cluster Percentage  : "
        f"{row['Percentage']:.2f}%"

    )

# 28. BRAND DISTRIBUTION BY CLUSTER

print("\n" + "=" * 70)
print("BRAND DISTRIBUTION BY CLUSTER")
print("=" * 70)

brand_cluster = pd.crosstab(

    product_df[
        "Cluster"
    ],

    product_df[
        "brand"
    ]

)

print(
    brand_cluster
)

# 29. SEGMENT DISTRIBUTION VISUALIZATION

segment_counts = (

    product_df[
        "Segment"
    ]
    .value_counts()
    .reindex(
        [
            "Budget",
            "Mid-Range",
            "Premium"
        ]
    )
    .dropna()

)

plt.figure(
    figsize=(9, 6)
)

plt.bar(

    segment_counts.index,

    segment_counts.values
    
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

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

segment_plot_file = "segment_distribution.png"


plt.savefig(

    segment_plot_file,

    dpi=300,

    bbox_inches="tight"

)

plt.show()

plt.close()

print(
    "Segment distribution plot saved as:",
    segment_plot_file
)

# 30. SAVE CLUSTER PROFILE

profile_file = "cluster_profile_summary.csv"

cluster_profile.to_csv(

    profile_file,

    index=False

)

print(

    "\nCluster profile saved as:",

    profile_file

)

# 31. SAVE CLUSTERED PRODUCT DATASET

output_file = "clustered_mobile_products.csv"

# Final column order for compatibility

final_columns = [

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
    "Segment"

]

product_df = product_df[
    final_columns
]

product_df.to_csv(

    output_file,

    index=False

)

print(

    "Clustered product dataset saved as:",

    output_file

)

# 32. SAVE CLUSTERING METADATA

metadata = {

    "number_of_clusters":
        NUMBER_OF_CLUSTERS,

    "clustering_features":
        clustering_features,

    "silhouette_score":
        float(silhouette),

    "random_state":
        42,

    "product_count":
        int(len(product_df)),

    "review_count":
        int(len(df))

}

metadata_file = "clustering_metadata.pkl"

joblib.dump(

    metadata,

    metadata_file

)

print(

    "Clustering metadata saved as:",

    metadata_file

)

# 33. FINAL VALIDATION

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

expected_output_columns = [

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
    "Segment"

]

missing_output_columns = [

    column
    for column in expected_output_columns
    if column not in product_df.columns

]

if missing_output_columns:

    raise ValueError(

        "\nERROR: Final output is missing columns:\n"

        + "\n".join(
            f"- {column}"
            for column in missing_output_columns
        )

    )

if len(product_df) != product_df[
    ["brand", "model"]
].drop_duplicates().shape[0]:

    raise ValueError(
        "\nERROR: Multiple rows exist for the same product."
    )

if product_df["Cluster"].nunique() != NUMBER_OF_CLUSTERS:

    raise ValueError(
        "\nERROR: Expected exactly 3 clusters."
    )

if product_df["Segment"].isnull().any():

    raise ValueError(
        "\nERROR: Some products do not have a Segment label."
    )

print(
    "\nFinal Output Columns:"
)


for number, column in enumerate(

    product_df.columns,

    start=1

):

    print(
        f"{number}. {column}"
    )


print(
    "\nFinal Product Dataset Shape:",
    product_df.shape
)

# 34. FINAL SUMMARY

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
    "3. elbow_inertia.csv"
)

print(
    "4. elbow_curve.png"
)

print(
    "5. cluster_segmentation.png"
)

print(
    "6. segment_distribution.png"
)

print(
    "7. mobile_scaler.pkl"
)

print(
    "8. mobile_kmeans_model.pkl"
)

print(
    "9. clustering_metadata.pkl"
)

print("\nMethod:")
print(
    "K-Means Clustering"
)

print(
    "\nNumber of Clusters:",
    NUMBER_OF_CLUSTERS
)

print(
    "\nElbow Point:",
    "K = 3"
)

print(
    "\nSilhouette Score:",
    round(
        silhouette,
        4
    )
)

print("\nSegment Labels:")

for cluster, segment in (

    sorted(
        segment_names.items()
    )

):

    print(

        f"Cluster {cluster} -> "
        f"{segment}"

    )


# 35. INERTIA SUMMARY

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

print("\n" + "=" * 70)
print("STEP 4 COMPLETED")
print("=" * 70)
