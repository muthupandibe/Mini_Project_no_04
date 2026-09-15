# ============================================================
# Step4_Clustering.py
# MOBILE PRODUCT CLUSTERING / SEGMENTATION
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

df.columns = (
    df.columns
    .str.strip()
)


print("\n" + "=" * 70)
print("MOBILE PRODUCT CLUSTERING / SEGMENTATION")
print("=" * 70)

print(
    "\nInput File:",
    input_file
)

print(
    "Original Dataset Shape:",
    df.shape
)


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
        "Please make sure Step2 creates "
        "engagement_score."

    )


# ============================================================
# 3. SELECT CLUSTERING FEATURES
# ============================================================

# IMPORTANT:
# This exact feature list and order MUST also be used
# in Step5_Recommendation.py.

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

df[clustering_features] = (

    df[
        clustering_features
    ]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )

)


# ============================================================
# 6. HANDLE MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES BEFORE CLUSTERING")
print("=" * 70)


missing_values = (

    df[
        clustering_features
    ]
    .isnull()
    .sum()

)


print(
    missing_values
)


for column in clustering_features:

    median_value = (

        df[column]
        .median()

    )


    if pd.isna(median_value):

        median_value = 0


    df[column] = (

        df[column]
        .fillna(median_value)

    )


print(
    "\nMissing values handled successfully."
)


print(
    "\nMissing values after handling:"
)


print(
    df[
        clustering_features
    ]
    .isnull()
    .sum()
)


# ============================================================
# 7. PRODUCT-LEVEL AGGREGATION
# ============================================================

# Multiple reviews may belong to the same mobile model.
# Therefore, we aggregate the review-level data into
# one product-level record for each brand + model.

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


# ============================================================
# 8. CREATE FEATURE MATRIX
# ============================================================

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


# ============================================================
# 9. STANDARDIZE FEATURES
# ============================================================

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


# ============================================================
# 10. SAVE SCALER
# ============================================================

scaler_file = "mobile_scaler.pkl"


joblib.dump(
    scaler,
    scaler_file
)


print(
    "\nScaler saved as:",
    scaler_file
)


# ============================================================
# 11. ELBOW METHOD - INERTIA CHECK
# ============================================================

print("\n" + "=" * 70)
print("ELBOW METHOD - INERTIA CHECK")
print("=" * 70)


inertia_values = []


k_values = range(
    2,
    9
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
# 13. SAVE ELBOW RESULTS
# ============================================================

elbow_results = pd.DataFrame({

    "K": list(k_values),

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


# ============================================================
# 14. ELBOW CURVE
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
# 15. K-MEANS CLUSTERING
# ============================================================

print("\n" + "=" * 70)
print("K-MEANS CLUSTERING")
print("=" * 70)


# IMPORTANT:
# Based on the project's confirmed elbow point,
# the final number of clusters is K = 3.

NUMBER_OF_CLUSTERS = 3


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


# ============================================================
# 16. SAVE K-MEANS MODEL
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
# 17. SILHOUETTE SCORE
# ============================================================

print("\n" + "=" * 70)
print("SILHOUETTE SCORE")
print("=" * 70)


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

    round(
        silhouette,
        4
    )

)


print(

    "Evaluation Sample Size:",

    sample_size

)


# ============================================================
# 18. CLUSTER DISTRIBUTION
# ============================================================

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


# ============================================================
# 19. PRICE VS RATING CLUSTER VISUALIZATION
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
# 20. CLUSTER-WISE PRODUCT PROFILE
# ============================================================

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


# ============================================================
# 21. CREATE CLUSTER SUMMARY
# ============================================================

cluster_profile = (

    product_df
    .groupby(
        "Cluster"
    )
    .size()
    .reset_index(
        name="Product_Count"
    )

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

        on="Cluster"

    )

)


# ============================================================
# 22. ASSIGN BUSINESS SEGMENT NAMES
# ============================================================

print("\n" + "=" * 70)
print("PRODUCT SEGMENT LABELING")
print("=" * 70)


# Sort clusters according to average price.
# K-Means cluster numbers are arbitrary.

sorted_clusters = (

    cluster_profile
    .sort_values(
        "price_usd"
    )[
        "Cluster"
    ]
    .tolist()

)


segment_names = {}


if len(sorted_clusters) == 3:

    segment_names = {

        sorted_clusters[0]: "Budget",

        sorted_clusters[1]: "Mid-Range",

        sorted_clusters[2]: "Premium"

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


# ============================================================
# 23. DISPLAY SEGMENT PROFILES
# ============================================================

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


# ============================================================
# 24. BUSINESS INTERPRETATION
# ============================================================

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

        f"Cluster Percentage  : "
        f"{row['Percentage']:.2f}%"

    )


# ============================================================
# 25. BRAND DISTRIBUTION BY CLUSTER
# ============================================================

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


# ============================================================
# 26. SEGMENT DISTRIBUTION VISUALIZATION
# ============================================================

segment_counts = (

    product_df[
        "Segment"
    ]
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
# 27. SAVE CLUSTER PROFILE
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
# 28. SAVE CLUSTERED PRODUCT DATASET
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
# 29. SAVE CLUSTERING METADATA
# ============================================================

metadata = {

    "number_of_clusters":
        NUMBER_OF_CLUSTERS,

    "clustering_features":
        clustering_features,

    "silhouette_score":
        float(silhouette),

    "random_state":
        42

}


metadata_file = (
    "clustering_metadata.pkl"
)


joblib.dump(

    metadata,

    metadata_file

)


print(

    "Clustering metadata saved as:",

    metadata_file

)


# ============================================================
# 30. FINAL SUMMARY
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
    "3. elbow_inertia.csv"
)


print(
    "4. mobile_scaler.pkl"
)


print(
    "5. mobile_kmeans_model.pkl"
)


print(
    "6. clustering_metadata.pkl"
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


# ============================================================
# 31. INERTIA SUMMARY
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


print("\n" + "=" * 70)
print("STEP 4 COMPLETED")
print("=" * 70)
