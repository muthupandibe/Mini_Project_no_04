# ============================================================
# Step5_Recommendation.py
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

# 1. FILE PATHS

input_file = "clustered_mobile_products.csv"

scaler_file = "mobile_scaler.pkl"

metadata_file = "clustering_metadata.pkl"

output_folder = "recommendations"

output_file = os.path.join(
    output_folder,
    "all_product_recommendations.csv"
)

recommendation_metadata_file = os.path.join(
    output_folder,
    "recommendation_metadata.json"
)


os.makedirs(
    output_folder,
    exist_ok=True
)

# 2. CHECK REQUIRED FILES

if not os.path.exists(input_file):

    raise FileNotFoundError(

        f"\nERROR: {input_file} not found.\n"

        "Please run Step4_Clustering.py first."

    )


if not os.path.exists(scaler_file):

    raise FileNotFoundError(

        f"\nERROR: {scaler_file} not found.\n"

        "Please run Step4_Clustering.py first."

    )

# 3. LOAD DATA

df = pd.read_csv(
    input_file
)


df.columns = (
    df.columns
    .str.strip()
)


scaler = joblib.load(
    scaler_file
)


print("\n" + "=" * 70)
print("MOBILE PRODUCT RECOMMENDATION SYSTEM")
print("=" * 70)


print(
    "\nInput File:",
    input_file
)


print(
    "Dataset Shape:",
    df.shape
)

# 4. EXACT SAME FEATURES AS STEP 4

PRODUCT_FEATURES = [

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
print("RECOMMENDATION FEATURES")
print("=" * 70)


for number, feature in enumerate(

    PRODUCT_FEATURES,

    start=1

):

    print(
        f"{number}. {feature}"
    )


print(
    "\nTotal Recommendation Features:",
    len(PRODUCT_FEATURES)
)

# 5. CHECK REQUIRED COLUMNS

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
    "Segment"

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

        "Please run the corrected Step4_Clustering.py again."

    )

# 6. VALIDATE PRODUCT DATA

duplicate_products = (

    df
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

        "\nERROR: Duplicate brand + model products found.\n"

        "Step4 should contain one row per product."

    )


if len(df) < 2:

    raise ValueError(

        "\nERROR: At least two products are required "
        "for recommendations."

    )

# 7. CONVERT FEATURES TO NUMERIC

for column in PRODUCT_FEATURES:

    df[column] = pd.to_numeric(

        df[column],

        errors="coerce"

    )

# 8. CHECK MISSING VALUES

missing_values = (

    df[
        PRODUCT_FEATURES
    ]
    .isnull()
    .sum()

)


if missing_values.sum() > 0:

    print(
        "\nMissing values found:"
    )

    print(
        missing_values[
            missing_values > 0
        ]
    )

    raise ValueError(

        "\nERROR: Recommendation features contain "
        "missing values.\n\n"

        "Do NOT manually fill them in Step5.\n"

        "Please rerun Step2 and Step4."

    )

# 9. CHECK INFINITE VALUES

if np.isinf(

    df[
        PRODUCT_FEATURES
    ]
    .to_numpy()

).any():

    raise ValueError(

        "\nERROR: Infinite values found "
        "in recommendation features."

    )

# 10. VALIDATE SCALER

if not hasattr(

    scaler,

    "n_features_in_"

):

    raise ValueError(

        "\nERROR: The saved scaler does not contain "
        "feature information.\n\n"

        "Please rerun Step4_Clustering.py."

    )


if scaler.n_features_in_ != len(

    PRODUCT_FEATURES

):

    raise ValueError(

        "\nERROR: Scaler feature count mismatch.\n\n"

        f"Scaler expects: "
        f"{scaler.n_features_in_}\n"

        f"Recommendation features: "
        f"{len(PRODUCT_FEATURES)}\n\n"

        "Please rerun Step4_Clustering.py."

    )


print(
    "\nScaler Feature Count:",
    scaler.n_features_in_
)

# 11. VALIDATE FEATURE ORDER

if hasattr(

    scaler,

    "feature_names_in_"

):

    scaler_features = list(

        scaler.feature_names_in_

    )


    if scaler_features != PRODUCT_FEATURES:

        raise ValueError(

            "\nERROR: Feature order mismatch between "
            "Step4 and Step5.\n\n"

            f"Step4 Scaler Features:\n"
            f"{scaler_features}\n\n"

            f"Step5 Features:\n"
            f"{PRODUCT_FEATURES}\n\n"

            "Please rerun Step4 using the corrected code."

        )


    print(
        "Feature Order Validation: PASSED"
    )


else:

    print(
        "Feature Order Validation: "
        "Feature names unavailable in saved scaler."
    )

# 12. CREATE FEATURE MATRIX

X = (

    df[
        PRODUCT_FEATURES
    ]
    .copy()

)


print("\n" + "=" * 70)
print("FEATURE MATRIX")
print("=" * 70)


print(
    "Feature Matrix Shape:",
    X.shape
)

# 13. APPLY SCALER

X_scaled = scaler.transform(
    X
)


print(
    "\nScaled Feature Matrix Shape:",
    X_scaled.shape
)

# 14. CALCULATE COSINE SIMILARITY

print("\n" + "=" * 70)
print("COSINE SIMILARITY")
print("=" * 70)


similarity_matrix = cosine_similarity(
    X_scaled
)


print(
    "\nCosine similarity matrix created."
)


print(
    "Similarity Matrix Shape:",
    similarity_matrix.shape
)

# 15. RECOMMENDATION SETTINGS

TOP_N = 5


print(
    "\nNumber of Recommendations per Product:",
    TOP_N
)

# 16. CREATE PRODUCT INDEX

product_keys = (

    df["brand"].astype(str)

    + "|||"

    + df["model"].astype(str)

)


product_index = {

    key: index

    for index, key in enumerate(
        product_keys
    )

}

# 17. GENERATE RECOMMENDATIONS

recommendation_rows = []


for selected_index in range(
    len(df)
):

    selected_product = df.iloc[
        selected_index
    ]


    selected_brand = (
        selected_product["brand"]
    )


    selected_model = (
        selected_product["model"]
    )


    selected_cluster = (
        selected_product["Cluster"]
    )


    selected_segment = (
        selected_product["Segment"]
    )

    # Get all similarity scores

    similarity_scores = (

        similarity_matrix[
            selected_index
        ]
        .copy()

    )

    # Exclude selected product itself

    similarity_scores[
        selected_index
    ] = -1

    # Create candidate dataframe

    candidates = df.copy()


    candidates[
        "Similarity_Score"
    ] = similarity_scores


    candidates[
        "_Index"
    ] = np.arange(
        len(df)
    )

    # Exclude selected product

    candidates = candidates[
        candidates["_Index"]
        != selected_index
    ]

    # Same-cluster candidates

    same_cluster = candidates[
        candidates["Cluster"]
        == selected_cluster
    ].copy()


    same_cluster = (

        same_cluster
        .sort_values(
            "Similarity_Score",
            ascending=False
        )

    )

    # Global candidates

    all_candidates = (

        candidates
        .sort_values(
            "Similarity_Score",
            ascending=False
        )

    )

    selected_recommendations = []

    for _, row in same_cluster.iterrows():

        if len(
            selected_recommendations
        ) >= TOP_N:

            break


        selected_recommendations.append(
            row
        )


    if len(
        selected_recommendations
    ) < TOP_N:

        selected_indices = {

            int(row["_Index"])

            for row in selected_recommendations

        }


        for _, row in all_candidates.iterrows():

            candidate_index = int(
                row["_Index"]
            )


            if candidate_index in selected_indices:

                continue


            selected_recommendations.append(
                row
            )


            selected_indices.add(
                candidate_index
            )


            if len(
                selected_recommendations
            ) >= TOP_N:

                break


    # Create recommendation records

    for rank, recommendation in enumerate(

        selected_recommendations,

        start=1

    ):

        recommendation_rows.append({

            "Selected_Brand":
                selected_brand,

            "Selected_Model":
                selected_model,

            "Selected_Segment":
                selected_segment,

            "Selected_Price_USD":
                float(
                    selected_product["price_usd"]
                ),

            "Selected_Rating":
                float(
                    selected_product["rating"]
                ),


            "Recommended_Brand":
                recommendation["brand"],

            "Recommended_Model":
                recommendation["model"],

            "Recommended_Segment":
                recommendation["Segment"],

            "Recommended_Price_USD":
                float(
                    recommendation["price_usd"]
                ),

            "Recommended_Rating":
                float(
                    recommendation["rating"]
                ),

            "Recommended_Battery":
                float(
                    recommendation[
                        "battery_life_rating"
                    ]
                ),

            "Recommended_Camera":
                float(
                    recommendation[
                        "camera_rating"
                    ]
                ),

            "Recommended_Performance":
                float(
                    recommendation[
                        "performance_rating"
                    ]
                ),

            "Recommended_Design":
                float(
                    recommendation[
                        "design_rating"
                    ]
                ),

            "Recommended_Display":
                float(
                    recommendation[
                        "display_rating"
                    ]
                ),

            "Recommended_Engagement":
                float(
                    recommendation[
                        "engagement_score"
                    ]
                ),

            "Similarity_Score":
                float(
                    recommendation[
                        "Similarity_Score"
                    ]
                ),

            "Recommendation_Rank":
                rank

        })

# 18. CREATE RECOMMENDATION DATAFRAME

recommendations_df = pd.DataFrame(
    recommendation_rows
)


if recommendations_df.empty:

    raise ValueError(

        "\nERROR: No recommendations were generated."

    )

# 19. ROUND NUMERIC VALUES

numeric_columns = [

    "Selected_Price_USD",
    "Selected_Rating",

    "Recommended_Price_USD",
    "Recommended_Rating",

    "Recommended_Battery",
    "Recommended_Camera",
    "Recommended_Performance",
    "Recommended_Design",
    "Recommended_Display",
    "Recommended_Engagement",

    "Similarity_Score"

]

for column in numeric_columns:

    recommendations_df[column] = (

        recommendations_df[column]
        .round(4)

    )

# 20. VALIDATE RECOMMENDATION OUTPUT

expected_output_columns = [

    "Selected_Brand",
    "Selected_Model",
    "Selected_Segment",
    "Selected_Price_USD",
    "Selected_Rating",

    "Recommended_Brand",
    "Recommended_Model",
    "Recommended_Segment",
    "Recommended_Price_USD",
    "Recommended_Rating",

    "Recommended_Battery",
    "Recommended_Camera",
    "Recommended_Performance",
    "Recommended_Design",
    "Recommended_Display",
    "Recommended_Engagement",

    "Similarity_Score",
    "Recommendation_Rank"

]

missing_output_columns = [

    column

    for column in expected_output_columns

    if column not in recommendations_df.columns

]

if missing_output_columns:

    raise ValueError(

        "\nERROR: Recommendation output "
        "is missing columns:\n"

        + "\n".join(

            f"- {column}"

            for column in missing_output_columns

        )

    )

# 21. VALIDATE TOP-N RECOMMENDATIONS

recommendation_counts = (

    recommendations_df
    .groupby(
        [
            "Selected_Brand",
            "Selected_Model"
        ]
    )
    .size()

)

invalid_counts = (

    recommendation_counts
    < min(
        TOP_N,
        len(df) - 1
    )

)

if invalid_counts.any():

    print(
        "\nWARNING: Some products have fewer "
        "than the requested number of recommendations."
    )

# 22. VALIDATE SELF-RECOMMENDATIONS

self_recommendations = (

    recommendations_df[
        (
            recommendations_df[
                "Selected_Brand"
            ]

            ==

            recommendations_df[
                "Recommended_Brand"
            ]
        )

        &

        (
            recommendations_df[
                "Selected_Model"
            ]

            ==

            recommendations_df[
                "Recommended_Model"
            ]
        )
    ]

)

if len(self_recommendations) > 0:

    raise ValueError(

        "\nERROR: Self-recommendation detected.\n"

        "A product must not recommend itself."

    )

# 23. SORT FINAL OUTPUT

recommendations_df = (

    recommendations_df
    .sort_values(

        [

            "Selected_Brand",
            "Selected_Model",
            "Recommendation_Rank"

        ]

    )
    .reset_index(
        drop=True
    )

)

# 24. SAVE RECOMMENDATIONS

recommendations_df.to_csv(

    output_file,

    index=False

)


print(
    "\nRecommendation file saved as:"
)

print(
    output_file
)


# 25. CREATE RECOMMENDATION METADATA

metadata = {

    "method":
        "Cosine Similarity",

    "top_n":
        TOP_N,

    "features":
        PRODUCT_FEATURES,

    "number_of_products":
        int(len(df)),

    "number_of_recommendation_rows":
        int(len(recommendations_df)),

    "scaler_features":
        int(scaler.n_features_in_),

    "cluster_aware_recommendation":
        True

}

with open(

    recommendation_metadata_file,

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        metadata,

        file,

        indent=4

    )


print(
    "Recommendation metadata saved as:"
)

print(
    recommendation_metadata_file
)

# 26. DISPLAY SAMPLE RECOMMENDATIONS

print("\n" + "=" * 70)
print("SAMPLE RECOMMENDATIONS")
print("=" * 70)


sample_product = df.iloc[0]


sample_brand = sample_product[
    "brand"
]


sample_model = sample_product[
    "model"
]


sample_recommendations = (

    recommendations_df[
        (
            recommendations_df[
                "Selected_Brand"
            ]

            ==

            sample_brand

        )

        &

        (
            recommendations_df[
                "Selected_Model"
            ]

            ==

            sample_model

        )
    ]

)

print(

    f"\nSelected Product: "
    f"{sample_brand} {sample_model}"

)

print("\nRecommended Products:")

print(

    sample_recommendations[
        [
            "Recommended_Brand",
            "Recommended_Model",
            "Recommended_Segment",
            "Recommended_Price_USD",
            "Recommended_Rating",
            "Similarity_Score",
            "Recommendation_Rank"
        ]
    ]
    .to_string(
        index=False
    )

)

# 27. RECOMMENDATION VALIDATION SUMMARY

print("\n" + "=" * 70)
print("RECOMMENDATION VALIDATION")
print("=" * 70)

print(
    "\nTotal Products:",
    len(df)
)

print(
    "Total Recommendation Rows:",
    len(recommendations_df)
)

print(
    "Expected Maximum Rows:",
    len(df) * TOP_N
)

print(
    "\nSelf-Recommendations:",
    len(self_recommendations)
)

print(
    "Feature Count Used:",
    len(PRODUCT_FEATURES)
)

print(
    "Similarity Method:",
    "Cosine Similarity"
)

print(
    "Cluster-Aware Filtering:",
    "Enabled"
)

# 28. FINAL OUTPUT COLUMNS

print("\n" + "=" * 70)
print("FINAL OUTPUT COLUMNS")
print("=" * 70)

for number, column in enumerate(

    recommendations_df.columns,

    start=1

):

    print(
        f"{number}. {column}"
    )

# 29. FINAL COMPLETION MESSAGE

print("\n" + "=" * 70)
print("STEP 5 COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    "\nRecommendation system method:"
)

print(
    "Cosine Similarity"
)

print(
    "\nRecommendation features:"
)

for feature in PRODUCT_FEATURES:

    print(
        f"- {feature}"
    )

print(
    "\nTop recommendations per product:",
    TOP_N
)

print(
    "\nOutput file:"
)

print(
    output_file
)

print("\n" + "=" * 70)
