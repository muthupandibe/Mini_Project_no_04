# ============================================================
# Step5_Recommendation.py
# ============================================================

import os
import joblib
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. FILE PATHS
# ============================================================

input_file = "clustered_mobile_products.csv"

scaler_file = "mobile_scaler.pkl"

output_folder = "recommendations"

output_file = os.path.join(
    output_folder,
    "all_product_recommendations.csv"
)


# ============================================================
# 2. CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(
    output_folder,
    exist_ok=True
)


# ============================================================
# 3. CHECK INPUT FILE
# ============================================================

if not os.path.exists(input_file):

    raise FileNotFoundError(
        f"\nERROR: {input_file} not found.\n"
        "Please run Step4_Clustering.py first."
    )


# ============================================================
# 4. CHECK SCALER FILE
# ============================================================

if not os.path.exists(scaler_file):

    raise FileNotFoundError(
        f"\nERROR: {scaler_file} not found.\n"
        "Please run Step4_Clustering.py first."
    )


# ============================================================
# 5. LOAD CLUSTERED PRODUCT DATA
# ============================================================

product_df = pd.read_csv(
    input_file
)

product_df.columns = (
    product_df.columns
    .str.strip()
)


print("\n" + "=" * 70)
print("MOBILE PRODUCT RECOMMENDATION SYSTEM")
print("=" * 70)

print(
    "\nProduct Dataset Shape:",
    product_df.shape
)


# ============================================================
# 6. REQUIRED COLUMNS
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

    "engagement_score",

    "Cluster",
    "Segment"

]


missing_columns = [

    column

    for column in required_columns

    if column not in product_df.columns

]


if missing_columns:

    raise ValueError(

        "\nERROR: Required columns are missing:\n"

        + "\n".join(

            f"- {column}"

            for column in missing_columns

        )

        + "\n\nPlease run Step2_Data_Preprocessing.py "
          "and Step4_Clustering.py again."

    )


# ============================================================
# 7. EXACT RECOMMENDATION FEATURES
# ============================================================

# IMPORTANT:
# These MUST be exactly the same features and
# same order used by Step4_Clustering.py.

product_features = [

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
    product_features,
    start=1
):

    print(
        f"{number}. {feature}"
    )


print(
    "\nTotal Recommendation Features:",
    len(product_features)
)


# ============================================================
# 8. LOAD SCALER
# ============================================================

print("\n" + "=" * 70)
print("LOADING STANDARD SCALER")
print("=" * 70)


scaler = joblib.load(
    scaler_file
)


print(
    "\nScaler loaded successfully."
)


# ============================================================
# 9. VALIDATE SCALER FEATURE COUNT
# ============================================================

if not hasattr(
    scaler,
    "n_features_in_"
):

    raise ValueError(
        "\nERROR: Loaded scaler does not contain "
        "n_features_in_.\n"
        "Please run Step4_Clustering.py again."
    )


scaler_feature_count = (
    scaler.n_features_in_
)


recommendation_feature_count = (
    len(product_features)
)


print(
    "\nScaler expected features:",
    scaler_feature_count
)


print(
    "Recommendation features:",
    recommendation_feature_count
)


if (
    scaler_feature_count
    != recommendation_feature_count
):

    raise ValueError(

        "\nERROR: Feature count mismatch!\n\n"

        f"Scaler expects "
        f"{scaler_feature_count} features.\n"

        f"Recommendation system provides "
        f"{recommendation_feature_count} features.\n\n"

        "IMPORTANT:\n"
        "Run Step2_Data_Preprocessing.py first,\n"
        "then run Step4_Clustering.py again,\n"
        "then run Step5_Recommendation.py."

    )


print(
    "\nScaler feature count validation passed."
)


# ============================================================
# 10. CHECK SCALER FEATURE ORDER
# ============================================================

if hasattr(
    scaler,
    "feature_names_in_"
):

    scaler_features = list(
        scaler.feature_names_in_
    )

    print("\n" + "=" * 70)
    print("SCALER FEATURE ORDER")
    print("=" * 70)

    for number, feature in enumerate(
        scaler_features,
        start=1
    ):

        print(
            f"{number}. {feature}"
        )


    if scaler_features != product_features:

        raise ValueError(

            "\nERROR: Feature order mismatch!\n\n"

            "Scaler features:\n"

            + "\n".join(

                f"- {feature}"

                for feature in scaler_features

            )

            + "\n\nRecommendation features:\n"

            + "\n".join(

                f"- {feature}"

                for feature in product_features

            )

            + "\n\n"
              "Please rerun Step4_Clustering.py "
              "and then Step5_Recommendation.py."

        )


    print(
        "\nScaler feature order validation passed."
    )


# ============================================================
# 11. CONVERT FEATURES TO NUMERIC
# ============================================================

print("\n" + "=" * 70)
print("NUMERIC FEATURE CONVERSION")
print("=" * 70)


for column in product_features:

    product_df[column] = pd.to_numeric(

        product_df[column],

        errors="coerce"

    )


print(
    "\nAll recommendation features converted "
    "to numeric format."
)


# ============================================================
# 12. HANDLE INFINITE VALUES
# ============================================================

product_df[
    product_features
] = (

    product_df[
        product_features
    ]

    .replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

)


# ============================================================
# 13. MISSING VALUE CHECK
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE CHECK")
print("=" * 70)


missing_before = (

    product_df[
        product_features
    ]

    .isnull()

    .sum()

)


print(
    "\nMissing values before handling:"
)

print(
    missing_before
)


# ============================================================
# 14. FILL MISSING VALUES
# ============================================================

for column in product_features:

    median_value = (

        product_df[column]

        .median()
    )


    if pd.isna(
        median_value
    ):

        median_value = 0


    product_df[column] = (

        product_df[column]

        .fillna(
            median_value
        )

    )


print(
    "\nMissing values after handling:"
)


print(

    product_df[
        product_features
    ]

    .isnull()

    .sum()

)


# ============================================================
# 15. CHECK UNIQUE PRODUCTS
# ============================================================

print("\n" + "=" * 70)
print("PRODUCT CHECK")
print("=" * 70)


duplicate_count = (

    product_df

    .duplicated(
        subset=[
            "brand",
            "model"
        ]
    )

    .sum()

)


print(
    "Duplicate product records:",
    duplicate_count
)


# ============================================================
# 16. REMOVE DUPLICATE PRODUCTS
# ============================================================

if duplicate_count > 0:

    product_df = (

        product_df

        .drop_duplicates(

            subset=[
                "brand",
                "model"
            ]

        )

        .reset_index(
            drop=True
        )

    )


print(
    "Unique Products:",
    len(product_df)
)


if len(product_df) < 2:

    raise ValueError(

        "\nERROR: At least 2 unique products "
        "are required for recommendation."

    )


# ============================================================
# 17. CREATE FEATURE MATRIX
# ============================================================

X = (

    product_df[
        product_features
    ]

    .copy()

)


print("\n" + "=" * 70)
print("FEATURE MATRIX")
print("=" * 70)


print(
    "\nFeature Matrix Shape:",
    X.shape
)


# ============================================================
# 18. STANDARDIZE FEATURES
# ============================================================

print("\n" + "=" * 70)
print("STANDARDIZATION")
print("=" * 70)


X_scaled = scaler.transform(
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
# 19. COSINE SIMILARITY
# ============================================================

print("\n" + "=" * 70)
print("COSINE SIMILARITY")
print("=" * 70)


similarity_matrix = (

    cosine_similarity(
        X_scaled
    )

)


print(
    "\nCosine similarity matrix created."
)


print(
    "Matrix Shape:",
    similarity_matrix.shape
)


# ============================================================
# 20. GENERATE RECOMMENDATIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING RECOMMENDATIONS")
print("=" * 70)


recommendation_list = []


TOP_N = 5


for product_index in range(
    len(product_df)
):


    # --------------------------------------------------------
    # SELECT PRODUCT
    # --------------------------------------------------------

    selected_product = (

        product_df.iloc[
            product_index
        ]

    )


    selected_cluster = (

        selected_product[
            "Cluster"
        ]

    )


    # --------------------------------------------------------
    # GET SIMILARITY SCORES
    # --------------------------------------------------------

    similarity_scores = list(

        enumerate(

            similarity_matrix[
                product_index
            ]

        )

    )


    # --------------------------------------------------------
    # REMOVE SELECTED PRODUCT
    # --------------------------------------------------------

    similarity_scores = [

        item

        for item in similarity_scores

        if item[0] != product_index

    ]


    # --------------------------------------------------------
    # PREFER SAME CLUSTER
    # --------------------------------------------------------

    same_cluster_scores = [

        item

        for item in similarity_scores

        if (

            product_df.iloc[
                item[0]
            ]["Cluster"]

            == selected_cluster

        )

    ]


    # --------------------------------------------------------
    # USE SAME CLUSTER IF ENOUGH PRODUCTS EXIST
    # --------------------------------------------------------

    if len(
        same_cluster_scores
    ) >= TOP_N:

        similarity_scores = (
            same_cluster_scores
        )


    # --------------------------------------------------------
    # SORT BY SIMILARITY
    # --------------------------------------------------------

    similarity_scores = sorted(

        similarity_scores,

        key=lambda x: x[1],

        reverse=True

    )


    # --------------------------------------------------------
    # TOP 5
    # --------------------------------------------------------

    top_recommendations = (

        similarity_scores[
            :TOP_N
        ]

    )


    # --------------------------------------------------------
    # STORE RECOMMENDATIONS
    # --------------------------------------------------------

    for rank, (

        recommended_index,

        similarity_score

    ) in enumerate(

        top_recommendations,

        start=1

    ):


        recommended_product = (

            product_df.iloc[
                recommended_index
            ]

        )


        recommendation_list.append({

            # ------------------------------------------------
            # SELECTED PRODUCT
            # ------------------------------------------------

            "Selected_Brand":

                selected_product[
                    "brand"
                ],


            "Selected_Model":

                selected_product[
                    "model"
                ],


            "Selected_Segment":

                selected_product[
                    "Segment"
                ],


            "Selected_Price_USD":

                round(

                    selected_product[
                        "price_usd"
                    ],

                    2

                ),


            "Selected_Rating":

                round(

                    selected_product[
                        "rating"
                    ],

                    2

                ),


            # ------------------------------------------------
            # RECOMMENDATION RANK
            # ------------------------------------------------

            "Recommendation_Rank":

                rank,


            # ------------------------------------------------
            # RECOMMENDED PRODUCT
            # ------------------------------------------------

            "Recommended_Brand":

                recommended_product[
                    "brand"
                ],


            "Recommended_Model":

                recommended_product[
                    "model"
                ],


            "Recommended_Segment":

                recommended_product[
                    "Segment"
                ],


            "Recommended_Price_USD":

                round(

                    recommended_product[
                        "price_usd"
                    ],

                    2

                ),


            "Recommended_Rating":

                round(

                    recommended_product[
                        "rating"
                    ],

                    2

                ),


            "Recommended_Battery":

                round(

                    recommended_product[
                        "battery_life_rating"
                    ],

                    2

                ),


            "Recommended_Camera":

                round(

                    recommended_product[
                        "camera_rating"
                    ],

                    2

                ),


            "Recommended_Performance":

                round(

                    recommended_product[
                        "performance_rating"
                    ],

                    2

                ),


            "Recommended_Design":

                round(

                    recommended_product[
                        "design_rating"
                    ],

                    2

                ),


            "Recommended_Display":

                round(

                    recommended_product[
                        "display_rating"
                    ],

                    2

                ),


            "Recommended_Engagement":

                round(

                    recommended_product[
                        "engagement_score"
                    ],

                    2

                ),


            # ------------------------------------------------
            # SIMILARITY
            # ------------------------------------------------

            "Similarity_Score":

                round(

                    similarity_score,

                    4

                )

        })


# ============================================================
# 21. CREATE RECOMMENDATION DATAFRAME
# ============================================================

recommendations = pd.DataFrame(
    recommendation_list
)


print(
    "\nTotal Recommendation Records:",
    len(recommendations)
)


# ============================================================
# 22. VALIDATE RECOMMENDATIONS
# ============================================================

print("\n" + "=" * 70)
print("RECOMMENDATION VALIDATION")
print("=" * 70)


if recommendations.empty:

    raise ValueError(

        "\nERROR: No recommendations were generated."

    )


# ============================================================
# 23. SIMILARITY STATISTICS
# ============================================================

average_similarity = (

    recommendations[
        "Similarity_Score"
    ]

    .mean()

)


highest_similarity = (

    recommendations[
        "Similarity_Score"
    ]

    .max()

)


lowest_similarity = (

    recommendations[
        "Similarity_Score"
    ]

    .min()

)


print(

    f"\nAverage Similarity Score : "
    f"{average_similarity:.4f}"

)


print(

    f"Highest Similarity Score : "
    f"{highest_similarity:.4f}"

)


print(

    f"Lowest Similarity Score  : "
    f"{lowest_similarity:.4f}"

)


print(
    "\nRecommendation quality is "
    "reported using cosine similarity."
)


# ============================================================
# 24. CHECK TOP-5 RECOMMENDATION COUNT
# ============================================================

recommendation_counts = (

    recommendations

    .groupby(
        [
            "Selected_Brand",
            "Selected_Model"
        ]
    )

    .size()

)


print(
    "\nRecommendation count statistics:"
)


print(
    recommendation_counts.describe()
)


# ============================================================
# 25. SAVE RECOMMENDATIONS
# ============================================================

recommendations.to_csv(

    output_file,

    index=False

)


print("\n" + "=" * 70)
print("RECOMMENDATION FILE SAVED")
print("=" * 70)


print(
    "\nOutput Folder:",
    output_folder
)


print(
    "\nOutput File:",
    output_file
)


# ============================================================
# 26. DISPLAY SAMPLE RECOMMENDATIONS
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE RECOMMENDATIONS")
print("=" * 70)


print(

    recommendations

    .head(20)

    .to_string(
        index=False
    )

)


# ============================================================
# 27. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("RECOMMENDATION SYSTEM COMPLETED SUCCESSFULLY")
print("=" * 70)


print("\nMethod Used:")

print(
    "1. Product-level data"
)

print(
    "2. 8 standardized product features"
)

print(
    "3. StandardScaler from Step4"
)

print(
    "4. Cosine Similarity"
)

print(
    "5. Same-cluster preference"
)

print(
    "6. Top-5 recommendations"
)


print("\nRecommendation Features:")

for feature in product_features:

    print(
        " -",
        feature
    )


print("\nOutput:")

print(
    "recommendations/"
    "all_product_recommendations.csv"
)


print("\n" + "=" * 70)
print("STEP 5 COMPLETED SUCCESSFULLY")
print("=" * 70)
