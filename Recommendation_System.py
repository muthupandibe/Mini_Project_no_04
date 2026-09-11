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
    product_df.columns.str.strip()
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

    )


# ============================================================
# 7. RECOMMENDATION FEATURES
# ============================================================

product_features = [

    "price_usd",

    "rating",

    "battery_life_rating",

    "camera_rating",

    "performance_rating",

    "design_rating",

    "display_rating"

]


# Add engagement score if available
if "engagement_score" in product_df.columns:

    product_features.append(
        "engagement_score"
    )


print("\n" + "=" * 70)
print("RECOMMENDATION FEATURES")
print("=" * 70)


for feature in product_features:

    print(
        "-",
        feature
    )


# ============================================================
# 8. CONVERT FEATURES TO NUMERIC
# ============================================================

for column in product_features:

    product_df[column] = pd.to_numeric(

        product_df[column],

        errors="coerce"

    )


# ============================================================
# 9. HANDLE MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE CHECK")
print("=" * 70)


missing_before = (
    product_df[product_features]
    .isnull()
    .sum()
)


print(
    "\nMissing values before handling:"
)

print(
    missing_before
)


for column in product_features:

    median_value = (
        product_df[column]
        .median()
    )


    if pd.isna(median_value):

        median_value = 0


    product_df[column] = (
        product_df[column]
        .fillna(median_value)
    )


print(
    "\nMissing values after handling:"
)


print(
    product_df[product_features]
    .isnull()
    .sum()
)


# ============================================================
# 10. CHECK UNIQUE PRODUCTS
# ============================================================

print("\n" + "=" * 70)
print("PRODUCT CHECK")
print("=" * 70)


duplicate_count = (
    product_df
    .duplicated(
        subset=["brand", "model"]
    )
    .sum()
)


print(
    "Duplicate product records:",
    duplicate_count
)


if duplicate_count > 0:

    product_df = (
        product_df
        .drop_duplicates(
            subset=["brand", "model"]
        )
        .reset_index(drop=True)
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
# 11. CREATE FEATURE MATRIX
# ============================================================

X = product_df[
    product_features
].copy()


# ============================================================
# 12. LOAD SCALER FROM STEP4
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
# 13. STANDARDIZE FEATURES
# ============================================================

X_scaled = scaler.transform(
    X
)


print(
    "Features standardized successfully."
)


# ============================================================
# 14. COSINE SIMILARITY
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
# 15. GENERATE RECOMMENDATIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING RECOMMENDATIONS")
print("=" * 70)


recommendation_list = []


TOP_N = 5


for product_index in range(
    len(product_df)
):


    selected_product = (
        product_df.iloc[
            product_index
        ]
    )


    selected_cluster = (
        selected_product["Cluster"]
    )


    # --------------------------------------------------------
    # Get similarity scores
    # --------------------------------------------------------

    similarity_scores = list(

        enumerate(

            similarity_matrix[
                product_index
            ]

        )

    )


    # --------------------------------------------------------
    # Remove selected product itself
    # --------------------------------------------------------

    similarity_scores = [

        item

        for item in similarity_scores

        if item[0] != product_index

    ]


    # --------------------------------------------------------
    # Prefer products from same cluster
    # --------------------------------------------------------

    same_cluster_scores = [

        item

        for item in similarity_scores

        if product_df.iloc[
            item[0]
        ]["Cluster"] == selected_cluster

    ]


    # If enough same-cluster products exist,
    # use same-cluster recommendations.

    if len(same_cluster_scores) >= TOP_N:

        similarity_scores = (
            same_cluster_scores
        )


    # --------------------------------------------------------
    # Sort by similarity
    # --------------------------------------------------------

    similarity_scores = sorted(

        similarity_scores,

        key=lambda x: x[1],

        reverse=True

    )


    # --------------------------------------------------------
    # Select Top N
    # --------------------------------------------------------

    top_recommendations = (
        similarity_scores[:TOP_N]
    )


    # --------------------------------------------------------
    # Store recommendations
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


            "Recommendation_Rank":
                rank,


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


            "Similarity_Score":
                round(
                    similarity_score,
                    4
                )

        })


# ============================================================
# 16. CREATE RECOMMENDATION DATAFRAME
# ============================================================

recommendations = pd.DataFrame(
    recommendation_list
)


print(
    "\nTotal Recommendation Records:",
    len(recommendations)
)


# ============================================================
# 17. RECOMMENDATION VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("RECOMMENDATION VALIDATION")
print("=" * 70)


if not recommendations.empty:


    average_similarity = (
        recommendations[
            "Similarity_Score"
        ].mean()
    )


    highest_similarity = (
        recommendations[
            "Similarity_Score"
        ].max()
    )


    lowest_similarity = (
        recommendations[
            "Similarity_Score"
        ].min()
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
# 18. SAVE RECOMMENDATIONS
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
# 19. DISPLAY SAMPLE RECOMMENDATIONS
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE RECOMMENDATIONS")
print("=" * 70)


if not recommendations.empty:

    print(

        recommendations.head(20)
        .to_string(index=False)

    )


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("RECOMMENDATION SYSTEM COMPLETED SUCCESSFULLY")
print("=" * 70)


print("\nMethod Used:")

print(
    "1. Product-level data"
)

print(
    "2. StandardScaler"
)

print(
    "3. Cosine Similarity"
)

print(
    "4. Same-segment preference"
)

print(
    "5. Top-5 recommendations"
)


print("\nOutput:")

print(
    "recommendations/"
    "all_product_recommendations.csv"
)


print("\n" + "=" * 70)
