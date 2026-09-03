# ============================================================
# FILE: Step5_Recommendation.py
# PROJECT: Mobile Product Segmentation and Recommendation System
# ============================================================

import os
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. FILE PATHS
# ============================================================

input_file = "cleaned_mobile_reviews.csv"

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
        "Please run Step2_Data_Preprocessing.py first."
    )


# ============================================================
# 4. LOAD CLEANED DATASET
# ============================================================

df = pd.read_csv(input_file)

df.columns = df.columns.str.strip()


print("\n" + "=" * 70)
print("MOBILE PRODUCT RECOMMENDATION SYSTEM")
print("=" * 70)

print("\nDataset Shape:", df.shape)


# ============================================================
# 5. REQUIRED COLUMNS
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
# 6. RECOMMENDATION FEATURES
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


print("\n" + "=" * 70)
print("RECOMMENDATION FEATURES")
print("=" * 70)


for feature in product_features:

    print("-", feature)


# ============================================================
# 7. CONVERT FEATURES TO NUMERIC
# ============================================================

for column in product_features:

    df[column] = pd.to_numeric(

        df[column],

        errors="coerce"

    )


# ============================================================
# 8. HANDLE MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE HANDLING")
print("=" * 70)


print("\nMissing values before cleaning:")

print(

    df[product_features]

    .isnull()

    .sum()

)


for column in product_features:

    median_value = df[column].median()


    if pd.isna(median_value):

        median_value = 0


    df[column] = df[column].fillna(

        median_value

    )


print("\nMissing values after cleaning:")

print(

    df[product_features]

    .isnull()

    .sum()

)


# ============================================================
# 9. CREATE UNIQUE PRODUCT PROFILES
# ============================================================

print("\n" + "=" * 70)
print("CREATING PRODUCT PROFILES")
print("=" * 70)


product_df = (

    df.groupby(

        ["brand", "model"],

        as_index=False

    )[product_features]

    .mean()

)


print(

    "\nNumber of Unique Products:",

    len(product_df)

)


if len(product_df) < 2:

    raise ValueError(

        "\nERROR: At least 2 unique products "

        "are required for recommendation."

    )


# ============================================================
# 10. FEATURE STANDARDIZATION
# ============================================================

print("\n" + "=" * 70)
print("FEATURE STANDARDIZATION")
print("=" * 70)


X = product_df[

    product_features

].copy()


scaler = StandardScaler()


X_scaled = scaler.fit_transform(X)


print(

    "\nProduct features standardized successfully."

)


# ============================================================
# 11. COSINE SIMILARITY
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING COSINE SIMILARITY")
print("=" * 70)


similarity_matrix = cosine_similarity(

    X_scaled

)


print(

    "\nCosine similarity matrix created."

)


print(

    "Matrix Shape:",

    similarity_matrix.shape

)


# ============================================================
# 12. GENERATE RECOMMENDATIONS FOR ALL PRODUCTS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING RECOMMENDATIONS FOR ALL PRODUCTS")
print("=" * 70)


recommendation_list = []


TOP_N = 5


for product_index in range(

    len(product_df)

):

    selected_product = product_df.iloc[

        product_index

    ]


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
    # Sort similarity scores
    # --------------------------------------------------------

    similarity_scores = sorted(

        similarity_scores,

        key=lambda x: x[1],

        reverse=True

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
    # Select Top 5
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

        recommended_product = product_df.iloc[

            recommended_index

        ]


        recommendation_list.append({

            "Selected_Brand":

                selected_product["brand"],


            "Selected_Model":

                selected_product["model"],


            "Selected_Price_USD":

                round(

                    selected_product["price_usd"],

                    2

                ),


            "Selected_Rating":

                round(

                    selected_product["rating"],

                    2

                ),


            "Recommendation_Rank":

                rank,


            "Recommended_Brand":

                recommended_product["brand"],


            "Recommended_Model":

                recommended_product["model"],


            "Recommended_Price_USD":

                round(

                    recommended_product["price_usd"],

                    2

                ),


            "Recommended_Rating":

                round(

                    recommended_product["rating"],

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
# 13. CREATE RECOMMENDATION DATAFRAME
# ============================================================

recommendations = pd.DataFrame(

    recommendation_list

)


print(

    "\nTotal Recommendation Records:",

    len(recommendations)

)


# ============================================================
# 14. VALIDATE RECOMMENDATION RELEVANCE
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


    if average_similarity >= 0.80:

        relevance = "HIGH"


    elif average_similarity >= 0.60:

        relevance = "MODERATE"


    else:

        relevance = "LOW"


    print(

        f"Overall Recommendation Relevance : "

        f"{relevance}"

    )


# ============================================================
# 15. SAVE ALL RECOMMENDATIONS
# ============================================================

recommendations.to_csv(

    output_file,

    index=False

)


print("\n" + "=" * 70)
print("RECOMMENDATION FILE SAVED")
print("=" * 70)


print(

    f"\nOutput Folder:\n"

    f"{output_folder}"

)


print(

    f"\nOutput File:\n"

    f"{output_file}"

)


# ============================================================
# 16. DISPLAY SAMPLE RECOMMENDATIONS
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
# 17. COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("RECOMMENDATION SYSTEM COMPLETED SUCCESSFULLY")
print("=" * 70)


print("\nMethod Used:")

print("1. Product-level aggregation")

print("2. Feature standardization")

print("3. Cosine Similarity")

print("4. Top-5 recommendations for every product")

print("5. Similarity-based validation")


print("\nOutput:")

print(

    "recommendations/all_product_recommendations.csv"

)


print("\n" + "=" * 70)