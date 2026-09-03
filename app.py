# ============================================================
# FILE: Step6_Streamlit.py
# PROJECT: Mobile Product Segmentation and Recommendation System
# ============================================================

import os
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

import plotly.express as px


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Mobile Product Segmentation & Recommendation",
    page_icon="📱",
    layout="wide"
)


# ============================================================
# 2. TITLE
# ============================================================

st.title(
    "📱 Mobile Product Segmentation & Recommendation System"
)

st.markdown(
    """
    This application analyzes mobile products using
    **K-Means clustering** and provides
    **similar-product recommendations using Cosine Similarity**.
    """
)


# ============================================================
# 3. FILE PATHS
# ============================================================

cleaned_file = "cleaned_mobile_reviews.csv"

clustered_file = "clustered_mobile_reviews.csv"

recommendation_file = os.path.join(
    "recommendations",
    "all_product_recommendations.csv"
)


# ============================================================
# 4. CHECK FILES
# ============================================================

missing_files = []

if not os.path.exists(cleaned_file):
    missing_files.append(cleaned_file)

if not os.path.exists(clustered_file):
    missing_files.append(clustered_file)

if not os.path.exists(recommendation_file):
    missing_files.append(recommendation_file)


if missing_files:

    st.error(
        "Required files are missing."
    )

    st.write(
        "Please run Step 2, Step 4 and Step 5 first."
    )

    for file in missing_files:
        st.write(f"- `{file}`")

    st.stop()


# ============================================================
# 5. LOAD DATA
# ============================================================

df = pd.read_csv(cleaned_file)

clustered_df = pd.read_csv(clustered_file)

recommendations = pd.read_csv(
    recommendation_file
)


# ============================================================
# 6. CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()

clustered_df.columns = (
    clustered_df.columns.str.strip()
)

recommendations.columns = (
    recommendations.columns.str.strip()
)


# ============================================================
# 7. REQUIRED COLUMNS
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

    st.error(
        "Required columns are missing from "
        "clustered_mobile_reviews.csv:"
    )

    for column in missing_columns:
        st.write(f"- {column}")

    st.stop()


# ============================================================
# 8. SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "Select Section",

    [
        "Dashboard",
        "Product Segmentation",
        "Recommendations"
    ]

)


# ============================================================
# 9. COMMON DATA PREPARATION
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
# PAGE 1 - DASHBOARD
# ============================================================

if page == "Dashboard":

    st.header("📊 Project Dashboard")


    # --------------------------------------------------------
    # KPI VALUES
    # --------------------------------------------------------

    total_products = len(clustered_df)

    total_brands = (
        clustered_df["brand"]
        .nunique()
    )

    total_segments = (
        clustered_df["Segment"]
        .nunique()
        if "Segment" in clustered_df.columns
        else clustered_df["Cluster"].nunique()
    )

    average_rating = (
        clustered_df["rating"]
        .mean()
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Total Products",
            f"{total_products:,}"
        )


    with col2:

        st.metric(
            "Total Brands",
            f"{total_brands:,}"
        )


    with col3:

        st.metric(
            "Product Segments",
            f"{total_segments:,}"
        )


    with col4:

        st.metric(
            "Average Rating",
            f"{average_rating:.2f}"
        )


    st.divider()


    # --------------------------------------------------------
    # PRICE DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "Mobile Product Price Distribution"
    )


    fig_price = px.histogram(

        clustered_df,

        x="price_usd",

        nbins=30,

        title="Distribution of Mobile Product Prices",

        labels={
            "price_usd": "Price (USD)"
        }

    )


    st.plotly_chart(
        fig_price,
        use_container_width=True
    )


    # --------------------------------------------------------
    # RATING DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "Mobile Product Rating Distribution"
    )


    fig_rating = px.histogram(

        clustered_df,

        x="rating",

        nbins=20,

        title="Distribution of Product Ratings",

        labels={
            "rating": "Rating"
        }

    )


    st.plotly_chart(

        fig_rating,

        use_container_width=True

    )


# ============================================================
# PAGE 2 - PRODUCT SEGMENTATION
# ============================================================

elif page == "Product Segmentation":

    st.header(
        "🎯 Product Segmentation & Cluster Analysis"
    )


    # --------------------------------------------------------
    # SEGMENT DISTRIBUTION
    # --------------------------------------------------------

    if "Segment" in clustered_df.columns:

        segment_counts = (

            clustered_df["Segment"]

            .value_counts()

            .reset_index()

        )

        segment_counts.columns = [

            "Segment",

            "Product_Count"

        ]


        st.subheader(
            "Product Segment Distribution"
        )


        fig_segment = px.bar(

            segment_counts,

            x="Segment",

            y="Product_Count",

            title="Number of Products by Segment",

            text="Product_Count",

            labels={
                "Product_Count":
                "Number of Products"
            }

        )


        st.plotly_chart(

            fig_segment,

            use_container_width=True

        )


    # --------------------------------------------------------
    # PRICE VS RATING
    # --------------------------------------------------------

    st.subheader(
        "Price vs Rating by Cluster"
    )


    fig_cluster = px.scatter(

        clustered_df,

        x="price_usd",

        y="rating",

        color="Segment"
        if "Segment" in clustered_df.columns
        else "Cluster",

        hover_data=[

            "brand",

            "model",

            "price_usd",

            "rating",

            "camera_rating",

            "performance_rating"

        ],

        title="Mobile Product Clusters",

        labels={

            "price_usd": "Price (USD)",

            "rating": "Rating"

        }

    )


    st.plotly_chart(

        fig_cluster,

        use_container_width=True

    )


    # --------------------------------------------------------
    # CLUSTER PROFILE
    # --------------------------------------------------------

    st.subheader(
        "Cluster Profile"
    )


    profile_columns = [

        "Cluster",

        "Product_Count",

        "Percentage",

        "price_usd",

        "rating",

        "battery_life_rating",

        "camera_rating",

        "performance_rating",

        "design_rating",

        "display_rating"

    ]


    available_profile_columns = [

        column

        for column in profile_columns

        if column in clustered_df.columns

    ]


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


    if "Segment" in clustered_df.columns:

        segment_mapping = (

            clustered_df[

                ["Cluster", "Segment"]

            ]

            .drop_duplicates()

        )


        cluster_profile = cluster_profile.merge(

            segment_mapping,

            on="Cluster",

            how="left"

        )


        display_columns = [

            "Cluster",

            "Segment",

            "Product_Count",

            "Percentage"

        ] + numeric_columns


    else:

        display_columns = [

            "Cluster",

            "Product_Count",

            "Percentage"

        ] + numeric_columns


    st.dataframe(

        cluster_profile[display_columns],

        use_container_width=True

    )


    # --------------------------------------------------------
    # BUSINESS INSIGHTS
    # --------------------------------------------------------

    st.subheader(
        "💡 Cluster Business Insights"
    )


    for _, row in cluster_profile.iterrows():

        if "Segment" in row:

            segment_name = row["Segment"]

        else:

            segment_name = (
                f"Cluster {int(row['Cluster'])}"
            )


        st.markdown(

            f"""
            **{segment_name}**

            - Average Price: ${row['price_usd']:.2f}
            - Average Rating: {row['rating']:.2f}
            - Average Camera Rating: {row['camera_rating']:.2f}
            - Average Performance Rating: {row['performance_rating']:.2f}
            - Number of Products: {int(row['Product_Count'])}
            """

        )


# ============================================================
# PAGE 3 - RECOMMENDATION SYSTEM
# ============================================================

elif page == "Recommendations":

    st.header(
        "🤖 Mobile Product Recommendation System"
    )


    st.markdown(

        """
        Select a mobile product below to find the
        **Top 5 similar products** based on price,
        ratings and product specifications.
        """

    )


    # --------------------------------------------------------
    # CREATE PRODUCT LIST
    # --------------------------------------------------------

    clustered_df["Product_Name"] = (

        clustered_df["brand"].astype(str)

        + " "

        + clustered_df["model"].astype(str)

    )


    product_names = sorted(

        clustered_df["Product_Name"]

        .unique()

        .tolist()

    )


    # --------------------------------------------------------
    # PRODUCT SELECTION
    # --------------------------------------------------------

    selected_product_name = st.selectbox(

        "Select a Mobile Product",

        product_names

    )


    selected_row = clustered_df[

        clustered_df["Product_Name"]

        == selected_product_name

    ].iloc[0]


    selected_brand = selected_row["brand"]

    selected_model = selected_row["model"]


    # --------------------------------------------------------
    # SELECTED PRODUCT DETAILS
    # --------------------------------------------------------

    st.subheader(
        "Selected Product"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(

            "Brand",

            selected_brand

        )


    with col2:

        st.metric(

            "Model",

            selected_model

        )


    with col3:

        st.metric(

            "Price",

            f"${selected_row['price_usd']:.2f}"

        )


    with col4:

        st.metric(

            "Rating",

            f"{selected_row['rating']:.2f}"

        )


    if "Segment" in selected_row.index:

        st.info(

            f"Product Segment: "
            f"**{selected_row['Segment']}**"

        )


    # --------------------------------------------------------
    # PRODUCT SPECIFICATIONS
    # --------------------------------------------------------

    st.subheader(
        "Product Specifications"
    )


    specification_data = pd.DataFrame({

        "Feature": [

            "Battery",

            "Camera",

            "Performance",

            "Design",

            "Display"

        ],

        "Rating": [

            selected_row[
                "battery_life_rating"
            ],

            selected_row[
                "camera_rating"
            ],

            selected_row[
                "performance_rating"
            ],

            selected_row[
                "design_rating"
            ],

            selected_row[
                "display_rating"
            ]

        ]

    })


    st.dataframe(

        specification_data,

        use_container_width=True,

        hide_index=True

    )


    st.divider()


    # --------------------------------------------------------
    # FIND RECOMMENDATIONS
    # --------------------------------------------------------

    selected_recommendations = recommendations[

        (

            recommendations[
                "Selected_Brand"
            ].astype(str)

            == str(selected_brand)

        )

        &

        (

            recommendations[
                "Selected_Model"
            ].astype(str)

            == str(selected_model)

        )

    ].copy()


    # --------------------------------------------------------
    # DISPLAY TOP 5
    # --------------------------------------------------------

    st.subheader(
        "⭐ Top 5 Similar Products"
    )


    if selected_recommendations.empty:

        st.warning(

            "No recommendations found for "
            "the selected product."

        )

    else:

        selected_recommendations = (

            selected_recommendations

            .sort_values(
                "Recommendation_Rank"
            )

        )


        display_columns = [

            "Recommendation_Rank",

            "Recommended_Brand",

            "Recommended_Model",

            "Recommended_Price_USD",

            "Recommended_Rating",

            "Recommended_Battery",

            "Recommended_Camera",

            "Recommended_Performance",

            "Similarity_Score"

        ]


        st.dataframe(

            selected_recommendations[
                display_columns
            ],

            use_container_width=True,

            hide_index=True

        )


        # ----------------------------------------------------
        # SIMILARITY CHART
        # ----------------------------------------------------

        fig_similarity = px.bar(

            selected_recommendations,

            x="Recommended_Model",

            y="Similarity_Score",

            title="Recommendation Similarity Scores",

            text="Similarity_Score",

            labels={

                "Recommended_Model":
                "Recommended Product",

                "Similarity_Score":
                "Similarity Score"

            }

        )


        fig_similarity.update_traces(

            texttemplate="%{text:.3f}",

            textposition="outside"

        )


        st.plotly_chart(

            fig_similarity,

            use_container_width=True

        )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        average_similarity = (

            selected_recommendations[

                "Similarity_Score"

            ].mean()

        )


        highest_similarity = (

            selected_recommendations[

                "Similarity_Score"

            ].max()

        )


        st.subheader(
            "Recommendation Relevance"
        )


        validation_col1, validation_col2 = (

            st.columns(2)

        )


        with validation_col1:

            st.metric(

                "Average Similarity",

                f"{average_similarity:.4f}"

            )


        with validation_col2:

            st.metric(

                "Highest Similarity",

                f"{highest_similarity:.4f}"

            )


        if average_similarity >= 0.80:

            st.success(

                "Recommendation Relevance: HIGH"

            )

        elif average_similarity >= 0.60:

            st.info(

                "Recommendation Relevance: MODERATE"

            )

        else:

            st.warning(

                "Recommendation Relevance: LOW"

            )


        st.caption(

            "Similarity score is a feature-based "
            "relevance indicator, not recommendation "
            "accuracy. No ground-truth user preference "
            "data is available in this project."

        )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.info(

    """
    **Project Methodology**

    • Data Cleaning  
    • EDA  
    • K-Means Clustering  
    • Product Segmentation  
    • Cosine Similarity  
    • Product Recommendation  
    • Streamlit Visualization
    """

)