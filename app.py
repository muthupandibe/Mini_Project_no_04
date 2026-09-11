# ============================================================
# Step6_Streamlit.py
# ============================================================

import os
import pandas as pd
import streamlit as st
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

st.title("📱 Mobile Product Segmentation & Recommendation System")

st.markdown(
    """
    This application analyzes mobile products using
    **K-Means Clustering** and provides
    **similar-product recommendations using Cosine Similarity**.
    """
)


# ============================================================
# 3. FILE PATHS
# ============================================================

cleaned_file = "cleaned_mobile_reviews.csv"

# IMPORTANT:
# Step4 corrected output
clustered_file = "clustered_mobile_products.csv"

# Step5 output
recommendation_file = os.path.join(
    "recommendations",
    "all_product_recommendations.csv"
)


# ============================================================
# 4. CHECK REQUIRED FILES
# ============================================================

missing_files = []

if not os.path.exists(cleaned_file):
    missing_files.append(cleaned_file)

if not os.path.exists(clustered_file):
    missing_files.append(clustered_file)

if not os.path.exists(recommendation_file):
    missing_files.append(recommendation_file)


if missing_files:

    st.error("❌ Required project files are missing.")

    st.markdown(
        """
        Please run the following project steps before
        starting the Streamlit application:

        **Step 2 → Data Preprocessing**

        **Step 4 → Clustering**

        **Step 5 → Recommendation System**
        """
    )

    st.write("Missing files:")

    for file in missing_files:
        st.write(f"- `{file}`")

    st.stop()


# ============================================================
# 5. LOAD DATA
# ============================================================

try:

    df = pd.read_csv(cleaned_file)

    clustered_df = pd.read_csv(clustered_file)

    recommendations = pd.read_csv(
        recommendation_file
    )

except Exception as e:

    st.error(
        f"❌ Error while loading project files: {e}"
    )

    st.stop()


# ============================================================
# 6. CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
)

clustered_df.columns = (
    clustered_df.columns
    .str.strip()
)

recommendations.columns = (
    recommendations.columns
    .str.strip()
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
    "display_rating",
    "Cluster"

]


missing_columns = [

    column

    for column in required_columns

    if column not in clustered_df.columns

]


if missing_columns:

    st.error(
        "❌ Required columns are missing from "
        "clustered_mobile_products.csv"
    )

    st.write("Missing columns:")

    for column in missing_columns:

        st.write(f"- `{column}`")

    st.stop()


# ============================================================
# 8. REQUIRED RECOMMENDATION COLUMNS
# ============================================================

recommendation_columns = [

    "Selected_Brand",
    "Selected_Model",
    "Recommended_Brand",
    "Recommended_Model",
    "Recommended_Price_USD",
    "Recommended_Rating",
    "Recommended_Battery",
    "Recommended_Camera",
    "Recommended_Performance",
    "Similarity_Score",
    "Recommendation_Rank"

]


missing_recommendation_columns = [

    column

    for column in recommendation_columns

    if column not in recommendations.columns

]


if missing_recommendation_columns:

    st.error(
        "❌ Required recommendation columns are missing."
    )

    st.write("Missing columns:")

    for column in missing_recommendation_columns:

        st.write(f"- `{column}`")

    st.stop()


# ============================================================
# 9. NUMERIC DATA PREPARATION
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


# Replace missing numeric values with median

for column in numeric_columns:

    if clustered_df[column].isna().any():

        median_value = clustered_df[column].median()

        clustered_df[column] = (
            clustered_df[column]
            .fillna(median_value)
        )


# ============================================================
# 10. CREATE PRODUCT NAME
# ============================================================

clustered_df["Product_Name"] = (

    clustered_df["brand"]
    .astype(str)
    .str.strip()

    + " "

    +

    clustered_df["model"]
    .astype(str)
    .str.strip()

)


# ============================================================
# 11. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(

    "Select Section",

    [
        "Dashboard",
        "Product Segmentation",
        "Recommendations"
    ]

)


# ============================================================
# PAGE 1 - DASHBOARD
# ============================================================

if page == "Dashboard":

    st.header("📊 Project Dashboard")

    st.markdown(
        """
        ### Project Overview

        This dashboard provides an overview of the mobile
        product dataset, product segments and recommendation
        system.
        """
    )

    # --------------------------------------------------------
    # KPI VALUES
    # --------------------------------------------------------

    total_products = (
        clustered_df["Product_Name"]
        .nunique()
    )

    total_brands = (
        clustered_df["brand"]
        .nunique()
    )

    total_clusters = (
        clustered_df["Cluster"]
        .nunique()
    )

    average_rating = (
        clustered_df["rating"]
        .mean()
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "📱 Total Products",
            f"{total_products:,}"
        )


    with col2:

        st.metric(
            "🏷️ Total Brands",
            f"{total_brands:,}"
        )


    with col3:

        st.metric(
            "🎯 Product Segments",
            f"{total_clusters:,}"
        )


    with col4:

        st.metric(
            "⭐ Average Rating",
            f"{average_rating:.2f}"
        )


    st.divider()


    # --------------------------------------------------------
    # PRICE DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "💰 Mobile Product Price Distribution"
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
        "⭐ Mobile Product Rating Distribution"
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


    # --------------------------------------------------------
    # BRAND DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "🏷️ Products by Brand"
    )


    brand_counts = (

        clustered_df["brand"]

        .value_counts()

        .reset_index()

    )


    brand_counts.columns = [

        "Brand",
        "Product_Count"

    ]


    fig_brand = px.bar(

        brand_counts,

        x="Brand",

        y="Product_Count",

        text="Product_Count",

        title="Number of Products by Brand"

    )


    st.plotly_chart(

        fig_brand,

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
            "📊 Product Segment Distribution"
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


    else:

        st.warning(
            "Segment column is not available."
        )


    # --------------------------------------------------------
    # PRICE VS RATING
    # --------------------------------------------------------

    st.subheader(
        "💰 Price vs Rating by Cluster"
    )


    color_column = (

        "Segment"

        if "Segment" in clustered_df.columns

        else "Cluster"

    )


    fig_cluster = px.scatter(

        clustered_df,

        x="price_usd",

        y="rating",

        color=color_column,

        hover_data=[

            "brand",

            "model",

            "price_usd",

            "rating",

            "camera_rating",

            "performance_rating",

            "battery_life_rating"

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
        "📋 Cluster Profile"
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


    # Add Segment name

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

        use_container_width=True,

        hide_index=True

    )


    # --------------------------------------------------------
    # BUSINESS INSIGHTS
    # --------------------------------------------------------

    st.subheader(
        "💡 Cluster Business Insights"
    )


    for _, row in cluster_profile.iterrows():

        if "Segment" in cluster_profile.columns:

            segment_name = row["Segment"]

        else:

            segment_name = (

                f"Cluster {int(row['Cluster'])}"

            )


        st.markdown(

            f"""
            ### {segment_name}

            - **Average Price:** ${row['price_usd']:.2f}
            - **Average Rating:** {row['rating']:.2f}
            - **Average Camera Rating:** {row['camera_rating']:.2f}
            - **Average Performance Rating:** {row['performance_rating']:.2f}
            - **Average Battery Rating:** {row['battery_life_rating']:.2f}
            - **Number of Products:** {int(row['Product_Count'])}
            """

        )


# ============================================================
# PAGE 3 - RECOMMENDATIONS
# ============================================================

elif page == "Recommendations":

    st.header(
        "🤖 Mobile Product Recommendation System"
    )


    st.markdown(
        """
        Select a mobile product to find the
        **Top 5 similar products** based on product
        characteristics using **Cosine Similarity**.
        """
    )


    # --------------------------------------------------------
    # PRODUCT LIST
    # --------------------------------------------------------

    product_names = sorted(

        clustered_df["Product_Name"]

        .dropna()

        .unique()

        .tolist()

    )


    if not product_names:

        st.warning(
            "No products are available."
        )

        st.stop()


    # --------------------------------------------------------
    # PRODUCT SELECTION
    # --------------------------------------------------------

    selected_product_name = st.selectbox(

        "📱 Select a Mobile Product",

        product_names

    )


    selected_row = (

        clustered_df[

            clustered_df["Product_Name"]

            == selected_product_name

        ]

        .iloc[0]

    )


    selected_brand = selected_row["brand"]

    selected_model = selected_row["model"]


    # --------------------------------------------------------
    # SELECTED PRODUCT DETAILS
    # --------------------------------------------------------

    st.subheader(
        "📱 Selected Product"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Brand",
            str(selected_brand)
        )


    with col2:

        st.metric(
            "Model",
            str(selected_model)
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

            f"🎯 Product Segment: "
            f"**{selected_row['Segment']}**"

        )


    # --------------------------------------------------------
    # PRODUCT SPECIFICATIONS
    # --------------------------------------------------------

    st.subheader(
        "⚙️ Product Specifications"
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

    selected_recommendations = (

        recommendations[

            (

                recommendations[
                    "Selected_Brand"
                ]

                .astype(str)

                == str(selected_brand)

            )

            &

            (

                recommendations[
                    "Selected_Model"
                ]

                .astype(str)

                == str(selected_model)

            )

        ]

        .copy()

    )


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
                "Cosine Similarity"

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
        # SIMILARITY VALIDATION
        # ----------------------------------------------------

        average_similarity = (

            selected_recommendations[
                "Similarity_Score"
            ]

            .mean()

        )


        highest_similarity = (

            selected_recommendations[
                "Similarity_Score"
            ]

            .max()

        )


        st.subheader(
            "📈 Recommendation Similarity"
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


        # ----------------------------------------------------
        # RELEVANCE MESSAGE
        # ----------------------------------------------------

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

            """
            Similarity score is a feature-based relevance
            indicator, not recommendation accuracy.
            No ground-truth user preference data is available
            in this project.
            """

        )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.info(

    """
    ### 📌 Project Methodology

    • Data Collection  
    • Data Preprocessing  
    • Exploratory Data Analysis  
    • K-Means Clustering  
    • Product Segmentation  
    • Cosine Similarity  
    • Product Recommendation  
    • Model Evaluation  
    • Insights & Reporting  
    • Streamlit Dashboard
    """

)
