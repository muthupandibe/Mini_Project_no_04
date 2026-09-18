# ============================================================
# STEP 7 – APPLICATION DEVELOPMENT & VISUALIZATION
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# 1. PAGE CONFIGURATION

st.set_page_config(
    page_title="Mobile Product Segmentation & Recommendation",
    page_icon="📱",
    layout="wide"
)

# 2. FILE PATHS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLUSTERED_FILE = os.path.join(
    BASE_DIR,
    "clustered_mobile_products.csv"
)

RECOMMENDATION_FILE = os.path.join(
    BASE_DIR,
    "recommendations",
    "all_product_recommendations.csv"
)

SCALER_FILE = os.path.join(
    BASE_DIR,
    "mobile_scaler.pkl"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "mobile_kmeans_model.pkl"
)

CLUSTER_PROFILE_FILE = os.path.join(
    BASE_DIR,
    "cluster_profile_summary.csv"
)

INSIGHTS_FILE = os.path.join(
    BASE_DIR,
    "insights",
    "mobile_product_insights_report.txt"
)

# 3. EXACT FEATURES USED 

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

# 4. REQUIRED COLUMNS

REQUIRED_CLUSTER_COLUMNS = [
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

# 5. LOAD DATA

@st.cache_data
def load_clustered_data():

    if not os.path.exists(CLUSTERED_FILE):
        return None

    df = pd.read_csv(CLUSTERED_FILE)

    return df


@st.cache_data
def load_precomputed_recommendations():

    if not os.path.exists(RECOMMENDATION_FILE):
        return None

    return pd.read_csv(RECOMMENDATION_FILE)


@st.cache_data
def load_cluster_profile():

    if not os.path.exists(CLUSTER_PROFILE_FILE):
        return None

    return pd.read_csv(CLUSTER_PROFILE_FILE)


@st.cache_resource
def load_scaler():

    if not os.path.exists(SCALER_FILE):
        return None

    return joblib.load(SCALER_FILE)

# 6. HELPER FUNCTIONS

def format_price(value):
    return f"${value:,.2f}"


def calculate_performance_score(df):

    performance_columns = [
        "rating",
        "battery_life_rating",
        "camera_rating",
        "performance_rating",
        "design_rating",
        "display_rating"
    ]

    return df[performance_columns].mean(axis=1)


def validate_features(df):

    missing = [
        col for col in PRODUCT_FEATURES
        if col not in df.columns
    ]

    if missing:
        st.error(
            "Required recommendation features are missing: "
            + ", ".join(missing)
        )
        st.stop()

# 7. USER PREFERENCE RECOMMENDATION FUNCTION

def recommend_from_user_preferences(
    df,
    scaler,
    user_features,
    top_n=5,
    selected_segment="All"
):

    # Validate scaler

    if not hasattr(scaler, "n_features_in_"):
        st.error(
            "Saved scaler does not contain feature information."
        )
        return pd.DataFrame()

    if scaler.n_features_in_ != len(PRODUCT_FEATURES):
        st.error(
            f"Scaler expects {scaler.n_features_in_} features, "
            f"but the recommendation system uses "
            f"{len(PRODUCT_FEATURES)} features."
        )
        return pd.DataFrame()

    # Create user preference DataFrame

    user_df = pd.DataFrame(
        [user_features],
        columns=PRODUCT_FEATURES
    )

    # Scale user input

    user_scaled = scaler.transform(user_df)

    # Scale product features

    product_matrix = df[PRODUCT_FEATURES].copy()

    product_scaled = scaler.transform(product_matrix)

    # Calculate cosine similarity

    similarity_scores = cosine_similarity(
        user_scaled,
        product_scaled
    )[0]

    result = df.copy()

    result["Similarity_Score"] = similarity_scores

    # Optional segment filtering

    if selected_segment != "All":

        segment_result = result[
            result["Segment"] == selected_segment
        ].copy()

        # If selected segment has products,
        # use only those products.
        if len(segment_result) > 0:
            result = segment_result

    # Sort by similarity

    result = result.sort_values(
        by="Similarity_Score",
        ascending=False
    )

    # Top N recommendations

    result = result.head(top_n).copy()

    result["Recommendation_Rank"] = range(
        1,
        len(result) + 1
    )

    return result

# 8. EXISTING PRODUCT RECOMMENDATION FUNCTION

def recommend_similar_product(
    df,
    selected_index,
    scaler,
    top_n=5
):

    product_matrix = df[PRODUCT_FEATURES]

    scaled_matrix = scaler.transform(
        product_matrix
    )

    similarity_matrix = cosine_similarity(
        scaled_matrix
    )

    selected_position = df.index.get_loc(
        selected_index
    )

    similarity_scores = similarity_matrix[
        selected_position
    ]

    result = df.copy()

    result["Similarity_Score"] = similarity_scores

    # Remove selected product itself
    result = result[
        result.index != selected_index
    ].copy()

    result = result.sort_values(
        by="Similarity_Score",
        ascending=False
    )

    result = result.head(top_n).copy()

    result["Recommendation_Rank"] = range(
        1,
        len(result) + 1
    )

    return result

# 9. LOAD CLUSTERED DATA

df = load_clustered_data()

if df is None:
    st.error(
        "clustered_mobile_products.csv was not found."
    )

    st.info(
        "Please run Step 4 before starting Step 7."
    )

    st.stop()

# 10. VALIDATE DATA

missing_columns = [
    col
    for col in REQUIRED_CLUSTER_COLUMNS
    if col not in df.columns
]

if missing_columns:

    st.error(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )

    st.stop()


validate_features(df)

# 11. DATA TYPE CONVERSION

for col in PRODUCT_FEATURES:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


if df[PRODUCT_FEATURES].isnull().any().any():

    st.error(
        "Missing or invalid values found in recommendation features."
    )

    st.stop()

# 12. PERFORMANCE SCORE

df["performance_score"] = calculate_performance_score(df)

# 13. LOAD SCALER

scaler = load_scaler()

if scaler is None:

    st.error(
        "mobile_scaler.pkl was not found."
    )

    st.info(
        "Please run Step 4 before starting Step 7."
    )

    st.stop()

# 14. SCALER FEATURE VALIDATION

if scaler.n_features_in_ != len(PRODUCT_FEATURES):

    st.error(
        f"Scaler expects {scaler.n_features_in_} features, "
        f"but Step 7 requires {len(PRODUCT_FEATURES)} features."
    )

    st.stop()

# 15. SIDEBAR

st.sidebar.title("📱 Mobile Recommendation System")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Dashboard",
        "📊 Product Segmentation",
        "⭐ Product Performance",
        "💰 Price vs Performance",
        "🎯 User Preferences",
        "📱 Similar Product",
        "📄 Insights Report"
    ]
)

# PAGE 1 – DASHBOARD

if page == "🏠 Dashboard":

    st.title(
        "📱 Mobile Product Segmentation & Recommendation System"
    )

    st.markdown(
        """
        ### Welcome!

        This application analyzes mobile products using:

        - K-Means Clustering
        - Product Segmentation
        - Cosine Similarity
        - User Preference Recommendation
        - Product Performance Analysis
        - Price vs Performance Analysis
        """
    )

    st.divider()

    # KPI CARDS

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "📱 Products",
        len(df)
    )

    col2.metric(
        "🏷️ Brands",
        df["brand"].nunique()
    )

    col3.metric(
        "🎯 Segments",
        df["Segment"].nunique()
    )

    col4.metric(
        "⭐ Average Rating",
        f"{df['rating'].mean():.2f}"
    )

    st.divider()

    # PRODUCT TABLE

    st.subheader("📋 Product Overview")

    display_columns = [
        "brand",
        "model",
        "price_usd",
        "rating",
        "performance_score",
        "Cluster",
        "Segment"
    ]

    display_df = df[display_columns].copy()

    display_df["price_usd"] = display_df[
        "price_usd"
    ].round(2)

    display_df["performance_score"] = display_df[
        "performance_score"
    ].round(2)

    st.dataframe(
        display_df,
        use_container_width=True
    )

    # BRAND DISTRIBUTION

    st.subheader("🏷️ Product Distribution by Brand")

    brand_counts = df["brand"].value_counts()

    fig, ax = plt.subplots()

    brand_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Brand")
    ax.set_ylabel("Number of Products")
    ax.set_title("Products by Brand")

    plt.xticks(rotation=45)

    st.pyplot(fig)

    plt.close(fig)

# PAGE 2 – PRODUCT SEGMENTATION

elif page == "📊 Product Segmentation":

    st.title("📊 Product Segmentation")

    st.write(
        "Products are grouped using K-Means clustering "
        "based on price, rating, specifications, and engagement."
    )

    # SEGMENT FILTER

    segments = ["All"] + sorted(
        df["Segment"].dropna().unique().tolist()
    )

    selected_segment = st.selectbox(
        "Select Segment",
        segments
    )

    if selected_segment == "All":

        segment_df = df.copy()

    else:

        segment_df = df[
            df["Segment"] == selected_segment
        ].copy()

    # KPI

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Products",
        len(segment_df)
    )

    col2.metric(
        "Average Price",
        format_price(segment_df["price_usd"].mean())
    )

    col3.metric(
        "Average Rating",
        f"{segment_df['rating'].mean():.2f}"
    )

    st.divider()

    # CLUSTER DISTRIBUTION

    st.subheader("Cluster / Segment Distribution")

    cluster_counts = df["Segment"].value_counts()

    fig, ax = plt.subplots()

    cluster_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Segment")
    ax.set_ylabel("Number of Products")
    ax.set_title("Product Segment Distribution")

    plt.xticks(rotation=0)

    st.pyplot(fig)

    plt.close(fig)

    # CLUSTER PROFILE

    st.subheader("📌 Segment Profile")

    profile = df.groupby(
        "Segment"
    )[PRODUCT_FEATURES].mean().round(2)

    st.dataframe(
        profile,
        use_container_width=True
    )

    # PRODUCTS

    st.subheader(
        f"📱 Products in {selected_segment} Segment"
    )

    table_columns = [
        "brand",
        "model",
        "price_usd",
        "rating",
        "performance_score",
        "Cluster",
        "Segment"
    ]

    st.dataframe(
        segment_df[table_columns],
        use_container_width=True
    )

# PAGE 3 – PRODUCT PERFORMANCE

elif page == "⭐ Product Performance":

    st.title("⭐ Product Performance Analysis")

    # HIGH PERFORMING

    st.subheader("🏆 High-Performing Products")

    high_performing = df.sort_values(
        "performance_score",
        ascending=False
    ).head(10)

    high_columns = [
        "brand",
        "model",
        "price_usd",
        "rating",
        "performance_score",
        "Segment"
    ]

    st.dataframe(
        high_performing[high_columns],
        use_container_width=True
    )

    # LOW PERFORMING

    st.subheader("📉 Lower-Performing Products")

    low_performing = df.sort_values(
        "performance_score",
        ascending=True
    ).head(10)

    st.dataframe(
        low_performing[high_columns],
        use_container_width=True
    )

    # BRAND PERFORMANCE

    st.subheader("🏷️ Brand Performance")

    brand_performance = (
        df.groupby("brand")
        .agg(
            Average_Rating=("rating", "mean"),
            Average_Performance=("performance_score", "mean"),
            Average_Price=("price_usd", "mean")
        )
        .round(2)
        .sort_values(
            "Average_Performance",
            ascending=False
        )
    )

    st.dataframe(
        brand_performance,
        use_container_width=True
    )

    fig, ax = plt.subplots()

    brand_performance[
        "Average_Performance"
    ].plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Brand")
    ax.set_ylabel("Performance Score")
    ax.set_title("Average Performance by Brand")

    plt.xticks(rotation=45)

    st.pyplot(fig)

    plt.close(fig)

# PAGE 4 – PRICE VS PERFORMANCE

elif page == "💰 Price vs Performance":

    st.title("💰 Price vs Performance Analysis")

    correlation = df[
        ["price_usd", "performance_score"]
    ].corr().iloc[0, 1]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Price",
        format_price(df["price_usd"].mean())
    )

    col2.metric(
        "Average Performance",
        f"{df['performance_score'].mean():.2f}"
    )

    col3.metric(
        "Price-Performance Correlation",
        f"{correlation:.3f}"
    )

    st.divider()

    # SCATTER PLOT

    fig, ax = plt.subplots()

    for segment in sorted(
        df["Segment"].unique()
    ):

        segment_data = df[
            df["Segment"] == segment
        ]

        ax.scatter(
            segment_data["price_usd"],
            segment_data["performance_score"],
            label=segment
        )

    ax.set_xlabel("Price (USD)")
    ax.set_ylabel("Performance Score")
    ax.set_title(
        "Price vs Performance by Segment"
    )

    ax.legend()

    st.pyplot(fig)

    plt.close(fig)

    st.info(
        "Correlation describes the observed linear relationship "
        "between price and performance. It does not establish causation."
    )

# PAGE 5 – USER PREFERENCE RECOMMENDATION

elif page == "🎯 User Preferences":

    st.title(
        "🎯 Personalized Mobile Recommendation"
    )

    st.markdown(
        """
        ### Enter your mobile preferences

        The system compares your preferences with existing
        products using **Cosine Similarity**.

        The recommendation engine uses the same 8 features
        used during clustering and recommendation model creation.
        """
    )

    st.divider()

    # USER INPUTS

    st.subheader("💰 Price & Rating Preferences")

    col1, col2 = st.columns(2)

    with col1:

        user_price = st.slider(
            "Maximum / Preferred Price (USD)",
            min_value=100.0,
            max_value=2000.0,
            value=700.0,
            step=10.0
        )

    with col2:

        user_rating = st.slider(
            "Preferred Rating",
            min_value=1.0,
            max_value=5.0,
            value=4.0,
            step=0.1
        )

    st.subheader("📱 Specification Preferences")

    col1, col2 = st.columns(2)

    with col1:

        user_battery = st.slider(
            "Battery Life Rating",
            min_value=1,
            max_value=5,
            value=4
        )

        user_camera = st.slider(
            "Camera Rating",
            min_value=1,
            max_value=5,
            value=4
        )

        user_performance = st.slider(
            "Performance Rating",
            min_value=1,
            max_value=5,
            value=4
        )

    with col2:

        user_design = st.slider(
            "Design Rating",
            min_value=1,
            max_value=5,
            value=4
        )

        user_display = st.slider(
            "Display Rating",
            min_value=1,
            max_value=5,
            value=4
        )

        user_engagement = st.slider(
            "Engagement Score Preference",
            min_value=0.0,
            max_value=float(
                max(1.0, df["engagement_score"].max())
            ),
            value=float(
                min(
                    2.0,
                    max(1.0, df["engagement_score"].max())
                )
            ),
            step=0.1
        )

    # SEGMENT PREFERENCE

    st.subheader("🎯 Segment Preference")

    segment_options = [
        "All"
    ] + sorted(
        df["Segment"].unique().tolist()
    )

    preferred_segment = st.selectbox(
        "Preferred Product Segment",
        segment_options
    )

    # TOP N

    top_n = st.slider(
        "Number of Recommendations",
        min_value=1,
        max_value=10,
        value=5
    )

    st.divider()

    # USER FEATURE VECTOR

    user_features = {
        "price_usd": user_price,
        "rating": user_rating,
        "battery_life_rating": user_battery,
        "camera_rating": user_camera,
        "performance_rating": user_performance,
        "design_rating": user_design,
        "display_rating": user_display,
        "engagement_score": user_engagement
    }

    st.subheader("🧾 Your Selected Preferences")

    preference_df = pd.DataFrame(
        {
            "Feature": list(user_features.keys()),
            "Preference": list(user_features.values())
        }
    )

    st.dataframe(
        preference_df,
        use_container_width=True,
        hide_index=True
    )

    # RECOMMEND BUTTON

    if st.button(
        "🔍 Find Recommended Mobiles",
        type="primary"
    ):

        recommendations = recommend_from_user_preferences(
            df=df,
            scaler=scaler,
            user_features=user_features,
            top_n=top_n,
            selected_segment=preferred_segment
        )

        if recommendations.empty:

            st.warning(
                "No products were available for the selected criteria."
            )

        else:

            st.success(
                f"Found {len(recommendations)} recommended products."
            )

            st.subheader(
                "📱 Recommended Mobile Products"
            )

            recommendation_display = recommendations[
                [
                    "brand",
                    "model",
                    "price_usd",
                    "rating",
                    "performance_score",
                    "battery_life_rating",
                    "camera_rating",
                    "performance_rating",
                    "design_rating",
                    "display_rating",
                    "Segment",
                    "Similarity_Score",
                    "Recommendation_Rank"
                ]
            ].copy()

            recommendation_display[
                "Similarity_Score"
            ] = recommendation_display[
                "Similarity_Score"
            ].round(4)

            recommendation_display[
                "performance_score"
            ] = recommendation_display[
                "performance_score"
            ].round(2)

            st.dataframe(
                recommendation_display,
                use_container_width=True,
                hide_index=True
            )

            # RECOMMENDATION CHART

            st.subheader(
                "📊 Recommendation Similarity"
            )

            chart_df = recommendations.copy()

            chart_df["Product"] = (
                chart_df["brand"]
                + " "
                + chart_df["model"]
            )

            chart_df = chart_df.sort_values(
                "Similarity_Score",
                ascending=True
            )

            fig, ax = plt.subplots()

            ax.barh(
                chart_df["Product"],
                chart_df["Similarity_Score"]
            )

            ax.set_xlabel(
                "Cosine Similarity"
            )

            ax.set_ylabel(
                "Product"
            )

            ax.set_title(
                "Recommended Products Based on User Preferences"
            )

            st.pyplot(fig)

            plt.close(fig)

            st.info(
                "A higher cosine similarity means the product's "
                "feature vector is more similar to your entered preferences."
            )


# PAGE 6 – SIMILAR PRODUCT RECOMMENDATION

elif page == "📱 Similar Product":

    st.title(
        "📱 Find Similar Products"
    )

    st.write(
        "Select an existing mobile product to find "
        "other products with similar characteristics."
    )

    # PRODUCT SELECTION

    df["Product"] = (
        df["brand"]
        + " | "
        + df["model"]
    )

    selected_product = st.selectbox(
        "Select a Product",
        sorted(df["Product"].unique())
    )

    selected_index = df[
        df["Product"] == selected_product
    ].index[0]

    selected_row = df.loc[
        selected_index
    ]

    # SELECTED PRODUCT DETAILS

    st.subheader("📌 Selected Product")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Brand",
        selected_row["brand"]
    )

    col2.metric(
        "Model",
        selected_row["model"]
    )

    col3.metric(
        "Price",
        format_price(
            selected_row["price_usd"]
        )
    )

    col4.metric(
        "Rating",
        f"{selected_row['rating']:.2f}"
    )

    st.write(
        f"**Segment:** {selected_row['Segment']}"
    )

    st.divider()

    # TOP N

    top_n = st.slider(
        "Number of Similar Products",
        min_value=1,
        max_value=10,
        value=5,
        key="similar_product_top_n"
    )

    if st.button(
        "🔍 Find Similar Products",
        type="primary"
    ):

        recommendations = recommend_similar_product(
            df=df,
            selected_index=selected_index,
            scaler=scaler,
            top_n=top_n
        )

        st.subheader(
            "🤝 Similar Product Recommendations"
        )

        recommendation_display = recommendations[
            [
                "brand",
                "model",
                "price_usd",
                "rating",
                "performance_score",
                "Segment",
                "Similarity_Score",
                "Recommendation_Rank"
            ]
        ].copy()

        recommendation_display[
            "Similarity_Score"
        ] = recommendation_display[
            "Similarity_Score"
        ].round(4)

        recommendation_display[
            "performance_score"
        ] = recommendation_display[
            "performance_score"
        ].round(2)

        st.dataframe(
            recommendation_display,
            use_container_width=True,
            hide_index=True
        )

        # SIMILARITY CHART

        chart_df = recommendations.copy()

        chart_df["Product"] = (
            chart_df["brand"]
            + " "
            + chart_df["model"]
        )

        chart_df = chart_df.sort_values(
            "Similarity_Score",
            ascending=True
        )

        fig, ax = plt.subplots()

        ax.barh(
            chart_df["Product"],
            chart_df["Similarity_Score"]
        )

        ax.set_xlabel(
            "Cosine Similarity"
        )

        ax.set_ylabel(
            "Product"
        )

        ax.set_title(
            f"Products Similar to {selected_product}"
        )

        st.pyplot(fig)

        plt.close(fig)


# PAGE 7 – INSIGHTS REPORT

elif page == "📄 Insights Report":

    st.title(
        "📄 Insights & Reporting"
    )

    if os.path.exists(INSIGHTS_FILE):

        with open(
            INSIGHTS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            report = file.read()

        st.text_area(
            "Project Insights",
            report,
            height=600
        )

        st.download_button(
            label="⬇️ Download Insights Report",
            data=report,
            file_name="mobile_product_insights_report.txt",
            mime="text/plain"
        )

    else:

        st.warning(
            "Insights report was not found."
        )

        st.info(
            "Please run Step 6 before using this page."
        )


# FOOTER

st.divider()

st.caption(
    "Mobile Product Segmentation & Recommendation System | "
    "K-Means Clustering + Cosine Similarity + Streamlit"
)
