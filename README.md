# 📱 Mobile Product Segmentation & Recommendation System

## 📌 Project Overview

The **Mobile Product Segmentation & Recommendation System** is an end-to-end Machine Learning project developed using **Python, Pandas, NumPy, Scikit-learn, Matplotlib, Joblib, and Streamlit**.

The project analyzes mobile product and customer review data, performs data preprocessing and exploratory data analysis, engineers product and engagement features, segments mobile products using **K-Means clustering**, and generates product recommendations using **Cosine Similarity**.

The final system provides an interactive **Streamlit dashboard** where users can:

* Explore mobile product segments
* Analyze product performance
* Compare price and performance
* Enter their preferred product characteristics
* Receive personalized recommendations based on their input
* Find products similar to an existing mobile
* View generated analytical insights

---

# 🎯 Project Objectives

The main objectives of this project are:

* Clean and preprocess mobile review and product data.
* Handle missing values and duplicate records.
* Perform exploratory data analysis (EDA).
* Engineer useful product and engagement features.
* Analyze relationships between price, ratings, and specifications.
* Standardize numerical features for Machine Learning.
* Segment mobile products using K-Means clustering.
* Determine a suitable number of clusters using the Elbow Method.
* Evaluate clustering quality using Silhouette Score.
* Build a similarity-based product recommendation system.
* Generate recommendations from user preferences.
* Generate business and analytical insights.
* Develop an interactive Streamlit application.

---

# 🗂️ Project Workflow

```text
Raw Mobile Review Dataset
          ↓
1. Data Collection
          ↓
2. Data Preprocessing
          ↓
3. Exploratory Data Analysis
          ↓
4. Feature Engineering & Scaling
          ↓
5. K-Means Clustering
          ↓
6. Product Segmentation
          ↓
7. Recommendation System
          ↓
8. Insights & Reporting
          ↓
9. Streamlit Application
          ↓
Business & Product Insights
```

---

# 📊 Dataset

## Dataset Name

```text
Mobile Reviews Sentiment null.csv
```

## Dataset Size

* **Rows:** 50,000
* **Columns:** 22
* **Brands:** 7
* **Countries:** 8
* **Models:** 22 unique mobile products
* **Approximate Price Range:** $180 – $1,500
* **Rating Range:** 1 – 5

## Original Dataset Columns

| Column                 | Description                     |
| ---------------------- | ------------------------------- |
| `review_id`            | Unique review identifier        |
| `customer_name`        | Customer name                   |
| `age`                  | Customer age                    |
| `brand`                | Mobile phone brand              |
| `model`                | Mobile model                    |
| `price_usd`            | Product price in USD            |
| `price_local`          | Product price in local currency |
| `currency`             | Currency                        |
| `exchange_rate_to_usd` | Exchange rate to USD            |
| `rating`               | Customer rating                 |
| `sentiment`            | Review sentiment                |
| `country`              | Customer country                |
| `language`             | Review language                 |
| `review_date`          | Review date                     |
| `verified_purchase`    | Purchase verification status    |
| `battery_life_rating`  | Battery life rating             |
| `camera_rating`        | Camera rating                   |
| `performance_rating`   | Performance rating              |
| `design_rating`        | Design rating                   |
| `display_rating`       | Display rating                  |
| `helpful_votes`        | Number of helpful votes         |
| `source`               | Review source                   |

---

# 🛠️ Technologies Used

## Programming Language

* Python 3.x

## Libraries

* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Joblib
* Streamlit

## Machine Learning Techniques

* Data Preprocessing
* Feature Engineering
* StandardScaler
* K-Means Clustering
* Elbow Method
* Silhouette Score
* Cosine Similarity

> The current Streamlit application uses Matplotlib and does not require Plotly or Seaborn.

---

# 📁 Final Project Structure

The latest implementation uses the following project structure:

```text
Project_no_04/
│
├── Mobile Reviews Sentiment null.csv
│
├── Step1_Data_Collection.py
├── Step2_Data_Preprocessing.py
├── Step3_EDA.py
├── Step4_Clustering.py
├── Step5_Recommendation.py
├── Step6_Insights.py
├── Step7_Streamlit.py
│
├── cleaned_mobile_reviews.csv
├── clustered_mobile_products.csv
├── cluster_profile_summary.csv
├── elbow_inertia.csv
│
├── mobile_scaler.pkl
├── mobile_kmeans_model.pkl
├── clustering_metadata.pkl
│
├── elbow_curve.png
├── cluster_segmentation.png
├── segment_distribution.png
│
├── eda_outputs/
│
├── recommendations/
│   ├── all_product_recommendations.csv
│   └── recommendation_metadata.json
│
└── insights/
    ├── cluster_wise_analysis.csv
    ├── high_performing_products.csv
    ├── low_performing_products.csv
    ├── price_range_performance.csv
    ├── product_attribute_patterns.csv
    ├── brand_performance_analysis.csv
    ├── segment_rating_analysis.csv
    ├── cluster_distribution.png
    ├── price_vs_performance.png
    └── mobile_product_insights_report.txt
```

---

# 🔄 Project Modules

# 1️⃣ Data Collection

## File

```text
Step1_Data_Collection.py
```

The Data Collection module loads the original mobile review dataset and performs an initial inspection.

### Operations

* Load the CSV dataset.
* Display dataset shape.
* Display column names.
* Inspect data types.
* Display sample records.
* Check basic dataset information.

### Input

```text
Mobile Reviews Sentiment null.csv
```

### Output

Initial dataset information used for subsequent preprocessing.

---

# 2️⃣ Data Preprocessing

## File

```text
Step2_Data_Preprocessing.py
```

The preprocessing stage prepares the raw dataset for Machine Learning and analysis.

### Operations

* Load raw dataset.
* Check missing values.
* Handle missing numerical values.
* Check duplicate records.
* Convert required columns to appropriate numerical types.
* Retain relevant product and analytical fields.
* Handle missing `helpful_votes`.
* Prevent negative engagement values.
* Create `engagement_score`.
* Save the cleaned dataset.

### Missing Value Handling

For the latest dataset:

* `price_usd` missing values are filled using the median.
* `rating` missing values are filled using the median.
* `helpful_votes` missing values are handled before calculating engagement.

### Engagement Feature

The project creates:

```text
engagement_score = log1p(helpful_votes)
```

This transforms helpful-vote counts into a more manageable numerical feature.

Negative helpful-vote values are clipped to zero before calculating the engagement score.

### Output

```text
cleaned_mobile_reviews.csv
```

### Final Preprocessed Columns

The latest preprocessing output contains:

```text
brand
model
country
price_usd
rating
battery_life_rating
camera_rating
performance_rating
design_rating
display_rating
helpful_votes
engagement_score
```

---

# 3️⃣ Exploratory Data Analysis

## File

```text
Step3_EDA.py
```

EDA is performed to understand the structure and relationships within the cleaned dataset.

### Analysis Performed

* Descriptive statistics
* Brand distribution
* Country distribution
* Product distribution
* Rating distribution
* Price distribution
* Brand-wise average rating
* Brand-wise average price
* Specification analysis
* Price vs rating
* Price vs performance
* Rating vs specifications
* Correlation analysis
* Product-level performance analysis

### Important EDA Findings

The cleaned dataset contains approximately:

```text
Mean Price       ≈ $687.33
Median Price     ≈ $637.59
Mean Rating      ≈ 3.12
```

The analysis also examines relationships between:

```text
Price
Rating
Battery
Camera
Performance
Design
Display
Engagement
```

### Output Files

```text
eda_outputs/
```

Important analytical outputs include:

```text
eda_specification_summary.csv
eda_brand_summary.csv
```

EDA provides the foundation for selecting and understanding the numerical features used in clustering and recommendation.

---

# 4️⃣ K-Means Clustering

## File

```text
Step4_Clustering.py
```

K-Means clustering is used to group similar mobile products based on their numerical characteristics.

---

## 🔢 Product-Level Clustering

The clustering process aggregates review-level data at the:

```text
brand + model
```

level.

This converts the 50,000 review records into product-level observations representing the available mobile products.

The project contains approximately:

```text
22 unique products
```

for product-level clustering.

---

# 📐 Clustering Features

The project uses the following **exact eight features** for clustering:

```text
1. price_usd
2. rating
3. battery_life_rating
4. camera_rating
5. performance_rating
6. design_rating
7. display_rating
8. engagement_score
```

These features must remain consistent across:

* Step 4 — Clustering
* Step 5 — Recommendation
* Step 7 — Streamlit

### Feature List

```python
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
```

---

# 📏 Feature Scaling

Before K-Means clustering, the numerical features are standardized using:

```text
StandardScaler
```

The fitted scaler is saved as:

```text
mobile_scaler.pkl
```

This same scaler is reused by the recommendation system and Streamlit application.

This prevents feature-scale differences from disproportionately affecting similarity and clustering calculations.

For example:

```text
price_usd → hundreds or thousands
rating → 1–5
specification ratings → 1–5
engagement_score → variable range
```

Standardization converts these variables to comparable scales.

---

# 📉 Elbow Method

The Elbow Method is used to examine different values of K.

The project evaluates:

```text
K = 2
K = 3
K = 4
K = 5
K = 6
K = 7
K = 8
```

The observed inertia values are approximately:

|  K | Inertia |
| -: | ------: |
|  2 |   91.29 |
|  3 |   64.21 |
|  4 |   53.76 |
|  5 |   44.66 |
|  6 |   37.31 |
|  7 |   32.04 |
|  8 |   27.48 |

The project selects:

```text
K = 3
```

based on the Elbow Method analysis.

### Output

```text
elbow_inertia.csv
elbow_curve.png
```

---

# 🔵 K-Means Segmentation

The final K-Means model uses:

```text
NUMBER_OF_CLUSTERS = 3
```

The model is saved as:

```text
mobile_kmeans_model.pkl
```

The resulting cluster labels are assigned descriptive segment names based on the characteristics of each cluster.

The labels are descriptive business interpretations and are **not inherent meanings of the numerical cluster IDs**.

---

# 🏷️ Product Segments

The final three segments are interpreted based on product-level characteristics:

### Budget

Products with relatively lower price levels and corresponding product characteristics.

### Mid-Range

Products occupying an intermediate position in terms of price and product characteristics.

### Premium

Products with relatively higher price levels and stronger overall product characteristics.

> The segment names are assigned after analyzing the cluster profiles. K-Means itself only produces numerical cluster labels.

---

# 📊 Cluster Evaluation

The clustering model is evaluated using:

## Inertia

Inertia measures the within-cluster sum of squared distances.

Lower inertia generally indicates more compact clusters, although inertia naturally decreases as K increases.

Therefore, the Elbow Method is used to help select an appropriate K.

---

## Silhouette Score

Silhouette Score evaluates cluster cohesion and separation.

### Simple Definition

> Silhouette Score indicates how well an observation fits within its assigned cluster compared with other clusters.

A higher value generally indicates better-defined separation and cohesion.

The final clustering script calculates the Silhouette Score for the selected clustering configuration.

---

# 📦 Clustering Outputs

The clustering module generates:

```text
clustered_mobile_products.csv
cluster_profile_summary.csv
elbow_inertia.csv
mobile_scaler.pkl
mobile_kmeans_model.pkl
clustering_metadata.pkl
elbow_curve.png
cluster_segmentation.png
segment_distribution.png
```

---

# 📄 Clustered Product Dataset

The final clustered dataset contains fields such as:

```text
brand
model
price_usd
rating
battery_life_rating
camera_rating
performance_rating
design_rating
display_rating
engagement_score
review_count
Cluster
Segment
```

This product-level dataset is used by the recommendation system and Streamlit application.

---

# 5️⃣ Recommendation System

## File

```text
Step5_Recommendation.py
```

The recommendation system uses **Cosine Similarity** to identify products with similar feature profiles.

---

# 📐 Recommendation Features

The recommendation system uses the **same eight features** as the clustering model:

```text
price_usd
rating
battery_life_rating
camera_rating
performance_rating
design_rating
display_rating
engagement_score
```

This feature consistency is important because the saved StandardScaler expects exactly eight features.

---

# 📏 Cosine Similarity

Cosine Similarity measures similarity between two feature vectors based on the angle between them.

### Simple Explanation

> Products with similar feature patterns receive higher cosine similarity.

Typical interpretation:

```text
1 → highly similar direction
0 → little directional similarity
```

The recommendation system ranks products according to their similarity score.

---

# 🎯 Recommendation Modes

The final system supports two recommendation modes.

## Mode 1 — User Preference Recommendation

Users can enter their preferred mobile-product characteristics.

The Streamlit application accepts:

* Preferred price
* Preferred rating
* Preferred battery rating
* Preferred camera rating
* Preferred performance rating
* Preferred design rating
* Preferred display rating
* Preferred engagement level
* Preferred product segment
* Number of recommendations

The user inputs are converted into the same eight-feature vector used by the trained model.

### Recommendation Pipeline

```text
User Preferences
       ↓
8-Feature Vector
       ↓
Saved StandardScaler
       ↓
Product Feature Matrix
       ↓
Cosine Similarity
       ↓
Similarity Ranking
       ↓
Top-N Products
```

This allows recommendations to be generated from a user's stated product preferences rather than requiring the user to select an existing mobile.

---

# 📱 Mode 2 — Similar Product Recommendation

Users can select an existing:

```text
Brand | Model
```

The application creates a feature vector for the selected product and compares it with other products.

The selected product itself is excluded from the recommendation results.

### Process

```text
Selected Product
       ↓
Product Feature Vector
       ↓
StandardScaler
       ↓
Cosine Similarity
       ↓
Exclude Selected Product
       ↓
Similarity Ranking
       ↓
Top-N Similar Products
```

---

# ⭐ Recommendation Ranking

The system returns recommendations according to:

```text
Similarity Score
```

Higher similarity indicates that the product has a more similar feature profile to the selected product or user preference vector.

The application supports a configurable:

```text
Top-N
```

recommendation count.

---

# 📁 Recommendation Outputs

The recommendation module generates:

```text
recommendations/all_product_recommendations.csv
recommendations/recommendation_metadata.json
```

The recommendation CSV contains fields such as:

```text
Selected_Brand
Selected_Model
Selected_Segment
Selected_Price_USD
Selected_Rating
Recommended_Brand
Recommended_Model
Recommended_Segment
Recommended_Price_USD
Recommended_Rating
Recommended_Battery
Recommended_Camera
Recommended_Performance
Recommended_Design
Recommended_Display
Recommended_Engagement
Similarity_Score
Recommendation_Rank
```

---

# 6️⃣ Insights & Reporting

## File

```text
Step6_Insights.py
```

The Insights module analyzes the clustered product dataset and generates analytical reports.

---

# 📊 Insights Generated

The module analyzes:

* Cluster-wise product characteristics
* Product performance
* High-performing products
* Low-performing products
* Price-range performance
* Product attribute patterns
* Brand performance
* Segment rating patterns

---

# 📈 Performance Analysis

A product performance score is calculated using:

```text
rating
battery_life_rating
camera_rating
performance_rating
design_rating
display_rating
```

The engagement score is treated separately because it represents customer interaction rather than a direct product-quality rating.

The analysis is used to identify observed product attribute patterns associated with ratings and engagement.

---

# 📁 Insights Outputs

The module generates:

```text
insights/cluster_wise_analysis.csv
insights/high_performing_products.csv
insights/low_performing_products.csv
insights/price_range_performance.csv
insights/product_attribute_patterns.csv
insights/brand_performance_analysis.csv
insights/segment_rating_analysis.csv
insights/cluster_distribution.png
insights/price_vs_performance.png
insights/mobile_product_insights_report.txt
```

---

# 7️⃣ Streamlit Application

## File

```text
Step7_Streamlit.py
```

The Streamlit application provides an interactive interface for exploring the Machine Learning results and generating recommendations.

---

# 🖥️ Streamlit Pages

The application contains the following main sections:

```text
🏠 Dashboard
📊 Product Segmentation
⭐ Product Performance
💰 Price vs Performance
🎯 User Preferences
📱 Similar Product
📄 Insights Report
```

---

# 🏠 Dashboard

The Dashboard provides an overview of the project.

It can display:

* Total products
* Average product price
* Average rating
* Segment information
* Product-level summary information

---

# 📊 Product Segmentation

This section allows users to explore the product segments generated by K-Means.

The application displays segment-related information from:

```text
clustered_mobile_products.csv
```

The three final segments are:

```text
Budget
Mid-Range
Premium
```

---

# ⭐ Product Performance

This section provides information about product ratings and specification performance.

Users can examine:

* Product ratings
* Battery ratings
* Camera ratings
* Performance ratings
* Design ratings
* Display ratings
* Engagement

---

# 💰 Price vs Performance

This section visualizes the relationship between:

```text
Price
```

and:

```text
Product Performance
```

The application uses Matplotlib for visualization.

---

# 🎯 User Preferences

This is a core recommendation feature of the final Streamlit application.

Users can enter:

```text
Preferred Price
Preferred Rating
Battery Preference
Camera Preference
Performance Preference
Design Preference
Display Preference
Engagement Preference
Preferred Segment
Top-N
```

The entered preferences are converted into the same eight-feature structure used during model development.

The application then calculates cosine similarity against the available product profiles.

---

# 📱 Similar Product

Users can select an existing mobile product using:

```text
Brand | Model
```

The system compares the selected product with other products and displays similar recommendations.

The selected product is excluded from its own recommendation list.

---

# 📄 Insights Report

The application loads the generated analytical report:

```text
insights/mobile_product_insights_report.txt
```

This allows users to review project findings directly from the Streamlit interface.

---

# 🔐 Feature Consistency

A critical implementation requirement of this project is that the same eight features must be used consistently across clustering, recommendation, and Streamlit.

## Exact Feature Order

```text
1. price_usd
2. rating
3. battery_life_rating
4. camera_rating
5. performance_rating
6. design_rating
7. display_rating
8. engagement_score
```

The saved scaler:

```text
mobile_scaler.pkl
```

is fitted using these eight features.

The Streamlit application validates:

```python
scaler.n_features_in_ == 8
```

before generating recommendations.

This prevents feature mismatch errors between training and prediction.

---

# 🧠 Machine Learning Pipeline

The overall Machine Learning pipeline is:

```text
Raw Data
   ↓
Data Cleaning
   ↓
Missing Value Handling
   ↓
Feature Engineering
   ↓
Product-Level Aggregation
   ↓
Feature Selection
   ↓
StandardScaler
   ↓
K-Means Clustering
   ↓
Cluster Evaluation
   ↓
Product Segmentation
   ↓
Cosine Similarity
   ↓
Recommendations
   ↓
Insights
   ↓
Streamlit Dashboard
```

---

# 📐 StandardScaler

StandardScaler transforms numerical features so that they are placed on a comparable standardized scale.

The transformation is important because the project contains variables with very different ranges.

For example:

```text
price_usd              → hundreds/thousands
rating                 → 1–5
battery_life_rating    → 1–5
camera_rating          → 1–5
performance_rating     → 1–5
design_rating          → 1–5
display_rating         → 1–5
engagement_score       → variable range
```

Without scaling, high-range variables such as price could have a disproportionate effect on distance-based algorithms.

---

# 📊 K-Means Clustering — Simple Explanation

K-Means is an unsupervised Machine Learning algorithm that groups similar observations into clusters.

In this project:

```text
K = 3
```

The algorithm groups mobile products according to similarities in:

```text
Price
Rating
Battery
Camera
Performance
Design
Display
Engagement
```

### Viva Explanation

> "K-Means clustering groups mobile products with similar numerical characteristics into three segments. StandardScaler is applied before clustering so that features with different numerical ranges are comparable."

---

# 📏 Silhouette Score — Simple Explanation

Silhouette Score evaluates how well data points fit within their assigned clusters.

A higher score generally indicates:

* Better cluster cohesion
* Better separation between clusters

### Viva Explanation

> "Silhouette Score helps evaluate whether the generated clusters are compact internally and sufficiently separated from one another."

---

# 📐 Cosine Similarity — Simple Explanation

Cosine Similarity compares two feature vectors based on their direction.

### Viva Explanation

> "Cosine Similarity is used to compare mobile-product feature profiles and identify products with similar characteristics."

---

# 💡 Business & Analytical Use Cases

The system can support several analytical use cases.

## Customer Perspective

Users can discover mobile products similar to:

* Their preferred specifications
* Their preferred price range
* Their preferred rating
* An existing mobile product

## Product Perspective

The system can identify groups of products with similar characteristics.

## Marketing Perspective

Brand, price, rating, specification, and engagement patterns can be analyzed for market-oriented insights.

## Portfolio Perspective

Product segments and performance patterns can be examined to understand the structure of the available mobile portfolio.

---

# ⚠️ Project Limitations

The current system has several limitations.

### 1. Synthetic / Review-Based Data

The analysis is based on the available mobile review dataset and therefore reflects the characteristics of that dataset.

### 2. Limited Product Count

Although the dataset contains 50,000 reviews, product-level aggregation results in approximately 22 mobile products.

### 3. No Direct Purchase-Choice Label

The recommendation system is based on product-feature similarity rather than actual purchase history.

### 4. Similarity-Based Recommendation

Cosine Similarity identifies products with similar feature vectors. It does not learn complex behavioral patterns from individual users.

### 5. Static Dataset

The current application operates on locally generated project files rather than continuously updated real-time market data.

### 6. Limited Personalization

User preference recommendations are based on manually entered preferences rather than a long-term user interaction history.

---

# 🚀 Future Enhancements

Possible improvements include:

1. Hybrid recommendation system.
2. User-personalized recommendation models.
3. Sentiment-aware recommendations.
4. Collaborative filtering.
5. Content-based + collaborative filtering hybrid models.
6. Deep-learning-based recommendation models.
7. Real-time mobile product and price updates.
8. Additional clustering algorithms such as DBSCAN and Agglomerative Clustering.
9. Customer-level segmentation.
10. Cloud deployment.
11. Model monitoring.
12. Mobile or web deployment.
13. Database integration.
14. User login and recommendation history.
15. Larger real-world product datasets.

---

# ▶️ Installation

## Step 1 — Create a Virtual Environment

```bash
python -m venv .venv
```

---

## Step 2 — Activate the Environment

### Windows

```bash
.venv\Scripts\activate
```

---

## Step 3 — Install Required Libraries

```bash
python -m pip install pandas numpy scikit-learn matplotlib joblib streamlit
```

If a `requirements.txt` file is available:

```bash
python -m pip install -r requirements.txt
```

---

# ▶️ How to Run the Project

Run the project modules in sequence.

## 1. Data Collection

```bash
python Step1_Data_Collection.py
```

---

## 2. Data Preprocessing

```bash
python Step2_Data_Preprocessing.py
```

This generates:

```text
cleaned_mobile_reviews.csv
```

---

## 3. Exploratory Data Analysis

```bash
python Step3_EDA.py
```

This generates the EDA outputs.

---

## 4. K-Means Clustering

```bash
python Step4_Clustering.py
```

This generates:

```text
clustered_mobile_products.csv
cluster_profile_summary.csv
elbow_inertia.csv
mobile_scaler.pkl
mobile_kmeans_model.pkl
clustering_metadata.pkl
```

and visualization files.

---

## 5. Recommendation System

```bash
python Step5_Recommendation.py
```

This generates:

```text
recommendations/all_product_recommendations.csv
recommendations/recommendation_metadata.json
```

---

## 6. Insights & Reporting

```bash
python Step6_Insights.py
```

This generates the files inside:

```text
insights/
```

---

## 7. Launch Streamlit

```bash
python -m streamlit run Step7_Streamlit.py
```

Alternatively:

```bash
streamlit run Step7_Streamlit.py
```

---

# ⚠️ Important Execution Order

The project modules depend on files generated by previous stages.

Therefore, the recommended execution order is:

```text
Step 1
  ↓
Step 2
  ↓
Step 3
  ↓
Step 4
  ↓
Step 5
  ↓
Step 6
  ↓
Step 7
```

The Streamlit application should use the final generated files from the same project run.

This helps prevent:

* Model/data mismatch
* Missing output files
* Feature mismatch
* Scaler mismatch
* Recommendation errors

---

# 📦 Expected Outputs

After completing the full pipeline, the project should contain outputs such as:

```text
cleaned_mobile_reviews.csv
clustered_mobile_products.csv
cluster_profile_summary.csv
elbow_inertia.csv
```

Model files:

```text
mobile_scaler.pkl
mobile_kmeans_model.pkl
clustering_metadata.pkl
```

Recommendation files:

```text
recommendations/all_product_recommendations.csv
recommendations/recommendation_metadata.json
```

Insight files:

```text
insights/cluster_wise_analysis.csv
insights/high_performing_products.csv
insights/low_performing_products.csv
insights/price_range_performance.csv
insights/product_attribute_patterns.csv
insights/brand_performance_analysis.csv
insights/segment_rating_analysis.csv
insights/mobile_product_insights_report.txt
```

Visualization files:

```text
elbow_curve.png
cluster_segmentation.png
segment_distribution.png
```

and:

```text
insights/cluster_distribution.png
insights/price_vs_performance.png
```

---

# 🧪 Project Validation Checklist

Before final submission, verify the following:

* [ ] Raw dataset loads successfully.
* [ ] Dataset shape is verified.
* [ ] Missing values are checked.
* [ ] Missing values are handled correctly.
* [ ] Duplicate records are checked.
* [ ] Cleaned dataset is generated.
* [ ] `engagement_score` is generated.
* [ ] EDA runs successfully.
* [ ] Product-level aggregation works.
* [ ] All eight clustering features exist.
* [ ] StandardScaler is fitted correctly.
* [ ] K-Means uses `K = 3`.
* [ ] Elbow Method is generated.
* [ ] Silhouette Score is calculated.
* [ ] Cluster labels are generated.
* [ ] Segment names are assigned from cluster profiles.
* [ ] Clustered product dataset is generated.
* [ ] Recommendation system runs successfully.
* [ ] Recommendation system uses the same eight features.
* [ ] Saved scaler expects eight features.
* [ ] Cosine Similarity is calculated correctly.
* [ ] User preference recommendation works.
* [ ] Similar-product recommendation works.
* [ ] Selected product is excluded from its own recommendations.
* [ ] Insights files are generated.
* [ ] Streamlit loads successfully.
* [ ] Streamlit uses the final generated files.
* [ ] All required Python packages are installed.

---

# 🔍 Important Technical Consistency Check

The following feature list must be identical in Steps 4, 5, and 7:

```python
[
    "price_usd",
    "rating",
    "battery_life_rating",
    "camera_rating",
    "performance_rating",
    "design_rating",
    "display_rating",
    "engagement_score"
]
```

The order must also remain unchanged.

The saved scaler:

```text
mobile_scaler.pkl
```

must be fitted using exactly these eight features.

This consistency prevents errors such as:

```text
X has 7 features, but StandardScaler is expecting 8 features
```

---

# 🧩 Recommendation System Architecture

## User Preference Mode

```text
User
 ↓
Enter Preferred Characteristics
 ↓
Create 8-Feature Preference Vector
 ↓
StandardScaler
 ↓
Compare Against Product Feature Matrix
 ↓
Cosine Similarity
 ↓
Optional Segment Filter
 ↓
Sort by Similarity
 ↓
Top-N Recommendations
```

## Similar Product Mode

```text
User
 ↓
Select Brand + Model
 ↓
Retrieve Product Features
 ↓
StandardScaler
 ↓
Cosine Similarity
 ↓
Exclude Selected Product
 ↓
Sort by Similarity
 ↓
Top-N Recommendations
```

---

# 📊 Project Architecture

```text
                 ┌──────────────────────────┐
                 │   Mobile Review Dataset  │
                 └────────────┬─────────────┘
                              ↓
                 ┌──────────────────────────┐
                 │    Data Preprocessing    │
                 └────────────┬─────────────┘
                              ↓
                 ┌──────────────────────────┐
                 │          EDA             │
                 └────────────┬─────────────┘
                              ↓
                 ┌──────────────────────────┐
                 │ Product Feature Creation │
                 └────────────┬─────────────┘
                              ↓
                 ┌──────────────────────────┐
                 │     StandardScaler       │
                 └────────────┬─────────────┘
                              ↓
                 ┌──────────────────────────┐
                 │      K-Means (K=3)       │
                 └────────────┬─────────────┘
                              ↓
                 ┌──────────────────────────┐
                 │   Product Segmentation   │
                 └────────────┬─────────────┘
                              ↓
             ┌────────────────┴────────────────┐
             ↓                                 ↓
┌─────────────────────────┐       ┌─────────────────────────┐
│ Similar Product Mode    │       │ User Preference Mode    │
└────────────┬────────────┘       └────────────┬────────────┘
             └────────────────┬────────────────┘
                              ↓
                 ┌──────────────────────────┐
                 │    Cosine Similarity     │
                 └────────────┬─────────────┘
                              ↓
                 ┌──────────────────────────┐
                 │ Top-N Recommendations    │
                 └────────────┬─────────────┘
                              ↓
                 ┌──────────────────────────┐
                 │   Streamlit Dashboard    │
                 └──────────────────────────┘
```

---

# 🎓 Viva / Interview Explanation

## What is the project?

> "This project is a Machine Learning-based Mobile Product Segmentation and Recommendation System. It processes mobile review data, creates product-level features, groups products using K-Means clustering, and recommends similar products using Cosine Similarity."

## Why K-Means?

> "K-Means is an unsupervised learning algorithm suitable for grouping products based on similarities in their numerical characteristics."

## Why StandardScaler?

> "The features have different numerical ranges, especially price and rating. StandardScaler converts them to a comparable scale before distance-based Machine Learning calculations."

## Why K = 3?

> "The Elbow Method was used to compare different cluster counts. The analysis indicated K = 3 as the selected clustering configuration for this project."

## Why Cosine Similarity?

> "Cosine Similarity compares product feature vectors based on their direction, making it suitable for identifying products with similar feature profiles."

## What is the recommendation input?

> "The application supports two modes. Users can either select an existing mobile product or enter their preferred price, rating, specifications, engagement level, and segment."

## What is engagement score?

> "Engagement score is derived from helpful votes using log1p transformation. It represents customer interaction with reviews and is included as one of the product-level recommendation and clustering features."

---

# ⚠️ Important Interpretation Note

The recommendation system is a **similarity-based system**.

It does not claim that a recommended product will definitely be purchased or preferred by a customer.

The recommendations represent products whose available feature profiles are similar to:

* The selected product, or
* The preferences entered by the user.

Similarly, the clustering results represent statistical groupings of products based on the selected features.

---

# 🚀 Future Development

The project can be extended with:

### Recommendation Improvements

* Collaborative filtering
* Hybrid recommendation
* User interaction history
* Purchase history
* Rating prediction
* Sentiment-aware recommendation

### Machine Learning Improvements

* DBSCAN
* Agglomerative Clustering
* Gaussian Mixture Models
* Advanced similarity metrics
* Hyperparameter optimization

### Application Improvements

* User accounts
* Recommendation history
* Database integration
* Real-time product data
* Price alerts
* Cloud deployment
* Mobile application

---

# 🏆 Final Outcome

The completed project provides an end-to-end Data Science and Machine Learning workflow:

```text
DATA
 ↓
DATA CLEANING
 ↓
FEATURE ENGINEERING
 ↓
EDA
 ↓
PRODUCT-LEVEL AGGREGATION
 ↓
FEATURE SCALING
 ↓
K-MEANS CLUSTERING
 ↓
PRODUCT SEGMENTATION
 ↓
COSINE SIMILARITY
 ↓
RECOMMENDATIONS
 ↓
INSIGHTS & REPORTING
 ↓
STREAMLIT APPLICATION
```

The project demonstrates practical knowledge of:

**Python + Pandas + NumPy + Data Cleaning + EDA + Feature Engineering + StandardScaler + Unsupervised Learning + K-Means + Cluster Evaluation + Cosine Similarity + Recommendation Systems + Data Visualization + Streamlit**

---

# 📌 Project Category

**Machine Learning / Data Science**

## Key Concepts Demonstrated

* Data collection
* Data preprocessing
* Missing-value handling
* Duplicate checking
* Feature engineering
* Exploratory Data Analysis
* Feature scaling
* Product-level aggregation
* Unsupervised Machine Learning
* K-Means clustering
* Elbow Method
* Silhouette Score
* Cosine Similarity
* Recommendation systems
* Product segmentation
* Data visualization
* Business insights
* Interactive Streamlit dashboards

---

# ⭐ Project Summary

> **The Mobile Product Segmentation & Recommendation System transforms mobile review data into meaningful product segments and similarity-based recommendations using Machine Learning. The system combines data preprocessing, feature engineering, exploratory data analysis, K-Means clustering, Cosine Similarity, analytical reporting, and an interactive Streamlit application to provide an end-to-end data-driven product analysis and recommendation solution.**

---

# 👨‍💻 Project Information

**Project:** Mobile Product Segmentation & Recommendation System

**Project Type:** Machine Learning / Data Science

**Primary Language:** Python

**Clustering Algorithm:** K-Means

**Number of Clusters:** 3

**Recommendation Technique:** Cosine Similarity

**Application Framework:** Streamlit

**Dataset Size:** 50,000 review records

**Product-Level Records:** Approximately 22 products

**Core ML Features:** 8

```text
price_usd
rating
battery_life_rating
camera_rating
performance_rating
design_rating
display_rating
engagement_score
```

---

## 🎯 Final System

```text
Mobile Reviews
      ↓
Preprocessing
      ↓
EDA
      ↓
Feature Engineering
      ↓
K-Means (K=3)
      ↓
Budget / Mid-Range / Premium
      ↓
Cosine Similarity
      ↓
User Preference Recommendations
      +
Similar Product Recommendations
      ↓
Insights
      ↓
Streamlit Dashboard
```

**End of README**
