# 📱 Mobile Product Segmentation and Recommendation System

## 📌 Project Overview

The **Mobile Product Segmentation and Recommendation System** is a Machine Learning project developed using **Python, Pandas, Scikit-learn, and Streamlit**.

The project analyzes mobile product and customer review data, performs data preprocessing and exploratory data analysis, segments mobile products using **K-Means clustering**, and generates similar-product recommendations using **Cosine Similarity**.

The final system provides an interactive **Streamlit dashboard** where users can explore product segments, analyze mobile-product performance, and obtain recommendations.

---

## 🎯 Project Objectives

The main objectives of this project are:

* Clean and preprocess mobile review and product data.
* Perform exploratory data analysis (EDA).
* Engineer useful product and engagement features.
* Analyze relationships between price, ratings, and specifications.
* Segment mobile products into meaningful groups using K-Means.
* Evaluate clustering quality using appropriate metrics.
* Build a product recommendation system using Cosine Similarity.
* Generate business and customer preference insights.
* Develop an interactive Streamlit application.

---

## 🗂️ Project Workflow

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
6. Cluster Analysis & Insights
          ↓
7. Recommendation System
          ↓
8. Streamlit Application
          ↓
Business Insights & Recommendations
```

---

# 📊 Dataset

### Dataset Name

`Mobile Reviews Sentiment null.csv`

### Dataset Size

* **Rows:** 50,000
* **Columns:** 22
* **Brands:** 7
* **Countries:** 8
* **Rating Range:** 1.0 – 5.0
* **Approximate Price Range:** $180 – $1,500

### Main Columns

| Column                 | Description              |
| ---------------------- | ------------------------ |
| `review_id`            | Unique review identifier |
| `customer_name`        | Customer name            |
| `age`                  | Customer age             |
| `brand`                | Mobile phone brand       |
| `model`                | Mobile model             |
| `price_usd`            | Product price in USD     |
| `price_local`          | Local currency price     |
| `currency`             | Currency                 |
| `exchange_rate_to_usd` | Exchange rate            |
| `rating`               | Customer rating          |
| `sentiment`            | Review sentiment         |
| `country`              | Customer country         |
| `language`             | Review language          |
| `review_date`          | Review date              |
| `verified_purchase`    | Purchase verification    |
| `battery_life_rating`  | Battery rating           |
| `camera_rating`        | Camera rating            |
| `performance_rating`   | Performance rating       |
| `design_rating`        | Design rating            |
| `display_rating`       | Display rating           |
| `helpful_votes`        | Helpful votes            |
| `source`               | Review source            |

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Plotly
* Streamlit
* Joblib

### Machine Learning

* K-Means Clustering
* StandardScaler
* Cosine Similarity
* Silhouette Score
* Inertia / Elbow Method

---

# 📁 Project Structure

```text
Mobile_Product_Segmentation/
│
├── data/
│   ├── Mobile Reviews Sentiment null.csv
│   └── cleaned_mobile_reviews.csv
│
├── recommendations/
│   └── all_product_recommendations.csv
│
├── insights/
│   ├── cluster_wise_analysis.csv
│   ├── high_performing_products.csv
│   ├── low_performing_products.csv
│   ├── brand_analysis.csv
│   └── customer_preference_patterns.csv
│
├── models/
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   └── clustering_features.pkl
│
├── Step1_Data_Collection.py
├── Step2_Data_Preprocessing.py
├── Step3_EDA.py
├── Step4_Clustering.py
├── Step5_Recommendation.py
├── Step6_Streamlit.py
├── Step7_Insights_Reporting.py
│
├── requirements.txt
└── README.md
```

> File names may be adjusted according to the final project folder structure.

---

# 🔄 Project Modules

## 1️⃣ Data Collection

**File:**

```text
Step1_Data_Collection.py
```

This module:

* Loads the raw CSV dataset.
* Displays dataset shape.
* Displays column names.
* Checks data types.
* Provides an initial understanding of the dataset.

### Output

Initial dataset information.

---

# 2️⃣ Data Preprocessing

**File:**

```text
Step2_Data_Preprocessing.py
```

The preprocessing stage performs:

* Missing-value handling
* Duplicate checking
* Data-type conversion
* Numerical feature preparation
* Categorical encoding
* Feature engineering
* Feature scaling

### Important Features

The project uses features related to:

* Price
* Customer rating
* Battery life
* Camera
* Performance
* Design
* Display
* Customer engagement

### Output

```text
cleaned_mobile_reviews.csv
```

---

# 3️⃣ Exploratory Data Analysis

**File:**

```text
Step3_EDA.py
```

EDA is performed to understand the underlying patterns in the dataset.

### Analysis Performed

* Brand distribution
* Country distribution
* Top-rated products
* Lowest-rated products
* Rating distribution
* Price vs rating
* Price vs performance
* Price vs camera rating
* Price vs battery rating
* Brand-wise average rating
* Brand-wise average price
* Specification comparison
* Correlation analysis

### Output

```text
eda_specification_summary.csv
eda_brand_summary.csv
```

EDA helps determine which variables are useful for subsequent machine-learning analysis.

---

# 4️⃣ K-Means Clustering

**File:**

```text
Step4_Clustering.py
```

K-Means clustering is used to segment mobile products into groups with similar characteristics.

### Clustering Features

The major features include:

```text
price_usd
rating
battery_life_rating
camera_rating
performance_rating
design_rating
display_rating
```

Categorical variables such as brand/model are appropriately transformed before machine-learning processing.

### Preprocessing

Numerical features are standardized using:

```text
StandardScaler
```

### Number of Clusters

The project uses:

```text
K = 4
```

### Cluster Evaluation

The clustering process considers:

* Inertia
* Elbow Method
* Silhouette Score

### Output

```text
clustered_mobile_reviews.csv
```

---

# 🔵 Product Segmentation

The four clusters can be interpreted based on their feature profiles.

Typical interpretations include:

### Cluster 0 — Value / Budget Segment

Products with relatively lower prices and suitable specifications.

### Cluster 1 — Balanced Segment

Products offering a balance between price and overall specifications.

### Cluster 2 — Performance Segment

Products characterized by stronger performance-related specifications.

### Cluster 3 — Premium Segment

Higher-priced products with stronger overall specification profiles.

> Cluster numbers themselves do not inherently mean "budget", "premium", etc. The business meaning is assigned after analyzing each cluster's characteristics.

---

# 5️⃣ Recommendation System

**File:**

```text
Step5_Recommendation.py
```

The recommendation module uses **Cosine Similarity** to identify products with similar feature profiles.

### Product Profile

Recommendations consider characteristics such as:

```text
price_usd
rating
battery_life_rating
camera_rating
performance_rating
design_rating
display_rating
```

Products are organized using:

```text
brand
model
```

### Recommendation Process

```text
Selected Product
       ↓
Product Feature Vector
       ↓
Cosine Similarity
       ↓
Similarity Ranking
       ↓
Top Similar Products
```

### Output

```text
recommendations/all_product_recommendations.csv
```

---

# 6️⃣ Streamlit Application

**File:**

```text
Step6_Streamlit.py
```

The Streamlit application provides an interactive interface for exploring the project results.

### Main Application Features

* Dashboard
* Product overview
* Cluster analysis
* Interactive filtering
* Product recommendations
* EDA visualizations
* Business insights
* KPI cards
* Interactive charts

### Application Flow

```text
User
 ↓
Streamlit Dashboard
 ↓
Select / Filter Product
 ↓
View Product Information
 ↓
Explore Cluster
 ↓
Generate Similar Product Recommendations
```

---

# 7️⃣ Insights & Reporting

**File:**

```text
Step7_Insights_Reporting.py
```

This module generates business-oriented insights from the clustered dataset.

### Analysis Includes

* Cluster-wise analysis
* High-performing products
* Low-performing products
* Brand analysis
* Customer preference patterns
* Price vs performance analysis

### Main Outputs

```text
insights/
```

Example files:

```text
cluster_wise_analysis.csv
high_performing_products.csv
low_performing_products.csv
brand_analysis.csv
customer_preference_patterns.csv
```

---

# 📈 Machine Learning Methodology

## K-Means Clustering

K-Means divides products into **K groups** based on similarity.

For this project:

```text
K = 4
```

The algorithm attempts to minimize the distance between data points and their assigned cluster centers.

### Simple Explanation

> "Products having similar price, ratings and specifications are grouped together."

---

# 📐 StandardScaler

StandardScaler is used before clustering so that variables with different numerical ranges do not dominate the distance calculation.

Example:

```text
Price → hundreds/thousands
Rating → 1–5
```

Scaling converts the features into comparable standardized values.

---

# 📊 Silhouette Score

Silhouette Score evaluates how well the observations fit within their assigned clusters.

### Simple Definition

> A higher silhouette score generally indicates better-separated and more cohesive clusters.

---

# 📏 Cosine Similarity

Cosine Similarity is used in the recommendation system.

### Simple Definition

> Cosine Similarity measures how similar two product feature vectors are based on their direction.

A value closer to:

```text
1 → highly similar
0 → less similar
```

---

# 💡 Business Insights

The system can support several business decisions.

### Customer Perspective

Customers can discover alternatives similar to a selected mobile product.

### Product Perspective

Businesses can identify different product segments.

### Marketing Perspective

Brand and specification preferences can support targeted marketing.

### Portfolio Perspective

High-performing and low-performing products can be analyzed separately.

---

# 🚀 Future Enhancements

Possible improvements include:

1. Hybrid recommendation system.
2. User-personalized recommendations.
3. Sentiment-aware recommendations.
4. Deep-learning-based recommendation models.
5. Real-time data updates.
6. Cloud deployment.
7. Model monitoring.
8. Additional clustering algorithms.
9. Customer-level segmentation.
10. Mobile/web deployment.

---

# ▶️ How to Run the Project

## Step 1 — Create Virtual Environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

---

## Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit plotly openpyxl joblib
```

---

# ▶️ Run the Project

Run the scripts in the following order:

### 1. Data Collection

```bash
python Step1_Data_Collection.py
```

### 2. Preprocessing

```bash
python Step2_Data_Preprocessing.py
```

### 3. EDA

```bash
python Step3_EDA.py
```

### 4. Clustering

```bash
python Step4_Clustering.py
```

### 5. Recommendation System

```bash
python Step5_Recommendation.py
```

### 6. Generate Insights

```bash
python Step7_Insights_Reporting.py
```

### 7. Launch Streamlit

```bash
streamlit run Step6_Streamlit.py
```

---

# 📦 Expected Outputs

After executing the complete pipeline, the project should generate outputs such as:

```text
cleaned_mobile_reviews.csv
clustered_mobile_reviews.csv
recommendations/all_product_recommendations.csv
insights/cluster_wise_analysis.csv
insights/high_performing_products.csv
insights/low_performing_products.csv
insights/brand_analysis.csv
```

Model files may include:

```text
models/kmeans_model.pkl
models/scaler.pkl
models/clustering_features.pkl
```

---

# ⚠️ Important Execution Order

The scripts depend on outputs created by earlier stages.

Therefore, use:

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
Step 7
  ↓
Step 6 / Streamlit
```

The Streamlit application should use the **final generated files from the same project run** to avoid model/data mismatch.

---

# 🧪 Project Validation Checklist

Before final submission, verify:

* [ ] Raw dataset loads successfully.
* [ ] Missing values are handled.
* [ ] Duplicate records are checked.
* [ ] Cleaned dataset is generated.
* [ ] EDA runs without errors.
* [ ] Required clustering features exist.
* [ ] StandardScaler is fitted correctly.
* [ ] K-Means uses 4 clusters.
* [ ] Silhouette score is calculated.
* [ ] Clustered dataset is generated.
* [ ] Recommendation file is generated.
* [ ] Recommendation results are sensible.
* [ ] Step 7 generates insight files.
* [ ] Streamlit loads the same final outputs.
* [ ] All required Python packages are installed.

---

# 🏆 Final Outcome

The completed system provides an end-to-end Machine Learning solution:

```text
DATA
 ↓
PREPROCESSING
 ↓
EDA
 ↓
FEATURE ENGINEERING
 ↓
K-MEANS CLUSTERING
 ↓
PRODUCT SEGMENTATION
 ↓
COSINE-SIMILARITY RECOMMENDATION
 ↓
BUSINESS INSIGHTS
 ↓
STREAMLIT DASHBOARD
```

The project demonstrates practical knowledge of:

**Python + Data Cleaning + EDA + Feature Engineering + Unsupervised Learning + Clustering + Similarity-Based Recommendation + Data Visualization + Streamlit**

---

## 👨‍💻 Project Category

**Machine Learning / Data Science**

### Key Concepts Demonstrated

* Data preprocessing
* Exploratory Data Analysis
* Feature engineering
* Feature scaling
* Unsupervised Machine Learning
* K-Means clustering
* Cluster evaluation
* Cosine similarity
* Recommendation systems
* Data visualization
* Interactive dashboards
* Business insights

---

## ⭐ Project Summary

> **The Mobile Product Segmentation and Recommendation System transforms raw mobile review data into meaningful product segments and similar-product recommendations using Machine Learning. The system combines data preprocessing, exploratory analysis, K-Means clustering, Cosine Similarity, and Streamlit visualization to provide an end-to-end data-driven decision-support solution.**
