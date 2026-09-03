# Mobile Product Segmentation and Recommendation System

## 📱 Project Overview

This project develops an end-to-end **Mobile Product Segmentation and Recommendation System** using Python and Machine Learning.

The system analyzes mobile product prices, ratings, and specifications to:

- Understand product characteristics
- Explore product and brand-level patterns
- Segment mobile products into four groups using **K-Means Clustering**
- Recommend similar mobile products using **Cosine Similarity**
- Provide interactive visualizations through **Streamlit**
- Generate business-oriented insights and reports

---

## 🎯 Objectives

1. Collect and understand mobile product review data.
2. Clean and preprocess the dataset.
3. Perform Exploratory Data Analysis (EDA).
4. Segment products into four similarity-based groups.
5. Build a similarity-based recommendation system.
6. Develop an interactive Streamlit application.
7. Generate insights to support data-driven decisions.

---

## 🗂️ Project Structure

```text
Mobile_Product_Project/
│
├── Mobile Reviews Sentiment null.csv
│
├── Step1_Data_Collection.py
├── Step2_Data_Preprocessing.py
├── Step3_EDA.py
├── Step4_Clustering.py
├── Step5_Recommendation.py
├── Step6_Streamlit.py
├── Step7_Insights_Reporting.py
│
├── cleaned_mobile_reviews.csv
├── clustered_mobile_reviews.csv
├── cluster_profile_summary.csv
│
├── recommendations/
│   └── all_product_recommendations.csv
│
└── insights/
    ├── cluster_wise_analysis.csv
    ├── high_performing_products.csv
    ├── low_performing_products.csv
    ├── price_range_performance.csv
    ├── customer_preference_patterns.csv
    ├── brand_performance_analysis.csv
    ├── price_vs_performance.png
    └── mobile_product_insights_report.txt
```

---

## 🔄 Project Workflow

```text
Raw Dataset
     ↓
Data Collection
     ↓
Data Cleaning & Preprocessing
     ↓
Exploratory Data Analysis
     ↓
K-Means Product Segmentation
     ↓
Cosine Similarity Recommendation
     ↓
Streamlit Application
     ↓
Insights & Reporting
```

---

# 1. Data Collection

### File

```text
Step1_Data_Collection.py
```

The original CSV dataset is loaded and inspected.

The data collection stage checks:

- Dataset shape
- Column names
- Data types
- Missing values
- Duplicate records
- Statistical summary
- First few records

### Input

```text
Mobile Reviews Sentiment null.csv
```

---

# 2. Data Cleaning & Preprocessing

### File

```text
Step2_Data_Preprocessing.py
```

The preprocessing stage performs:

- Column name cleaning
- Numeric data conversion
- Missing value handling
- Median imputation for numerical columns
- Mode/Unknown handling for categorical values
- Duplicate removal
- Infinite value handling

Important project design decision:

> **No one-hot encoding and no feature scaling are performed during preprocessing.**

The original `brand` and `model` information is preserved because they are required for product identification and downstream analysis.

### Output

```text
cleaned_mobile_reviews.csv
```

---

# 3. Exploratory Data Analysis

### File

```text
Step3_EDA.py
```

EDA is performed using the cleaned dataset.

The analysis includes:

- Product distribution by brand
- Top-rated products
- Low-rated products
- Price analysis
- Price vs rating analysis
- Price vs specification analysis
- Rating distribution
- Correlation analysis
- Brand-wise comparison
- Specification-wise comparison

EDA uses the original business-scale values.

Scaling is not required during this stage.

### Outputs

```text
eda_specification_summary.csv
eda_brand_summary.csv
```

---

# 4. K-Means Product Segmentation

### File

```text
Step4_Clustering.py
```

K-Means clustering is used to group products with similar characteristics.

### Features Used

- Price
- Rating
- Battery life rating
- Camera rating
- Performance rating
- Design rating
- Display rating

### Machine Learning Process

```text
Product Features
       ↓
Missing Value Handling
       ↓
StandardScaler
       ↓
Elbow Method
       ↓
K-Means Clustering
       ↓
4 Clusters
       ↓
Product Segments
```

The project uses:

```text
Number of Clusters = 4
```

The clusters are interpreted using average price as:

1. Budget
2. Mid-Range
3. Upper Mid-Range
4. Premium

The **Elbow Method** and **Silhouette Score** are used to evaluate the clustering structure.

### Outputs

```text
clustered_mobile_reviews.csv
cluster_profile_summary.csv
```

---

# 5. Recommendation System

### File

```text
Step5_Recommendation.py
```

A **content-based similarity recommendation system** is implemented.

The system uses:

- Price
- Rating
- Battery life
- Camera
- Performance
- Design
- Display

### Recommendation Process

```text
Product Profiles
       ↓
Feature Standardization
       ↓
Cosine Similarity
       ↓
Similarity Scores
       ↓
Ranking
       ↓
Top 5 Similar Products
```

Recommendations are generated for every unique product.

### Output

```text
recommendations/all_product_recommendations.csv
```

The recommendation output contains:

- Selected product
- Recommended product
- Recommendation rank
- Recommended price
- Recommended rating
- Recommended specifications
- Similarity score

### Validation

Recommendation relevance is assessed using similarity scores.

The system categorizes overall relevance as:

- **HIGH**
- **MODERATE**
- **LOW**

> Similarity score is used as a relevance indicator. It is not recommendation accuracy because the dataset does not contain ground-truth user recommendation outcomes.

---

# 6. Streamlit Application

### File

```text
Step6_Streamlit.py
```

An interactive web application is developed using **Streamlit**.

The application contains three major sections.

### Dashboard

Displays:

- Total products
- Number of brands
- Number of segments
- Average rating
- Price distribution
- Rating distribution

### Product Segmentation

Displays:

- Segment distribution
- Cluster information
- Price vs rating visualization
- Cluster profiles
- Business interpretation

### Recommendations

Users can select a mobile product and view:

- Selected product information
- Price
- Rating
- Specifications
- Product segment
- Top 5 similar products
- Similarity scores
- Similarity visualization
- Relevance interpretation

---

## ▶️ Running the Streamlit Application

Install the required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly streamlit
```

Run the application:

```bash
streamlit run Step6_Streamlit.py
```

The Streamlit application will open in the browser.

---

# 7. Insights & Reporting

### File

```text
Step7_Insights_Reporting.py
```

This stage converts analytical results into business-oriented insights.

The analysis includes:

### Product Segmentation

- Cluster size
- Segment percentage
- Average price
- Average rating
- Average performance

### High-Performing Products

The Top 10 products are identified using an overall performance score.

### Low-Performing Products

The Bottom 10 products are identified using the same performance measure.

### Price vs Performance

The relationship between price and overall product performance is analyzed.

### Customer Preference Patterns

Patterns are inferred from the average ratings of measured product attributes.

### Brand Analysis

Brands are compared using:

- Product count
- Average price
- Average rating
- Average performance

### Output Folder

```text
insights/
```

The folder contains CSV reports, a visualization, and a text-based insights report.

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning |
| Matplotlib | Visualization |
| Seaborn | Statistical visualization |
| Plotly | Interactive visualization |
| Streamlit | Web application |

---

# 🤖 Machine Learning Techniques

### K-Means Clustering

Used for:

> **Mobile product segmentation**

Products are grouped based on similarity in price, ratings, and specifications.

### Cosine Similarity

Used for:

> **Product recommendation**

Products with similar feature vectors receive higher similarity scores.

### StandardScaler

Used before:

- K-Means clustering
- Cosine similarity

Scaling is intentionally performed only in the Machine Learning stages.

---

# 📊 Key Project Outputs

The project produces:

- Cleaned dataset
- Clustered product dataset
- Cluster profile
- Product recommendations
- High-performing products
- Low-performing products
- Price-range analysis
- Brand performance analysis
- Customer preference patterns
- Price vs performance visualization
- Business insights report
- Interactive Streamlit dashboard

---

# 💡 Business Value

This system can support:

- Product portfolio analysis
- Market segmentation
- Product comparison
- Customer product discovery
- Competitive analysis
- Value-for-money analysis
- Data-driven product decisions

Businesses can use product segments to understand different market groups and use similarity-based recommendations to help users discover alternative products.

---

# ⚠️ Project Limitation

The dataset contains product-level information and ratings.

It does **not** contain individual customer purchase history or explicit user preference profiles.

Therefore:

- Customer preference patterns are inferred from product ratings and specifications.
- Recommendations are based on product-feature similarity.
- Recommendation validation uses similarity/relevance measures rather than actual user-click or purchase accuracy.

---

# 🚀 Future Enhancements

Future versions can include:

1. Review-text sentiment analysis.
2. User-specific recommendation history.
3. Hybrid recommendation systems.
4. Real-time product price updates.
5. Price prediction.
6. Advanced recommendation algorithms.
7. Online deployment of the Streamlit application.
8. User login and personalized recommendations.

---

# ✅ Conclusion

The project successfully implements an end-to-end data science workflow for mobile product analysis.

It combines:

```text
Data Collection
      +
Data Preprocessing
      +
EDA
      +
K-Means Clustering
      +
Cosine Similarity
      +
Streamlit
      +
Insights
```

The final system provides both **product segmentation** and **similar-product recommendations**, while presenting the results through an interactive application and structured analytical reports.

---

## 👨‍💻 Project Type

**End-to-End Data Science & Machine Learning Project**

**Domain:** Mobile Products / E-Commerce Analytics

**Tech Stack:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly, Streamlit

**Machine Learning:** K-Means Clustering, Cosine Similarity

---

## 📌 Execution Order

Run the files in the following order:

```text
1. Step1_Data_Collection.py
2. Step2_Data_Preprocessing.py
3. Step3_EDA.py
4. Step4_Clustering.py
5. Step5_Recommendation.py
6. Step6_Streamlit.py
7. Step7_Insights_Reporting.py
```

For the Streamlit application:

```bash
streamlit run Step6_Streamlit.py
```

For insights and reporting:

```bash
python Step7_Insights_Reporting.py
```

---

## 📄 Project Status

**Status: Completed**

The project covers the complete workflow from raw data collection to machine learning, recommendation, interactive visualization, and business insights.
