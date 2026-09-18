# 🛍️ ShopSense – Customer Intelligence

## 📌 Project Overview

**ShopSense** is a Customer Intelligence project that analyzes customer shopping behavior using **Machine Learning**.

The project combines **Customer Segmentation (Clustering)** and **Classification** to understand different customer groups and predict the segment of a new customer.

The project also includes an interactive **Streamlit web application** for exploring the analysis and making predictions.

---

## 🎯 Project Objectives

* Analyze customer shopping behavior.
* Clean and prepare the dataset.
* Identify different customer segments using **K-Means Clustering**.
* Profile and understand each customer segment.
* Compare different classification models.
* Predict the customer segment for a new customer.
* Build an interactive Streamlit dashboard.

---

## 📊 Dataset

The dataset contains **3,900 customer records** and **16 features** related to shopping behavior.

### Main Features

* Customer ID
* Age
* Gender
* Item Purchased
* Category
* Purchase Amount (USD)
* Location
* Size
* Color
* Season
* Review Rating
* Subscription Status
* Discount Applied
* Previous Purchases
* Payment Method
* Frequency of Purchases

---

## 🧹 Data Preparation

The project includes:

* Checking missing values
* Checking duplicated records
* Removing duplicates
* Encoding categorical information
* Creating `Gender_Binary`
* Selecting relevant features for clustering
* Scaling numerical features using `StandardScaler`

---

## 🤖 Machine Learning

### 1. Customer Segmentation

**K-Means Clustering** is used to divide customers into groups with similar characteristics.

The clustering features include:

* Age
* Gender
* Purchase Amount
* Review Rating
* Previous Purchases

The number of clusters was evaluated using:

* **Elbow Method**
* **Silhouette Score**

The final model creates **5 customer segments**.

### 2. Customer
