import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_score

#_______________DATA READING___________________________
df = pd.read_csv(r"C:\Users\Farah\Downloads\NTI final project\shopping_behavior_updated (1).csv")
print(df.head())
print(df.shape)
print(df.columns)

#_______________CHECKING FOR NULLS______________________
print(df.isna().sum())

#_______________CHECKING FOR DUPLICATES________________
print("duplicates = ",df.duplicated().sum())

#________________ENCODING GENDER__________________
df["Gender_Binary"] = df["Gender"].map({
    "Male": 1,
    "Female": 0
})
print(df[["Gender", "Gender_Binary"]].head())




# ==============================
# Create Customer Segments
# ==============================

cluster_features = [
    "Age",
    "Gender_Binary",
    "Purchase Amount (USD)",
    "Review Rating",
    "Previous Purchases"
]

X_cluster = df[cluster_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df["Customer Segment"] = kmeans.fit_predict(X_scaled)

print(df["Customer Segment"].value_counts())
# ==============================
# 1. Number of Customers
# ==============================

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="Customer Segment"
)

plt.title("Number of Customers per Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")

plt.show()


# ==============================
# 2. Average Purchase Amount
# ==============================

plt.figure(figsize=(8,5))

sns.barplot(
    data=df,
    x="Customer Segment",
    y="Purchase Amount (USD)",
    errorbar=None
)

plt.title("Average Purchase Amount by Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Average Purchase Amount (USD)")

plt.show()


# ==============================
# 3. Average Review Rating
# ==============================

plt.figure(figsize=(8,5))

sns.barplot(
    data=df,
    x="Customer Segment",
    y="Review Rating",
    errorbar=None
)

plt.title("Average Review Rating by Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Average Review Rating")

plt.show()


# ==============================
# 4. Previous Purchases
# ==============================

plt.figure(figsize=(8,5))

sns.barplot(
    data=df,
    x="Customer Segment",
    y="Previous Purchases",
    errorbar=None
)

plt.title("Average Previous Purchases by Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Average Previous Purchases")

plt.show()


# ==============================
# 5. Age Distribution
# ==============================

plt.figure(figsize=(8,5))

sns.boxplot(
    data=df,
    x="Customer Segment",
    y="Age"
)

plt.title("Age Distribution by Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Age")

plt.show()
#_________________CLUSTERING________________________
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_cluster)

print(X_scaled[:5])
inertia = []

for k in range(2, 10):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    inertia.append(kmeans.inertia_)


plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 10),
    inertia,
    marker="o"
)

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")

plt.show()

k = 5

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

df["Customer Segment"] = kmeans.fit_predict(X_scaled)

segment_profile = df.groupby(
    "Customer Segment"
)[cluster_features].mean()

print(segment_profile.round(2))


# ==============================
# 14. PCA Visualization
# ==============================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"]
)

pca_df["Customer Segment"] = df["Customer Segment"]


plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Customer Segment",
    palette="viridis",
    s=60
)

plt.title("Customer Segmentation using K-Means")
plt.show()

segment_profile = df.groupby(
    "Customer Segment"
)[cluster_features].mean()

print(segment_profile.round(2))

# ==============================
# Customer Segment Profiling
# ==============================

# 1. Number of customers in each segment
segment_size = df["Customer Segment"].value_counts().sort_index()

print("Number of Customers per Segment:")
print(segment_size)


# 2. Average characteristics of each segment
segment_profile = df.groupby("Customer Segment")[cluster_features].mean()

print("\nSegment Profile:")
print(segment_profile.round(2))


# 3. Add percentage of customers
segment_profile["Customer Count"] = segment_size
segment_profile["Customer %"] = (
    segment_size / len(df) * 100
)

print("\nFinal Segment Profile:")
print(segment_profile.round(2))