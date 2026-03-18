import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ===============================
# LOAD DATA (FIXED PATH)
# ===============================
df = pd.read_csv("data/app_user_behavior_dataset.csv")

print("First 5 rows:")
print(df.head())

print("\nShape:", df.shape)
print("\nColumns:", df.columns)

# ===============================
# DATA CLEANING
# ===============================
print("\nMissing values before cleaning:")
print(df.isnull().sum())

df.drop_duplicates(inplace=True)

numeric_cols = df.select_dtypes(include=['int64','float64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# ===============================
# EDA
# ===============================
print("\nStatistical Summary:")
print(df.describe())

plt.figure()
sns.histplot(df["engagement_score"], bins=30, kde=True)
plt.title("Engagement Score Distribution")
plt.show()

plt.figure()
sns.histplot(df["days_since_last_login"], bins=30, kde=True)
plt.title("Days Since Last Login")
plt.show()

numeric_df = df.select_dtypes(include=['int64','float64'])
plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ===============================
# FEATURE SELECTION
# ===============================
features = [
"sessions_per_week","avg_session_duration_min","daily_active_minutes",
"feature_clicks_per_session","notifications_opened_per_week",
"in_app_search_count","pages_viewed_per_session","support_tickets_raised",
"days_since_last_login","ads_clicked_last_30_days","content_downloads",
"social_shares","rating_given","churn_risk_score",
"engagement_score","account_age_days"
]

X = df[features]

# ===============================
# SCALING
# ===============================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# ELBOW METHOD
# ===============================
from sklearn.cluster import KMeans

inertia = []
for k in range(2,11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(range(2,11), inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("K")
plt.ylabel("Inertia")
plt.show()

# ===============================
# CLUSTERING METHODS
# ===============================

# KMEANS
kmeans = KMeans(n_clusters=6, random_state=42)
df["kmeans_cluster"] = kmeans.fit_predict(X_scaled)

# ===============================
# HIERARCHICAL (SAFE VERSION)
# ===============================
from sklearn.cluster import AgglomerativeClustering

# Use only a sample to avoid memory crash
sample_size = 2000
X_sample = X_scaled[:sample_size]

hc = AgglomerativeClustering(n_clusters=6)
hc_labels = hc.fit_predict(X_sample)

# Create full column with default value
df["hc_cluster"] = -1

# Assign cluster labels only for sample
df.loc[:sample_size-1, "hc_cluster"] = hc_labels

print("\nHierarchical Cluster Counts (sampled):")
print(pd.Series(hc_labels).value_counts())

# DBSCAN
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=1.2, min_samples=10)
df["dbscan_cluster"] = dbscan.fit_predict(X_scaled)

# GMM
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=6, random_state=42)
df["gmm_cluster"] = gmm.fit_predict(X_scaled)

# ===============================
# CLUSTER DISTRIBUTION GRAPH
# ===============================
plt.figure(figsize=(10,6))

df["kmeans_cluster"].value_counts().sort_index().plot(label="KMeans")
df["hc_cluster"].value_counts().sort_index().plot(label="Hierarchical")
df["gmm_cluster"].value_counts().sort_index().plot(label="GMM")

plt.legend()
plt.title("Cluster Size Comparison")
plt.show()

# ===============================
# SILHOUETTE SCORE
# ===============================
from sklearn.metrics import silhouette_score

scores = {}

scores["KMeans"] = silhouette_score(X_scaled, df["kmeans_cluster"])
scores["Hierarchical"] = silhouette_score(X_scaled, df["hc_cluster"])
scores["GMM"] = silhouette_score(X_scaled, df["gmm_cluster"])

# DBSCAN SAFE HANDLING
mask = df["dbscan_cluster"] != -1

if len(set(df.loc[mask,"dbscan_cluster"])) > 1:
    scores["DBSCAN"] = silhouette_score(
        X_scaled[mask],
        df.loc[mask,"dbscan_cluster"]
    )
else:
    scores["DBSCAN"] = np.nan

print("\nSilhouette Scores:", scores)

# ===============================
# SILHOUETTE GRAPH
# ===============================
plt.bar(scores.keys(), scores.values())
plt.title("Silhouette Comparison")
plt.show()

# ===============================
# SUMMARY TABLE
# ===============================
summary = pd.DataFrame({
"Method":["KMeans","Hierarchical","DBSCAN","GMM"],
"Clusters":[
df["kmeans_cluster"].nunique(),
df["hc_cluster"].nunique(),
df["dbscan_cluster"].nunique(),
df["gmm_cluster"].nunique()
],
"Silhouette":[
scores["KMeans"],
scores["Hierarchical"],
scores["DBSCAN"],
scores["GMM"]
]
})

print(summary)

# ===============================
# FINAL KMEANS ANALYSIS
# ===============================
cluster_profile = df.groupby("kmeans_cluster")[features].mean()
cluster_counts = df["kmeans_cluster"].value_counts()

cluster_profile["user_count"] = cluster_counts

print(cluster_profile)

# ===============================
# CLUSTER NAMES
# ===============================
cluster_names = {
0:"New Active Users",
1:"Long-Term Regular Users",
2:"Established Balanced Users",
3:"Low-Volume Stable Users",
4:"Consistent High-Activity Users",
5:"Deep Session Users"
}

df["cluster_label"] = df["kmeans_cluster"].map(cluster_names)

# ===============================
# PCA VISUALIZATION
# ===============================
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(pca_data, columns=["PC1","PC2"])
pca_df["cluster"] = df["cluster_label"]

sns.scatterplot(x="PC1", y="PC2", hue="cluster", data=pca_df)
plt.title("PCA Clusters")
plt.show()

# ===============================
# SAVE OUTPUT
# ===============================
df.to_csv("outputs/app_user_clustered_data.csv", index=False)

print("SUCCESS: File saved")