import pandas as pd

# Load dataset
df = pd.read_csv("../data/app_user_behavior_dataset.csv")

# Initial inspection
print("First 5 rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

# Data info
print("\nDataset Info:")
print(df.info())

# Missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

# Handle missing values
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Verify cleaning
print("\nMissing values after cleaning:")
print(df.isnull().sum())

#EDA
#statistical summary
print("\nStatistical Summary:")
print(df.describe())

# DISTRIBUTION OF ENGAGEMENT SCORE
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))
sns.histplot(df["engagement_score"], bins=30, kde=True)
plt.title("Distribution of Engagement Score")
plt.xlabel("Engagement Score")
plt.ylabel("Frequency")
plt.show()

# DAYS SINCE LAST LOGIN
plt.figure(figsize=(8,5))
sns.histplot(df["days_since_last_login"], bins=30, kde=True)
plt.title("Days Since Last Login Distribution")
plt.xlabel("Days")
plt.ylabel("Users")
plt.show()

#CORRELATION HEATMAP
# Correlation heatmap (numeric features only)
plt.figure(figsize=(12,8))
numeric_df = df.select_dtypes(include=['int64', 'float64'])
sns.heatmap(numeric_df.corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

#FEATURE SELECTION
#checking columns
print(df.columns)

#FINAL FEATURES
features = [
    "sessions_per_week",
    "avg_session_duration_min",
    "daily_active_minutes",
    "feature_clicks_per_session",
    "notifications_opened_per_week",
    "in_app_search_count",
    "pages_viewed_per_session",
    "support_tickets_raised",
    "days_since_last_login",
    "ads_clicked_last_30_days",
    "content_downloads",
    "social_shares",
    "rating_given",
    "churn_risk_score",
    "engagement_score",
    "account_age_days"
]

X = df[features]


#VERIFY FEATURE MATRIX
print("\nSelected features preview:")
print(X.head())

print("\nFeature matrix shape:")
print(X.shape)


#DATA SCALING
from sklearn.preprocessing import StandardScaler
# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Scaling completed")
print(X_scaled[:5])

#ELBOW METHOD
from sklearn.cluster import KMeans

inertia = []

K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(K_range, inertia, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.show()

#Fit kmeans model
# Apply KMeans with optimal K
kmeans = KMeans(n_clusters=6, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to original dataframe
df["cluster"] = clusters

print("Cluster counts:")
print(df["cluster"].value_counts())

# Cluster profiling - mean values
cluster_profile = df.groupby("cluster")[features].mean()

print("\nCluster-wise feature averages:")
print(cluster_profile)

cluster_counts = df["cluster"].value_counts().sort_index()

print("\nUsers per cluster:")
print(cluster_counts)

cluster_summary = cluster_profile.copy()
cluster_summary["user_count"] = cluster_counts

print("\nCluster Summary:")
print(cluster_summary)

#FINAL CLUSTER NAMING
cluster_names = {
    0: "New Active Users",
    1: "Long-Term Regular Users",
    2: "Established Balanced Users",
    3: "Low-Volume Stable Users",
    4: "Consistent High-Activity Users",
    5: "Deep Session Users"
}

df["cluster_label"] = df["cluster"].map(cluster_names)
print(numeric_df)

#import PCA
from sklearn.decomposition import PCA

#APPLY PCA ON SCALED DATA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_scaled)

#CREATE PCA DATAFRAME
pca_df = pd.DataFrame(
    data=pca_components,
    columns=["PC1", "PC2"]
)

pca_df["cluster_label"] = df["cluster_label"]

#PLOT PCA CLUSTERS
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))
sns.scatterplot(
    x="PC1",
    y="PC2",
    hue="cluster_label",
    data=pca_df,
    palette="Set2",
    alpha=0.7
)

plt.title("PCA Visualization of User Clusters")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="User Segment")
plt.show()



# Save final clustered data
df.to_csv("../outputs/app_user_clustered_data.csv", index=False)

print("Final clustered data saved successfully!")
