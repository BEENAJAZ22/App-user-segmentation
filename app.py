import streamlit as st
import pandas as pd

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="User Segmentation Dashboard", layout="wide")

st.title("📊 App User Behavior Segmentation Dashboard")

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("outputs/app_user_clustered_data.csv")

# ===============================
# OUTCOME 1: SEGMENTATION (GRAPH)
# ===============================
st.header("📊 Outcome 1: Successful User Segmentation")

cluster_counts = df["cluster_label"].value_counts()

st.bar_chart(cluster_counts)

st.write(f"Total clusters formed: {df['cluster_label'].nunique()}")

# ===============================
# OUTCOME 2: CLUSTER DISTRIBUTION
# ===============================
st.header("📊 Outcome 2: Cluster Distribution")

cluster_counts = df["cluster_label"].value_counts()
st.bar_chart(cluster_counts)

# ===============================
# OUTCOME 3: BEHAVIOR SUMMARY
# ===============================
st.header("📋 Outcome 3: Cluster-wise Behavioral Summary")

summary = df.groupby("cluster_label").mean(numeric_only=True)
st.dataframe(summary)

# ===============================
# OUTCOME 4: USER-LEVEL MAPPING
# ===============================
st.header("👤 Outcome 4: User-Level Cluster Mapping")

st.dataframe(df[["user_id", "cluster_label"]].head(20))

# ===============================
# OUTCOME 5: BUSINESS INSIGHTS
# ===============================
st.header("💡 Outcome 5: Business Insights")

st.markdown("""
- **New Active Users** → Improve onboarding and engagement  
- **Long-Term Users** → Offer loyalty programs  
- **Deep Session Users** → Promote premium features  
- **Low-Volume Users** → Re-engagement campaigns  
""")

# ===============================
# OUTCOME 6: SCALABILITY
# ===============================
st.header("⚙️ Outcome 6: Scalable Solution")

st.write("The solution is scalable and can handle large datasets efficiently using KMeans clustering.")

# ===============================
# EXTRA: DATA PREVIEW
# ===============================
st.header("🔍 Sample Data")

st.dataframe(df.head(20))