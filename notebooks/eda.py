import pandas as pd
# Load dataset
df = pd.read_csv("../data/app_user_behavior_dataset.csv")

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
