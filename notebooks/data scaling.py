import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("../data/app_user_behavior_dataset.csv")

# Feature selection (MUST be repeated)
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

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Scaling completed")
print(X_scaled[:5])
