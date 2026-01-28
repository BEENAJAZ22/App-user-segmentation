import pandas as pd
# Load dataset
df = pd.read_csv("../data/app_user_behavior_dataset.csv")

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


