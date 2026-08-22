import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

data = pd.read_csv('seo_dataset.csv')

features = [
    'content_length',
    'keyword_density',
    'num_internal_links',
    'num_external_links',
    'has_meta_description',
    'has_alt_text',
    'avg_time_on_page_sec',
    'bounce_rate',
    'scroll_depth_percent',
    'domain_authority',
    'page_authority',
    'backlink_count',
    'serp_position_before'
]

X = data[features]
y = data['ranking_improved']

X = X.fillna(X.median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Detailed Performance Breakdown:")
print(classification_report(y_test, y_pred))

importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\nMost Important SEO Features:")
print(importances)

joblib.dump(model, 'seo_model.pkl')
joblib.dump(features, 'model_features.pkl')

print("\nModel saved as 'seo_model.pkl'. Ready for predictions!")