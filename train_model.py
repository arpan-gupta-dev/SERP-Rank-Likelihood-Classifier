import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

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
    'serp_position_before',
]

X = data[features].copy()
y = data['ranking_improved']

X = X.fillna(X.median())

if 'backlink_count' in X.columns:
  X['backlink_count'] = np.log1p(X['backlink_count'])

scaler = RobustScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("=" * 50)
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Detailed Performance Breakdown:")
print(classification_report(y_test, y_pred))
print("=" * 50)

importances = pd.Series(model.feature_importances_, index=features).sort_values(
    ascending=False
)
print("\nMost Important SEO Features:")
print(importances)

joblib.dump(model, 'seo_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(features, 'model_features.pkl')

print("\nModel saved successfully!")
