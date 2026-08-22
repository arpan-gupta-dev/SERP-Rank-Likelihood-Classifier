import pandas as pd
import joblib

def predict():
    model = joblib.load('seo_model.pkl')
    features = joblib.load('model_features.pkl')

    sample_page = {
        'content_length': 2100,
        'keyword_density': 0.024,
        'num_internal_links': 12,
        'num_external_links': 4,
        'has_meta_description': 1,
        'has_alt_text': 1,
        'avg_time_on_page_sec': 145,
        'bounce_rate': 0.42,
        'scroll_depth_percent': 68.5,
        'domain_authority': 55,
        'page_authority': 48,
        'backlink_count': 350,
        'serp_position_before': 18
    }

    input_data = pd.DataFrame([sample_page])[features]

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    if prediction == 1:
        print("Prediction: Ranking is likely to IMPROVE!")
    else:
        print("Prediction: Ranking is NOT likely to improve.")

    print(f"Confidence (Improvement Probability): {probabilities[1] * 100:.2f}%")

if __name__ == '__main__':
    predict()