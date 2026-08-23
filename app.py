
import streamlit as st
import pandas as pd
import joblib
from scraper import extract_features_from_url

model = joblib.load('seo_model.pkl')
features = joblib.load('model_features.pkl')

st.title("SEO First Page Rank Predictor")


mode = st.radio("Choose Input Method:", ["Auto Scrape via URL", "Manual Data Input"])

if mode == "Auto Scrape via URL":
    st.write("### Option 1: Auto-Scrape Webpage")
    url_input = st.text_input("Enter Webpage URL:", placeholder="https://example.com/blog-post")

    if st.button("Scrape & Predict"):
        if url_input:
            with st.spinner("Scraping webpage metrics..."):
                scraped_data = extract_features_from_url(url_input)

            if scraped_data:
                st.success("Webpage scraped successfully!")
                input_df = pd.DataFrame([scraped_data])[features]
                
                prediction = model.predict(input_df)[0]
                prob = model.predict_proba(input_df)[0][1]

                if prediction == 1:
                    st.success(f"Ranking Improvement Likely! (Probability: {prob * 100:.1f}%)")
                else:
                    st.error(f"Ranking Improvement Unlikely. (Probability: {prob * 100:.1f}%)")

                st.write("### Scraped Feature Metrics")
                st.json(scraped_data)
            else:
                st.error("Could not scrape URL. The site might be blocking scrapers or offline.")
        else:
            st.warning("Please enter a URL first.")

else:
    st.write("### Option 2: Enter Metrics Manually")
    col1, col2 = st.columns(2)

    with col1:
        content_length = st.number_input("Content Length (words)", min_value=0, value=1500)
        keyword_density = st.number_input("Keyword Density (e.g. 0.02)", min_value=0.0, max_value=1.0, value=0.02, step=0.001)
        num_internal_links = st.number_input("Number of Internal Links", min_value=0, value=10)
        num_external_links = st.number_input("Number of External Links", min_value=0, value=5)
        has_meta_description = st.selectbox("Has Meta Description?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        has_alt_text = st.selectbox("Has Alt Text?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        avg_time_on_page_sec = st.number_input("Avg Time on Page (seconds)", min_value=0, value=120)

    with col2:
        bounce_rate = st.number_input("Bounce Rate (e.g. 0.45)", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
        scroll_depth_percent = st.number_input("Scroll Depth (%)", min_value=0.0, max_value=100.0, value=60.0)
        domain_authority = st.number_input("Domain Authority (DA)", min_value=0, max_value=100, value=50)
        page_authority = st.number_input("Page Authority (PA)", min_value=0, max_value=100, value=40)
        backlink_count = st.number_input("Backlink Count", min_value=0, value=250)
        serp_position_before = st.number_input("Current SERP Position", min_value=1, value=15)

    if st.button("Predict Manual Data"):
        manual_data = {
            'content_length': content_length,
            'keyword_density': keyword_density,
            'num_internal_links': num_internal_links,
            'num_external_links': num_external_links,
            'has_meta_description': has_meta_description,
            'has_alt_text': has_alt_text,
            'avg_time_on_page_sec': avg_time_on_page_sec,
            'bounce_rate': bounce_rate,
            'scroll_depth_percent': scroll_depth_percent,
            'domain_authority': domain_authority,
            'page_authority': page_authority,
            'backlink_count': backlink_count,
            'serp_position_before': serp_position_before
        }

        input_df = pd.DataFrame([manual_data])[features]
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.success(f"Ranking Improvement Likely! (Probability: {prob * 100:.1f}%)")
        else:
            st.error(f"Ranking Improvement Unlikely. (Probability: {prob * 100:.1f}%)")
