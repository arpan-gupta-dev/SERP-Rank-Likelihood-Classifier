import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

def extract_features_from_url(url):
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=8)
        soup = BeautifulSoup(response.text, 'html.parser')

        text = soup.get_text(separator=' ')
        words = text.split()
        content_length = len(words)

        domain = urlparse(url).netloc
        internal_links = 0
        external_links = 0

        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.startswith('/') or domain in href:
                internal_links += 1
            elif href.startswith('http'):
                external_links += 1

        has_meta = 1 if soup.find('meta', attrs={'name': 'description'}) else 0
        has_alt = 1 if any(img.get('alt') for img in soup.find_all('img')) else 0

        return {
            'content_length': content_length,
            'keyword_density': 0.02,
            'num_internal_links': internal_links,
            'num_external_links': external_links,
            'has_meta_description': has_meta,
            'has_alt_text': has_alt,
            'avg_time_on_page_sec': 120,
            'bounce_rate': 0.45,
            'scroll_depth_percent': 65.0,
            'domain_authority': 45,
            'page_authority': 35,
            'backlink_count': 150,
            'serp_position_before': 15
        }
    except Exception as e:
        return None