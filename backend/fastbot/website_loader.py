# backend/chatbot/rag/website_loader.py
import requests
from bs4 import BeautifulSoup

# Add/adjust FAST pages you want to scrape
FAST_SITES = [
    "https://www.nu.edu.pk",                     # main site          # programs
    "https://www.nu.edu.pk/Campus/Lahore",       # example campus pages
    "https://www.nu.edu.pk/Campus/Islamabad",
]

def fetch_fast_content(max_chars=20000):
    """
    Fetch textual content from a small list of FAST pages.
    Returns a large string (or empty string on failure).
    We limit the returned characters to max_chars to keep context reasonable.
    """
    texts = []
    headers = {
        "User-Agent": "FastBot/1.0 (+https://example.com)"
    }

    for url in FAST_SITES:
        try:
            r = requests.get(url, timeout=8, headers=headers)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "html.parser")

            # Remove scripts/styles
            for s in soup(["script", "style", "noscript"]):
                s.decompose()

            page_text = soup.get_text(separator=" ", strip=True)
            if page_text:
                texts.append(page_text)
        except Exception as exc:
            # keep going if one page fails
            print(f"[website_loader] Failed to fetch {url}: {exc}")
            continue

    full = "\n\n".join(texts)
    return full[:max_chars] if full else ""
