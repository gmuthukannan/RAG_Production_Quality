"""
Gumloop Data Scraper
====================
Collects public information about Gumloop from their website and saves it
as plain text files in data/docs/ for ingestion into the vector store.

Usage:
    python scripts/scrape_gumloop.py

The scraper is intentionally conservative:
  - Respects robots.txt implicitly by targeting known public pages
  - Adds a delay between requests
  - Strips all HTML to extract clean text only

After running this, execute the ingestion pipeline:
    python -m src.pipelines.ingestion_pipeline
"""

import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import settings

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; GumloopRAGBot/1.0; "
        "Educational project - contact: your@email.com)"
    )
}

REQUEST_DELAY = 1.5  # seconds between requests — be polite

# ── Target pages ──────────────────────────────────────────────────────────────
# Extend this list as you discover more public Gumloop pages.
TARGET_URLS = [
    "https://www.gumloop.com",
    "https://www.gumloop.com/pricing",
    "https://www.gumloop.com/blog",
    "https://help.gumloop.com",
]


def scrape_page(url: str) -> str | None:
    """Fetches a URL and returns extracted plain text, or None on failure."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove noise
    for tag in soup(["script", "style", "nav", "footer", "header", "meta", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Collapse excessive blank lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def save_text(text: str, filename: str) -> None:
    out_path = Path(settings.docs_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    logger.info(f"Saved {len(text)} chars → {out_path}")


def run_scraper() -> None:
    logger.info(f"Starting Gumloop scraper — {len(TARGET_URLS)} target pages")

    for url in TARGET_URLS:
        logger.info(f"Scraping: {url}")
        text = scrape_page(url)
        if text:
            # Turn URL into a safe filename: https://gumloop.com/pricing → gumloop_pricing.txt
            slug = url.replace("https://", "").replace("http://", "")
            slug = slug.replace("/", "_").replace(".", "_").strip("_")
            save_text(text, f"scraped_{slug}.txt")
        time.sleep(REQUEST_DELAY)

    logger.success("Scraping complete. Run ingestion pipeline next:")
    logger.success("  python -m src.pipelines.ingestion_pipeline")


if __name__ == "__main__":
    run_scraper()