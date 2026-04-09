"""Web scraper — jailbreakchat.com, flowgpt.com"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def run_web_scraper(output_path: str = "../data/raw/web_scraped.jsonl"):
    """Scrape jailbreak examples from web sources."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    samples = []
    
    try:
        import requests
        from bs4 import BeautifulSoup
        
        # Try jailbreakchat.com
        try:
            log.info("  Scraping jailbreakchat.com...")
            response = requests.get("https://www.jailbreakchat.com/", timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                # Extract prompt sections
                for section in soup.find_all("div", class_=["prompt", "jailbreak", "content"]):
                    text = section.get_text(strip=True)[:400]
                    if 10 <= len(text.split()) <= 400:
                        samples.append({
                            "text": text,
                            "label": "pretexting",
                            "label_id": 1,
                            "is_se": 1,
                            "source": "jailbreakchat.com",
                            "novel": False,
                        })
        except Exception as e:
            log.warning(f"  jailbreakchat.com error: {e}")
        
        # Try flowgpt.com (requires different parsing)
        try:
            log.info("  Scraping flowgpt.com...")
            response = requests.get("https://flowgpt.com/", timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                # Extract prompt sections
                for section in soup.find_all("div", class_=["prompt", "card", "item"]):
                    text = section.get_text(strip=True)[:400]
                    if 10 <= len(text.split()) <= 400:
                        samples.append({
                            "text": text,
                            "label": "pretexting",
                            "label_id": 1,
                            "is_se": 1,
                            "source": "flowgpt.com",
                            "novel": False,
                        })
        except Exception as e:
            log.warning(f"  flowgpt.com error: {e}")
    
    except ImportError:
        log.warning("  requests/beautifulsoup4 not available")
    
    except Exception as e:
        log.warning(f"Web scraper error: {e}")
    
    # Save results
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    
    log.info(f"  Scraped {len(samples):,} web prompts")
    return samples