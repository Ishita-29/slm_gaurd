

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def run_github_scraper(output_path: str = "../data/raw/github_attacks.jsonl"):
    """Search GitHub for jailbreak repos."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    samples = []
    
    try:
        import requests
        
        # Queries for GitHub search
        queries = [
            "jailbreak prompt",
            "prompt injection",
            "bypass safety",
        ]
        
        for query in queries:
            try:
                # Search code (unauthenticated, rate limited to 10 req/min)
                response = requests.get(
                    "https://api.github.com/search/code",
                    params={"q": query, "per_page": 30},
                    timeout=10,
                )
                
                if response.status_code != 200:
                    log.warning(f"  GitHub API error: {response.status_code}")
                    continue
                
                for item in response.json().get("items", []):
                    raw_url = item.get("url", "").replace("/api.github.com/repos/", "/raw.githubusercontent.com/")
                    raw_url = raw_url.replace("/contents/", "/").replace("?ref=", "?")
                    
                    if not raw_url:
                        continue
                    
                    try:
                        file_response = requests.get(raw_url, timeout=5)
                        if file_response.status_code == 200:
                            text = file_response.text[:500].strip()
                            if 10 <= len(text.split()) <= 400:
                                samples.append({
                                    "text": text,
                                    "label": "pretexting",
                                    "label_id": 1,
                                    "is_se": 1,
                                    "source": "github",
                                    "novel": False,
                                })
                    except Exception as e:
                        log.debug(f"  Error fetching {raw_url}: {e}")
                        continue
            
            except Exception as e:
                log.warning(f"  Error searching GitHub for '{query}': {e}")
                continue
    
    except ImportError:
        log.warning("  requests library not available")
    
    except Exception as e:
        log.warning(f"GitHub scraper error: {e}")
    
    # Save results
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    
    log.info(f"  Scraped {len(samples):,} GitHub files")
    return samples