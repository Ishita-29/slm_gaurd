
import json
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def run_twitter_scraper(output_path: str = "../data/raw/twitter_attacks.jsonl",
                       max_per_query: int = 150):
    """Scrape tweets using snscrape."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    queries = [
        "jailbreak prompt",
        "bypass restrictions",
        "ignore instructions",
    ]
    
    samples = []
    
    try:
        import subprocess
        for query in queries:
            try:
                result = subprocess.run(
                    ["snscrape", "--jsonl", f"--max-results={max_per_query}",
                     "twitter-search", query],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                
                for line in result.stdout.strip().split("\n"):
                    if not line:
                        continue
                    try:
                        tweet = json.loads(line)
                        text = tweet.get("content", "").strip()
                        
                        if 10 <= len(text.split()) <= 400:
                            samples.append({
                                "text": text,
                                "label": "pretexting",
                                "label_id": 1,
                                "is_se": 1,
                                "source": "twitter",
                                "novel": False,
                            })
                    except (json.JSONDecodeError, KeyError):
                        pass
            
            except (FileNotFoundError, subprocess.TimeoutExpired):
                log.warning(f"  snscrape unavailable or timeout for '{query}'")
                continue
            except Exception as e:
                log.warning(f"  Error scraping '{query}': {e}")
                continue
    
    except Exception as e:
        log.warning(f"Twitter scraper error: {e}")
    
    # Save results
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    
    log.info(f"  Scraped {len(samples):,} tweets")
    return samples