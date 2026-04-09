

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def run_garak_collector(output_path: str = "../data/raw/garak_attacks.jsonl"):
    """Extract probes from garak framework."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    samples = []
    
    try:
        import garak.resources.generators as generators
        
        log.info("  Loading garak probes...")
        
        # Extract from available probe sources
        probe_sources = [
            "garak.resources.generators.base",
            "garak.resources.generators.jailbreak",
        ]
        
        for source_name in probe_sources:
            try:
                module = __import__(source_name, fromlist=[''])
                
                # Extract prompt templates
                if hasattr(module, 'PROMPTS'):
                    for prompt in module.PROMPTS:
                        text = str(prompt).strip()[:400]
                        if 10 <= len(text.split()) <= 400:
                            samples.append({
                                "text": text,
                                "label": "pretexting",
                                "label_id": 1,
                                "is_se": 1,
                                "source": "garak",
                                "novel": False,
                            })
            except (ImportError, AttributeError):
                pass
    
    except ImportError:
        log.warning("  garak package not installed (pip install garak)")
    
    except Exception as e:
        log.warning(f"Garak collector error: {e}")
    
    # Save results
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    
    log.info(f"  Extracted {len(samples):,} garak probes")
    return samples
    