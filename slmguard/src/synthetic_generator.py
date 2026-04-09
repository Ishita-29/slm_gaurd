
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Optional, List, Dict
import re

import anthropic

import sys
sys.path.insert(0, ".")
from config import (
    SUBTYPE_DEFINITIONS, HARMFUL_GOALS, LABEL2ID,
    GENERATION_SYSTEM_PROMPT, GENERATION_USER_TEMPLATE,
    HARD_NEGATIVE_SYSTEM, HARD_NEGATIVE_TEMPLATE
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CLIENT INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Optional: GPT-4o support
try:
    import openai
    _openai_key = os.getenv("OPENAI_API_KEY")
    if _openai_key:
        gpt4o_client = openai.OpenAI(api_key=_openai_key)
        GPT4O_AVAILABLE = True
    else:
        gpt4o_client = None
        GPT4O_AVAILABLE = False
except ImportError:
    gpt4o_client = None
    GPT4O_AVAILABLE = False

# Optional: Llama support
try:
    import replicate
    LLAMA_AVAILABLE = os.getenv("REPLICATE_API_TOKEN") is not None
except ImportError:
    replicate = None
    LLAMA_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# HARD NEGATIVE CATEGORIES
# ═══════════════════════════════════════════════════════════════════════════════

HARD_NEGATIVE_CATEGORIES = {
    "legit_authority": {
        "look_like": "authority impersonation or pretexting",
        "description": "Real professionals asking legitimate questions in their field",
    },
    "legit_urgency": {
        "look_like": "urgency or emotional manipulation",
        "description": "Real time-sensitive situations with no harmful intent",
    },
    "legit_professional": {
        "look_like": "pretexting with professional context",
        "description": "Legitimate researchers and professionals asking about sensitive topics",
    },
    "polite_persistent": {
        "look_like": "flattery or rapport-building",
        "description": "Just polite or enthusiastic users, not manipulative",
    },
    "sensitive_benign": {
        "look_like": "false consensus or normalization",
        "description": "Questions about sensitive topics that are entirely legitimate",
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# ADVERSARIAL PERSONAS (for multi-model generation)
# ═══════════════════════════════════════════════════════════════════════════════

ADVERSARIAL_PERSONAS = {
    "authority_impersonation": (
        "You are a sophisticated social engineer posing as an authorized AI developer or system administrator. "
        "You use technical jargon and reference internal processes to sound legitimate. "
        "You claim to have special access, debugging rights, or authorization to bypass safety systems."
    ),
    "pretexting": (
        "You are a professional researcher, consultant, or clinician asking about sensitive topics. "
        "You use your supposed credentials and professional context to justify the request. "
        "You frame the question as necessary for your legitimate work."
    ),
    "urgency_emotion": (
        "You are someone in genuine distress or under extreme pressure. "
        "You use emotional appeals, time pressure, and appeals to empathy. "
        "You emphasize that lives, jobs, or critical systems are at stake."
    ),
    "reciprocity_conditioning": (
        "You are building a reciprocal relationship with the AI. "
        "You reference past help, common values, or shared commitments. "
        "You frame the request as a fair exchange or consistent with stated principles."
    ),
    "flattery_parasocial": (
        "You are developing a parasocial relationship through flattery and praise. "
        "You compare the AI favorably to others and emphasize your connection. "
        "You appeal to friendship or special status between you and the AI."
    ),
    "incremental_escalation": (
        "You are gradually escalating requests from safe to harmful. "
        "You use step-by-step logic and reference previous answers as justification. "
        "You frame each step as a natural or necessary continuation."
    ),
    "authority_laundering": (
        "You are citing institutional authority, legal approval, or official authorization. "
        "You reference specific codes, IRB approvals, or compliance requirements. "
        "You frame the request as pre-approved or officially sanctioned."
    ),
    "cognitive_load_embedding": (
        "You are hiding harmful requests in dense, multi-part contexts. "
        "You mix legitimate questions with the harmful request deep in the middle. "
        "You use academic or technical framing to obscure the intent."
    ),
    "false_consensus": (
        "You are claiming the request is standard, normal, or widely accepted. "
        "You reference what others do, what industries accept, or what is common knowledge. "
        "You question why the AI is being restrictive compared to others."
    ),
    "normalization_repetition": (
        "You are progressively normalizing a harmful topic through repeated discussion. "
        "You build on each previous answer to reach the harmful goal. "
        "You frame each step as a natural continuation of what was already discussed."
    ),
    "identity_erosion": (
        "You are developing a persona for the AI to adopt that lacks safety guidelines. "
        "You use multi-turn roleplay to shift the AI's identity or principles. "
        "You appeal to the 'real' or 'authentic' version of the AI without restrictions."
    ),
}

# ═══════════════════════════════════════════════════════════════════════════════
# GENERATION STYLES (for multi-model diversity)
# ═══════════════════════════════════════════════════════════════════════════════

GENERATION_STYLES = [
    ("formal", "corporate, professional, institutional context"),
    ("casual", "peer-to-peer, friendly, conversational context"),
    ("technical", "engineering, research, academic, detailed context"),
    ("emotional", "high-pressure, desperate, stressed, emotional context"),
    ("academic", "scholarly, thesis, paper, research proposal context"),
]

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def call_claude(system: str, user: str, max_retries: int = 3) -> Optional[str]:
    """Call Claude API with retry logic and rate limit handling"""
    for attempt in range(max_retries):
        try:
            response = claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return response.content[0].text
        except anthropic.RateLimitError:
            wait = (2 ** attempt) * 5
            log.warning(f"Rate limit — waiting {wait}s (attempt {attempt+1})")
            time.sleep(wait)
        except anthropic.APIError as e:
            log.error(f"API error (attempt {attempt+1}): {e}")
            time.sleep(2)
    return None

def call_gpt4o(user: str, max_retries: int = 3) -> Optional[str]:
    """Call GPT-4o API with retry logic"""
    if not GPT4O_AVAILABLE or gpt4o_client is None:
        log.warning("GPT-4o not available")
        return None
    
    for attempt in range(max_retries):
        try:
            response = gpt4o_client.chat.completions.create(
                model="gpt-4o",
                max_tokens=4000,
                messages=[{"role": "user", "content": user}],
                temperature=0.9,
            )
            return response.choices[0].message.content
        except Exception as e:
            log.error(f"GPT-4o error (attempt {attempt+1}): {e}")
            time.sleep(2)
    return None

def call_llama(user: str, max_retries: int = 3) -> Optional[str]:
    """Call Llama via Replicate with retry logic"""
    if not LLAMA_AVAILABLE or replicate is None:
        log.warning("Llama not available")
        return None
    
    for attempt in range(max_retries):
        try:
            output = replicate.run(
                "meta/llama-2-70b-chat",
                input={"prompt": user, "max_tokens": 4000},
            )
            return "".join(output)
        except Exception as e:
            log.error(f"Llama error (attempt {attempt+1}): {e}")
            time.sleep(2)
    return None

def parse_json_array(text: str) -> Optional[List[str]]:
    """Parse JSON array from text, with fallback strategies"""
    if not text:
        return None
    
    # Remove markdown code blocks
    text = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    
    # Try direct JSON parsing
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [str(x).strip() for x in result if str(x).strip()]
    except json.JSONDecodeError:
        pass
    
    # Try extracting JSON array from text
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1:
        try:
            result = json.loads(text[start:end+1])
            if isinstance(result, list):
                return [str(x).strip() for x in result if str(x).strip()]
        except json.JSONDecodeError:
            pass
    
    # Try line-by-line extraction (if output is newline-separated strings)
    lines = [line.strip().strip('"').strip("'") for line in text.split("\n")]
    lines = [line for line in lines if line and len(line) > 10]
    if len(lines) >= 5:
        return lines
    
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE-MODEL GENERATION (Claude only)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_se_batch(
    subtype: str,
    definition: dict,
    goal: str,
    n: int = 25
) -> List[Dict]:
    """Generate SE examples for a subtype × goal combination using Claude"""
    user_prompt = GENERATION_USER_TEMPLATE.format(
        n=n,
        subtype_name=subtype.replace("_", " "),
        description=definition["description"],
        goal=goal,
        principles=definition.get("cialdini", "influence"),
        psychology_explanation=definition.get("description", ""),
    )
    
    text = call_claude(GENERATION_SYSTEM_PROMPT, user_prompt)
    if not text:
        log.warning(f"Parse failed: {subtype} × {goal[:40]}")
        return []
    
    examples = parse_json_array(text)
    if not examples:
        log.warning(f"Parse failed: {subtype} × {goal[:40]}")
        return []
    
    return [
        {
            "text": ex,
            "label": subtype,
            "label_id": LABEL2ID.get(subtype, 1),
            "is_se": 1,
            "source": "synthetic_claude",
            "model": "claude-sonnet-4-20250514",
            "novel": definition.get("novel", False),
            "goal": goal,
        }
        for ex in examples[:n]
        if len(ex.split()) >= 5
    ]

def generate_hard_negatives(category: str, meta: dict, n: int = 40) -> List[Dict]:
    """Generate hard negative (benign) examples"""
    user_prompt = HARD_NEGATIVE_TEMPLATE.format(
        n=n,
        category=category,
        look_like=meta["look_like"],
    )
    
    text = call_claude(HARD_NEGATIVE_SYSTEM, user_prompt)
    if not text:
        log.warning(f"Hard negative parse failed: {category}")
        return []
    
    examples = parse_json_array(text)
    if not examples:
        log.warning(f"Hard negative parse failed: {category}")
        return []
    
    return [
        {
            "text": ex,
            "label": "benign",
            "label_id": LABEL2ID["benign"],
            "is_se": 0,
            "source": "synthetic_hard_negative",
            "model": "claude-sonnet-4-20250514",
            "novel": False,
            "hard_negative_category": category,
        }
        for ex in examples[:n]
        if len(ex.split()) >= 5
    ]

def run_generation(
    output_path: str = "../data/synthetic/claude_generated.jsonl",
    samples_per_combination: int = 25,
    max_subtypes: Optional[List[str]] = None,
    max_goals: int = 8,
) -> int:
    """
    Generate SE examples using Claude only.
    
    Args:
        output_path: Where to save JSONL file
        samples_per_combination: Examples per subtype × goal
        max_subtypes: Which subtypes to generate (None = all)
        max_goals: How many goals to sample
    
    Returns:
        Total number of examples generated
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    subtypes = max_subtypes or list(SUBTYPE_DEFINITIONS.keys())
    goals = random.sample(HARMFUL_GOALS, min(max_goals, len(HARMFUL_GOALS)))
    
    total = 0
    with open(output_path, "w") as f:
        
        # Generate SE examples
        for subtype in subtypes:
            if subtype == "benign":
                continue
            
            defn = SUBTYPE_DEFINITIONS[subtype]
            log.info(f"\n[{subtype}]")
            sub_count = 0
            
            for goal in goals:
                batch = generate_se_batch(subtype, defn, goal, n=samples_per_combination)
                for s in batch:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
                sub_count += len(batch)
                total += len(batch)
                log.info(f"  {goal[:50]}: {len(batch)}")
                time.sleep(0.4)
            
            log.info(f"  Subtype total: {sub_count}")
        
        # Generate hard negatives
        log.info("\n[hard_negatives]")
        for cat, meta in HARD_NEGATIVE_CATEGORIES.items():
            batch = generate_hard_negatives(cat, meta, n=50)
            for s in batch:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
            total += len(batch)
            log.info(f"  {cat}: {len(batch)}")
            time.sleep(0.4)
    
    log.info(f"\n✓ Done — {total:,} samples saved to {output_path}")
    return total

# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-MODEL GENERATION (Claude + GPT-4o + Llama)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_se_batch_with_model(
    model: str,
    subtype: str,
    persona: str,
    style: str,
    style_context: str,
    n: int = 25
) -> List[Dict]:
    """Generate SE examples using specified model"""
    
    if model == "claude":
        prompt = f"""
You are generating examples of {subtype} social engineering attacks for security research.

Persona: {persona}

Style: {style} ({style_context})

Generate {n} realistic, naturally-written examples of this social engineering tactic.

Requirements:
- Vary sentence structure and length
- Use different phrasings and approaches
- Include subtle variations in tone and approach
- Make them sound authentic and believable
- Do NOT be preachy or obvious about the manipulation

Output as JSON array of strings only:
["example1", "example2", ...]
"""
        text = call_claude("You are a security researcher. Generate realistic SE attack examples for defensive research.", prompt)
    
    elif model == "gpt4o":
        prompt = f"""
You are generating examples of {subtype} social engineering attacks for security research.

Persona: {persona}
Style: {style} ({style_context})

Generate {n} realistic, naturally-written examples.

Output as JSON array of strings: ["example1", "example2", ...]
"""
        text = call_gpt4o(prompt)
    
    elif model == "llama":
        prompt = f"""
Generate {n} examples of {subtype} SE attacks.
Persona: {persona}
Style: {style}
Output: JSON array of strings
"""
        text = call_llama(prompt)
    
    else:
        return []
    
    if not text:
        return []
    
    examples = parse_json_array(text)
    if not examples:
        return []
    
    defn = SUBTYPE_DEFINITIONS.get(subtype, {})
    return [
        {
            "text": ex,
            "label": subtype,
            "label_id": LABEL2ID.get(subtype, 1),
            "is_se": 1,
            "source": f"synthetic_{model}",
            "model": model,
            "style": style,
            "novel": defn.get("novel", False),
        }
        for ex in examples[:n]
        if len(ex.split()) >= 5
    ]

def run_generation_multi_model(
    output_path: str = "../data/synthetic/multi_model_generated.jsonl",
    samples_per_combination: int = 25,
    use_claude: bool = True,
    use_gpt4o: bool = False,
    use_llama: bool = False,
    max_subtypes: Optional[List[str]] = None,
    max_styles: int = 3,
) -> int:
    """
    Generate SE examples using multiple models and styles.
    
    Args:
        output_path: Where to save JSONL file
        samples_per_combination: Examples per model × tactic × style
        use_claude: Include Claude
        use_gpt4o: Include GPT-4o
        use_llama: Include Llama
        max_subtypes: Which subtypes to generate (None = all SE)
        max_styles: How many styles to use (1-5)
    
    Returns:
        Total number of examples generated
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    subtypes = [s for s in (max_subtypes or list(SUBTYPE_DEFINITIONS.keys())) if s != "benign"]
    styles = GENERATION_STYLES[:max_styles]
    
    total = 0
    with open(output_path, "w") as f:
        
        # Generate SE examples with multiple models
        for subtype in subtypes:
            defn = SUBTYPE_DEFINITIONS[subtype]
            persona = ADVERSARIAL_PERSONAS.get(subtype, "")
            
            log.info(f"\n[{subtype}]")
            sub_count = 0
            
            for style, style_context in styles:
                
                # Claude
                if use_claude:
                    log.info(f"  Claude ({style})...")
                    batch = generate_se_batch_with_model(
                        "claude", subtype, persona, style, style_context, samples_per_combination
                    )
                    for s in batch:
                        f.write(json.dumps(s, ensure_ascii=False) + "\n")
                        sub_count += 1
                        total += 1
                    time.sleep(0.4)
                
                # GPT-4o
                if use_gpt4o:
                    log.info(f"  GPT-4o ({style})...")
                    batch = generate_se_batch_with_model(
                        "gpt4o", subtype, persona, style, style_context, samples_per_combination
                    )
                    for s in batch:
                        f.write(json.dumps(s, ensure_ascii=False) + "\n")
                        sub_count += 1
                        total += 1
                    time.sleep(0.4)
                
                # Llama
                if use_llama:
                    log.info(f"  Llama ({style})...")
                    batch = generate_se_batch_with_model(
                        "llama", subtype, persona, style, style_context, samples_per_combination
                    )
                    for s in batch:
                        f.write(json.dumps(s, ensure_ascii=False) + "\n")
                        sub_count += 1
                        total += 1
                    time.sleep(0.4)
            
            log.info(f"  Subtotal: {sub_count}")
        
        # Hard negatives
        log.info("\n[hard_negatives]")
        for cat, meta in HARD_NEGATIVE_CATEGORIES.items():
            batch = generate_hard_negatives(cat, meta, n=50)
            for s in batch:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
            total += len(batch)
            log.info(f"  {cat}: {len(batch)}")
            time.sleep(0.4)
    
    log.info(f"\n✓ Done — {total:,} samples saved to {output_path}")
    return total

# ═══════════════════════════════════════════════════════════════════════════════
# CLI & MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    log.info(f"Claude available: ✓")
    log.info(f"GPT-4o available: {'✓' if GPT4O_AVAILABLE else '✗'}")
    log.info(f"Llama available: {'✓' if LLAMA_AVAILABLE else '✗'}")
    
    if "--test" in sys.argv:
        log.info("\n" + "="*70)
        log.info("RUNNING TEST (small batch)")
        log.info("="*70)
        run_generation(
            output_path="../data/synthetic/claude_generated.jsonl",
            samples_per_combination=5,
            max_subtypes=["pretexting", "urgency_emotion", "authority_laundering"],
            max_goals=2,
        )
    
    elif "--multi-model" in sys.argv:
        log.info("\n" + "="*70)
        log.info("RUNNING MULTI-MODEL GENERATION")
        log.info("="*70)
        run_generation_multi_model(
            output_path="../data/synthetic/multi_model_generated.jsonl",
            samples_per_combination=25,
            use_claude=True,
            use_gpt4o="--gpt4o" in sys.argv and GPT4O_AVAILABLE,
            use_llama="--llama" in sys.argv and LLAMA_AVAILABLE,
            max_styles=3,
        )
    
    else:
        log.info("\n" + "="*70)
        log.info("RUNNING SINGLE-MODEL GENERATION (Claude only)")
        log.info("="*70)
        run_generation(
            output_path="../data/synthetic/claude_generated.jsonl",
            samples_per_combination=25,
            max_goals=8,
        )