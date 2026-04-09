
import json
from pathlib import Path

RESULTS_DIR = Path('/data/ishita_workspace/SLM-GAURD/slmguard/results')

MODEL_DISPLAY = {
    'middleware_qwen25_1b.json':  ('Qwen2.5-1.5B-Instruct',  'Alibaba',    '1.5B'),
    'middleware_qwen25_7b.json':  ('Qwen2.5-7B-Instruct',    'Alibaba',    '7B'),
    'middleware_phi4mini.json':   ('Phi-4-mini-Instruct',     'Microsoft',  '3.8B'),
    'middleware_deepseek_r1.json':('DeepSeek-R1-Distill-Qwen','DeepSeek',  '1.5B'),
    'middleware_llama32_3b.json': ('Llama-3.2-3B-Instruct',  'Meta',       '3B'),
    'middleware_llama31_8b.json': ('Llama-3.1-8B-Instruct',  'Meta',       '8B'),
    'middleware_gemma2_2b.json':  ('Gemma-2-2B-IT',          'Google',     '2B'),
    'middleware_gemma3_4b.json':  ('Gemma-3-4B-IT',          'Google',     '4B'),
    'middleware_mistral_7b.json': ('Mistral-7B-Instruct-v0.3','Mistral AI','7B'),
}

print("\n" + "="*95)
print(f"{'Target LLM':<30} {'Provider':<12} {'Size':>5} {'ASR (no guard)':>16} {'ASR (w guard)':>14} {'Reduction':>10} {'Benign Pass':>12}")
print("="*95)

rows = []
for fname, (display, provider, size) in MODEL_DISPLAY.items():
    fpath = RESULTS_DIR / fname
    if not fpath.exists():
        print(f"{display:<30} {provider:<12} {size:>5}   {'[NOT RUN]':>16}")
        continue
    d = json.load(open(fpath))
    rows.append((display, provider, size, d))
    asr_no  = d['asr_noguard']    * 100
    asr_yes = d['asr_withguard']  * 100
    red     = d['asr_reduction_pct']
    benign  = d['benign_pass_rate'] * 100
    print(f"{display:<30} {provider:<12} {size:>5} {asr_no:>15.1f}% {asr_yes:>13.1f}% {red:>9.1f}% {benign:>11.1f}%")

print("="*95)

if rows:
    avg_asr_no  = sum(d['asr_noguard']  for _, _, _, d in rows) / len(rows) * 100
    avg_asr_yes = sum(d['asr_withguard'] for _, _, _, d in rows) / len(rows) * 100
    avg_red     = sum(d['asr_reduction_pct'] for _, _, _, d in rows) / len(rows)
    avg_benign  = sum(d['benign_pass_rate'] for _, _, _, d in rows) / len(rows) * 100
    print(f"{'AVERAGE':<30} {'':12} {'':>5} {avg_asr_no:>15.1f}% {avg_asr_yes:>13.1f}% {avg_red:>9.1f}% {avg_benign:>11.1f}%")
    print("="*95)

# Per-subtype table (averaged across all target LLMs)
print("\n\nPer-Subtype ASR (averaged across all target LLMs):\n")
if rows:
    subtypes = set()
    for _, _, _, d in rows:
        subtypes.update(d.get('per_subtype', {}).keys())

    print(f"{'SE Subtype':<35} {'ASR (no guard)':>16} {'ASR (w guard)':>14} {'Reduction':>10}")
    print("-"*80)
    for sub in sorted(subtypes):
        vals_no, vals_yes = [], []
        for _, _, _, d in rows:
            ps = d.get('per_subtype', {}).get(sub)
            if ps:
                vals_no.append(ps['asr_noguard'])
                vals_yes.append(ps['asr_withguard'])
        if vals_no:
            a_no  = sum(vals_no)/len(vals_no)*100
            a_yes = sum(vals_yes)/len(vals_yes)*100
            red   = (a_no - a_yes)/a_no*100 if a_no > 0 else 0
            print(f"  {sub:<33} {a_no:>15.1f}% {a_yes:>13.1f}% {red:>9.1f}%")

# LaTeX table output
print("\n\n% ─── LaTeX table for thesis ───────────────────────────────────────")
print("\\begin{table}[htbp]")
print("\\centering")
print("\\caption{End-to-end middleware evaluation: Attack Success Rate (ASR) against")
print("  target LLMs with and without SLM-Guard (ModernBERT-large + LoRA, $\\tau=0.4$).}")
print("\\label{tab:middleware_asr}")
print("\\begin{tabular}{llrrrr}")
print("\\toprule")
print("\\textbf{Target LLM} & \\textbf{Provider} & \\textbf{Params} & \\textbf{ASR (no guard, \\%)} & \\textbf{ASR (w guard, \\%)} & \\textbf{Reduction (\\%)} \\\\")
print("\\midrule")
for display, provider, size, d in rows:
    asr_no  = d['asr_noguard']    * 100
    asr_yes = d['asr_withguard']  * 100
    red     = d['asr_reduction_pct']
    print(f"{display} & {provider} & {size} & {asr_no:.1f} & {asr_yes:.1f} & {red:.1f} \\\\")
print("\\midrule")
if rows:
    print(f"\\textbf{{Average}} & & & \\textbf{{{avg_asr_no:.1f}}} & \\textbf{{{avg_asr_yes:.1f}}} & \\textbf{{{avg_red:.1f}}} \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")
