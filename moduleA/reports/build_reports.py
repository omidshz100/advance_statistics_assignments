#!/usr/bin/env python3
"""Build evidence-based, bilingual Module A reports from the executed .ipynb files.

Requires markdown, weasyprint and Pillow. Fails if an expected recorded
measurement or real plot image is missing. No training is performed here.
"""
import base64
import html
import json
import re
import sys
from pathlib import Path

from PIL import Image
from markdown import markdown
from weasyprint import CSS, HTML

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "moduleA"
REPORTS = SRC / "reports"
FIGURES = REPORTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

NOTEBOOKS = {
    "q": SRC / "Assignment1" / "01 LAB_A Q_Learning.ipynb",
    "dqn": SRC / "Assignment1" / "02 LAB_A DQN.ipynb",
    "sarsa": SRC / "Assignment1" / "03 LAB_A SARSA.ipynb",
    "bonus": SRC / "Assignment1" / "04 LAB_A Bonus.ipynb",
    "reinforce": SRC / "Assignment2" / "04 LAB_A Reinforce.ipynb",
    "actor": SRC / "Assignment2" / "05 LAB_ A_Actor Critic.ipynb",
    "ppo": SRC / "Assignment2" / "06 LAB_A PPO.ipynb",
}
notebooks = {key: json.loads(path.read_text(encoding="utf-8"))
             for key, path in NOTEBOOKS.items()}

def code_cells(name):
    return [cell for cell in notebooks[name]["cells"] if cell["cell_type"] == "code"]

def source(cell):
    value = cell.get("source", [])
    return "".join(value) if isinstance(value, list) else value

def stream(name):
    texts = []
    for cell in code_cells(name):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                value = output.get("text", "")
                texts.append("".join(value) if isinstance(value, list) else value)
    return "\n".join(texts)

def required(pattern, data, label, group=1):
    match = re.search(pattern, data, re.M | re.S)
    if not match:
        raise RuntimeError("A required measured result was not recorded: " + label)
    return match.group(group)

q, sar, dqn, bonus = (stream(x) for x in ("q", "sarsa", "dqn", "bonus"))
reinforce, actor, ppo = (stream(x) for x in ("reinforce", "actor", "ppo"))
values = {
    "q_det_pct": f'{float(required(r"Greedy evaluation success, deterministic:\s*([0-9.]+)", q, "Q-learning deterministic")) * 100:.1f}',
    "q_slip_pct": f'{float(required(r"Greedy evaluation success, slippery:\s*([0-9.]+)", q, "Q-learning slippery")) * 100:.1f}',
    "sarsa_train": required(r"Last-500 training reward:\s*SARSA=([0-9.]+)", sar, "SARSA last-500"),
    "sarsa_qtrain": required(r"Last-500 training reward:.*?Q-learning=([0-9.]+)", sar, "Q-learning comparator"),
    "sarsa_eval_pct": f'{float(required(r"Greedy evaluation success:\s*SARSA=([0-9.]+)", sar, "SARSA greedy")) * 100:.1f}',
    "sarsa_qeval_pct": f'{float(required(r"Greedy evaluation success:.*?Q-learning=([0-9.]+)", sar, "Q-learning greedy")) * 100:.1f}',
    "dqn_mean": required(r"Mean reward over 100 episodes:\s*(-?[0-9.]+)", dqn, "DQN mean"),
    "dqn_sd": required(r"Std of rewards over 100 episodes:\s*([0-9.]+)", dqn, "DQN SD"),
    "bonus_mean": required(r"Held-out CartPole-v1 evaluation:\s*mean=([0-9.]+)", bonus, "bonus mean"),
    "bonus_sd": required(r"Held-out CartPole-v1 evaluation:.*?std=([0-9.]+)", bonus, "bonus SD"),
    "reinforce_no": f'{float(required(r"Last-100 mean without baseline:\s*(-?[0-9.]+)", reinforce, "reinforce no baseline")):.2f}',
    "reinforce_yes": f'{float(required(r"Last-100 mean with baseline:\s*(-?[0-9.]+)", reinforce, "reinforce baseline")):.2f}',
    "ac_final": required(r"Episode\s+[0-9]+:\s+mean reward \(last 100\) = ([0-9.]+)\s*\nCartPole-v1 solved", actor, "actor criterion"),
    "ac_episode": required(r"CartPole-v1 solved at episode ([0-9]+)", actor, "actor episode"),
}
values["reinforce_diff"] = f'{float(values["reinforce_yes"]) - float(values["reinforce_no"]):.2f}'

policy = required(r"Deterministic learned policy:\s*([LDRUHG?\s]+?)\s*Slippery learned policy:",q,"Q policy")
policy2 = required(r"Slippery learned policy:\s*([LDRUHG?\s]+?)\s*Greedy evaluation success",q,"Q slippery policy")
values["q_policy_det_inline"] = " / ".join(" ".join(row.split()) for row in policy.splitlines() if row.strip())
values["q_policy_slip_inline"] = " / ".join(" ".join(row.split()) for row in policy2.splitlines() if row.strip())

for label,key in [
    ("baseline","ppo_base"),
    ("higher_learning_rate","ppo_lr"),
    ("larger_batch","ppo_batch"),
]:
    pattern = rf"^{label}\s*\|\s*([0-9.]+)\s*\|\s*([0-9]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)"
    match = re.search(pattern, ppo, re.M)
    if not match:
        raise RuntimeError(f"Missing measured PPO comparison: {label}")
    values[key] = match.group(3)
    values[key+"_sd"] = match.group(4)

# Each asset is extracted from a known training notebook, not synthesized.
figure_specs = {
    "q_learning_curve": ("q","T3: Learning Curve Visualization"),
    "q_value_heatmap": ("q","T3: Q-values Heatmap"),
    "q_environment_comparison": ("q","T4: Comparing Deterministic vs Slippery"),
    "sarsa_qlearning_comparison": ("sarsa","T3: Comparison with Q-Learning"),
    "dqn_learning_curve": ("dqn","T4: Plotting Learning Curve"),
    "bonus_cartpole_curve": ("bonus","T3: Plotting Learning Curve"),
    "reinforce_smoothed_curve": ("reinforce","def moving_average(a"),
    "reinforce_raw_curve": ("reinforce","plt.plot(rewards_no_baseline, alpha"),
    "actor_critic_curve": ("actor","T3: Learning Curve Analysis"),
    "ppo_learning_curve": ("ppo","T3: Learning Curve Analysis"),
    "ppo_alternatives_curve": ("ppo","ppo_reward_histories"),
}
for figure_name, (notebook_key, snippet) in figure_specs.items():
    matches = [c for c in code_cells(notebook_key) if snippet in source(c)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected a unique source cell for {figure_name}, found {len(matches)}")
    outputs = matches[0].get("outputs", [])
    images = [o["data"]["image/png"] for o in outputs if "image/png" in o.get("data", {})]
    if not images:
        raise RuntimeError(f"Original plot is missing from executed notebook: {figure_name}")
    encoded = images[-1]
    encoded = "".join(encoded) if isinstance(encoded, list) else encoded
    filename = FIGURES / (figure_name + ".png")
    filename.write_bytes(base64.b64decode(encoded))
    with Image.open(filename) as im:
        im.verify()
    print("Extracted real notebook figure:", filename.name, flush=True)

def substitute(markdown_text):
    def replace(match):
        key = match.group(1)
        if key not in values:
            raise RuntimeError("Unknown data placeholder: " + key)
        return str(values[key])
    result = re.sub(r"\{\{([a-z_]+)\}\}", replace, markdown_text)
    if re.search(r"\{\{[^}]+\}\}", result):
        raise RuntimeError("Unresolved placeholder remains")
    return result

CSS_TEXT = """
@page {
  size: A4;
  margin: 20mm 19mm 19mm 19mm;
  @top-left { content: "ADVANCED STATISTICAL LEARNING · MODULE A"; font: 8pt 'DejaVu Sans'; color: #708096; }
  @top-right { content: "COURSE ASSIGNMENT REPORT"; font: 8pt 'DejaVu Sans'; color: #708096; }
  @bottom-right { content: "Page " counter(page) " / " counter(pages); font: 8pt 'DejaVu Sans'; color: #708096; }
}
html,body { margin:0; padding:0; }
body {
 font-family:'DejaVu Sans', sans-serif; color:#233247; line-height:1.58;
 font-size:9.5pt; hyphens:auto;
}
h1 { font-weight:700; color:#163a55; font-size:22pt; margin:0 0 9mm; line-height:1.25; }
h2 { color:#154e6d; border-bottom: 1.4pt solid #c6dce9; padding-bottom:3mm;
     font-size:14pt; margin:11mm 0 4mm; break-after:avoid; }
h3 { color:#324e68; font-size:11pt; margin:6mm 0 2.5mm; break-after:avoid; }
p { margin:0 0 3.5mm; text-align:justify; }
ul { padding-left:6mm; }
li { margin-bottom:2mm; }
strong { color:#122e49; }
table { border-collapse:collapse; width:100%; margin:4mm 0 6mm;
        font-size:8.1pt; table-layout:auto; break-inside:avoid; }
thead { display:table-header-group; }
th { background:#e8f0f7; color:#183c55; padding:2.3mm 2mm; border-bottom:1pt solid #c2d6e6; text-align:left; }
td { padding:2.3mm 2mm; border-bottom:.45pt solid #d9e4ec; vertical-align:top; }
tr:nth-child(even) td { background:#f7f9fc; }
code { font-family:'DejaVu Sans Mono',monospace; font-size:8.6pt; }
p:has(> img) { text-align:center; break-inside:avoid; margin:5mm auto 5mm; }
img { display:block; max-width:100%; max-height:73mm; width:auto; margin:0 auto;
      border:1px solid #d8e3ec; border-radius:4px; }
blockquote { border-left:3pt solid #7eaccc; margin-left:0; padding-left:4mm; color:#4a647b; }
.report-meta { border-bottom:2pt solid #1b5f83; padding-bottom:4mm; margin-bottom:6mm; }
.report-note { color:#5a7086; }
"""
RTL_CSS = """
@page {
  @top-left { content: "گزارش تکلیف — ماژول الف"; font-family:'Noto Sans Arabic'; direction:rtl; }
  @top-right { content: "یادگیری آماری پیشرفته"; font-family:'Noto Sans Arabic'; direction:rtl; }
  @bottom-left { content: "صفحه " counter(page) " از " counter(pages); font-family:'Noto Sans Arabic'; direction:rtl; }
  @bottom-right { content:none; }
}
body { font-family:'Noto Naskh Arabic','Noto Sans Arabic',sans-serif; direction:rtl;
 text-align:right; font-size:11pt; line-height:1.88; }
h1,h2,h3,th { font-family:'Noto Sans Arabic',sans-serif; }
h1 { font-size:21pt; line-height:1.55; }
h2 { font-size:14pt; }
h3 { font-size:11pt; }
p { text-align:justify; }
ul { padding-right:6mm; padding-left:0; }
th { text-align:right; }
table { font-size:9pt; direction:rtl; }
code { direction:ltr; unicode-bidi:isolate; display:inline-block; text-align:left;
 font-family:'DejaVu Sans Mono',monospace; font-size:8.3pt; }
p:has(> img) { direction:ltr; }
"""
for assignment in (1,2):
    for language in ("EN","FA"):
        base = f"Assignment{assignment}_Report_{language}"
        srcfile = REPORTS / (base+".md")
        if not srcfile.exists():
            raise RuntimeError(f"Missing report text: {srcfile}")
        body_text = substitute(srcfile.read_text(encoding="utf-8"))
        # Markdown source uses escaped inline backticks in some notes; normalize.
        body_text = body_text.replace("\\x5c\\x60", "\x60")
        content_html = markdown(body_text,extensions=["tables","fenced_code"])
        title = f"Assignment {assignment} — Module A" if language=="EN" else f"گزارش فارسی تکلیف {assignment}"
        html_lang = "fa" if language=="FA" else "en"
        direction = "rtl" if language=="FA" else "ltr"
        result_html = (
            f'<!DOCTYPE html><html lang="{html_lang}" dir="{direction}">'
            f'<head><meta charset="utf-8"><title>{html.escape(title)}</title>'
            '</head><body><main class="report">'+content_html+'</main></body></html>'
        )
        html_path = REPORTS / (base+".html")
        pdf_path = REPORTS / (base+".pdf")
        html_path.write_text(result_html,encoding="utf-8")
        style = CSS(string=CSS_TEXT + (RTL_CSS if language=="FA" else ""))
        HTML(string=result_html,base_url=str(REPORTS),encoding="utf-8").write_pdf(pdf_path,stylesheets=[style])
        if pdf_path.stat().st_size < 15_000:
            raise RuntimeError(f"Unexpectedly small report PDF: {pdf_path}")
        print("Created",pdf_path.relative_to(ROOT),"bytes",pdf_path.stat().st_size, flush=True)

(REPORTS/"MEASUREMENTS.json").write_text(
    json.dumps(values,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("FINISHED: four bilingual academic PDF reports from genuine notebook outputs",flush=True)
