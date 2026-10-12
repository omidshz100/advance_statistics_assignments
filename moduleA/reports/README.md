# Module A — Bilingual academic reports

Four PDFs generated from fully executed notebooks and genuine saved figures.

| Assignment | English (for lecturer review) | Persian (personal study) |
|---|---|---|
| Assignment 1 — Q-Learning, SARSA, DQN, Bonus | [PDF](Assignment1_Report_EN.pdf) / [HTML](Assignment1_Report_EN.html) | [PDF](Assignment1_Report_FA.pdf) / [HTML](Assignment1_Report_FA.html) |
| Assignment 2 — REINFORCE, Actor-Critic, PPO | [PDF](Assignment2_Report_EN.pdf) / [HTML](Assignment2_Report_EN.html) | [PDF](Assignment2_Report_FA.pdf) / [HTML](Assignment2_Report_FA.html) |

**Submission:** The official Module A Assignment 1 instructions require executed, annotated Jupyter notebooks. These reports are optional supporting documents and do not replace the notebooks.

**Integrity:** All eleven plots in [figures/](figures/) come directly from saved notebook outputs. Observed figures are parsed automatically; see [measurements](MEASUREMENTS.json) and the [builder script](build_reports.py).

**Limits:** The bonus CartPole DQN did not meet 475/100 in the recorded execution; the Actor-Critic figure of merit comes from a training window, not a separate evaluation; REINFORCE used LunarLander-v3 due to deprecation of v2. Results are from specific saved runs, not multi-seed statistical proofs.
