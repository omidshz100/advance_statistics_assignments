# Module A — official assignment reconciliation (12 October 2026)

Sources used for the checklist: **ASSIGNMENT_1(2).pdf** (October 2025, pp. 1–4) and **ASSIGNMENT_2(2).pdf** (November 2025, pp. 1–3), supplied by the student. This file documents rubric alignment, NOT a claim of final trained performance.

## Assignment 1

| Official requirement | Notebook/code evidence | Outstanding proof |
|---|---|---|
| Q-Learning T1: epsilon-greedy tabular Q-learning | `01 LAB_A Q_Learning.ipynb`: Q-table and epsilon-greedy training | Re-run full episode count |
| Q-Learning T2: Q update, TD error, epsilon decay and hyperparameter table | Explanatory Markdown and training code | Review final values |
| Q-Learning T3: reward curve and Q-value heatmap | Existing plotting cells | Re-execute cells and save output |
| Q-Learning T4: slippery/deterministic curves AND learned policies | Two training loops; **added** side-by-side greedy policy maps and independent success rates | Record the measured policy results |
| SARSA T1–T2: SARSA algorithm and on-policy theory | `03 LAB_A SARSA.ipynb`: policy action bootstrapping and theory | Re-run full training |
| SARSA T3: equal-hyperparameter comparison, convergence, stability and winner | Existing learning curves; **added** independent evaluation comparison | Interpret actual results and note stochastic variation |
| DQN T1–T3: network, replay, target, epsilon, Bellman loss, training mechanics | `02 LAB_A DQN.ipynb`: network and Markdown explanations | Re-run corrected code; existing LunarLander is permitted as a control task, while CartPole is an example in the rubric |
| DQN T4: rewards and moving average | Plotting cells | Save the new learning curve output |
| Bonus T1–T3: choose new environment, adapt algorithm, average rewards, justify solving | `04 LAB_A Bonus.ipynb`: CartPole transfer and reward curve; **added** 100-episode evaluation | Do not claim "solved" without actual evaluation >=475 mean |

## Assignment 2

| Official requirement | Notebook/code evidence | Outstanding proof |
|---|---|---|
| REINFORCE T1: policy, hyperparameters and training on LunarLander-v2 | `04 LAB_A Reinforce.ipynb`; prefers v2 when supported, explicitly records v3 fallback for modern Gymnasium | Fully train and document actual env version and results |
| REINFORCE T2: Monte Carlo, variance problem, implement baseline and compare | Markdown; running baseline from *earlier* episodes; two training runs | Save full learning curves and measured differences |
| REINFORCE T3: episode rewards and hyperparameter table | Existing smoothed plot and table; **added raw reward plot** | Run full training |
| Actor–Critic T1: policy, value, **one-step TD error**, actor update | `05 LAB_ A_Actor Critic.ipynb`: corrected online TD implementation | Re-run complete training |
| Actor–Critic T2: explain actor, critic, advantage and TD | Corrected Markdown tied to actual code | Review for exam |
| Actor–Critic T3: reward curve and hyperparameter table | Plot and table cells | Save new training curve |
| PPO T1: one environment; SB3 policy and hyperparameters | `06 LAB_A PPO.ipynb`: CartPole-v1, SB3, seed 42 | Re-run full training |
| PPO T2: on-policy, probability ratio, clipping, entropy | Markdown questions addressed | Review for exam |
| PPO T3: learning curves, table, **two actually tried alternatives** | **Added** two additional 25k-step training runs, 30-episode held-out evaluation and overlaid moving-average plots | Save the three observed series and report their actual differences |

## Submission status

**Assignment 1** explicitly requires completed **.ipynb notebooks with code, outputs and visualizations**, and clearly labeled tasks in Markdown (p. 4). Both assignments use annotated notebooks to answer the task questions; no stand-alone PDF report is specified in these two supplied official assignment PDFs.

The correction branch is a **draft PR** because the notebook code has been tested with deliberately short episode counts, but final full-length training outputs have not yet been independently verified. The separately configured GitHub Actions full-notebook execution runs are intended to create and save those outputs. Check their status before calling Module A complete or merging the PR.

Scientific caveats: training rewards are distinct from greedy held-out evaluation; a 500-reward CartPole episode does not prove an average of 475 across 100 episodes; modern Gymnasium may only provide LunarLander-v3, in which case recorded results are from v3, not v2.
