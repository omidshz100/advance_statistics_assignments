# Module A — Reinforcement Learning (reviewed 12 October 2026)

This folder contains seven teaching notebooks, not a consolidated benchmark or a claim that all agents solve their environments. **The original assignment prompts are embedded in the notebooks.** A separate official rubric/instructions document for Module A was not present in the repository during this review.

## Assignment 1

| Notebook | Algorithm | Environment | Status of revision |
|---|---|---|---|
| 01 LAB_A Q_Learning.ipynb | Tabular Q-learning (off-policy) | FrozenLake-v1 | Terminal bootstrap target and video capture repaired |
| 02 LAB_A DQN.ipynb | DQN, replay memory, target network | LunarLander-v3 | Terminal replay masks fixed, MPS support, modern video |
| 03 LAB_A SARSA.ipynb | Tabular SARSA (on-policy) | FrozenLake-v1 | Terminal targets fixed in SARSA and Q-learning comparison |
| 04 LAB_A Bonus.ipynb | DQN transfer exercise | CartPole-v1 | Replay buffer target checks improved |

## Assignment 2

| Notebook | Algorithm | Environment | Status of revision |
|---|---|---|---|
| 04 LAB_A Reinforce.ipynb | Episodic REINFORCE, earlier-episodes running baseline | LunarLander-v3 | Gymnasium migration, valid causal baseline, modern video |
| 05 LAB_ A_Actor Critic.ipynb | One-step TD Actor-Critic | CartPole-v1 | Replaced Monte Carlo return-minus-value update with true one-step TD updates |
| 06 LAB_A PPO.ipynb | PPO, Stable-Baselines3 | CartPole-v1 | Correct Monitor timesteps axis, Gymnasium video, separate policy evaluation |

**Why the code changes matter:** A terminal transition should not bootstrap the future value; an all-terminal replay minibatch must not trigger `torch.cat([])`; a baseline computed from the same episode's action-dependent returns is not the same as an earlier-episode state-independent control variate; Monte Carlo advantage is not identical to one-step TD error; training rewards are not an independent policy evaluation.

## Installation and reproduction

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r moduleA/requirements.txt
python -m unittest discover -s moduleA/tests -p 'test_*.py'
jupyter lab
```

From Jupyter, open each notebook and select **Run All**. DQN, REINFORCE and Actor-Critic perform hundreds/thousands of training episodes; training can take substantial time. LunarLander-v3 requires Box2D and may need a working SWIG/toolchain depending on platform. Video capture additionally needs available ffmpeg/MoviePy dependencies.

**Validation boundary:** Changes were reviewed at source level, but full Gymnasium/Box2D training and policy evaluation were **not re-executed** in the assistant's current runtime; the necessary Gymnasium, Box2D and Stable-Baselines3 packages were not available there. Saved outputs from the *old implementations* are preserved in the Git history, but not presented as results for the revised algorithms. Syntax checks do not establish learned performance or runtime compatibility.

## Scientific interpretation and remaining work

- Train with fixed seeds (ideally multiple seeds), log learning curves and run greedy policy evaluations on fresh environment episodes.
- Report evaluation episode count, mean and standard deviation separately from training rewards, including negative outcomes.
- The modified LunarLander REINFORCE uses **LunarLander-v3** rather than legacy **LunarLander-v2**, so legacy scores are not directly comparable without rerunning.
- The modified Actor-Critic now uses one-step TD updates rather than its previous episodic Monte Carlo advantage. Training curves must be rerun.
- For PPO, report the fresh evaluation produced by the final cell rather than inferring policy performance solely from the Monitor log.
- Verify the seven notebooks against the official Module A grading instructions if they exist outside this repository.

This review intentionally does not alter the original video files or the Module B coursework.
