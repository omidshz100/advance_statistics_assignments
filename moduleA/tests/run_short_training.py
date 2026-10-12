"""Execute the actual notebook training cells with drastically reduced steps.

This is an integration smoke test ONLY; it does NOT validate policy quality.
It requires Gymnasium/Box2D, Torch, and Stable-Baselines3.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "Assignment1/01 LAB_A Q_Learning.ipynb": 9,
    "Assignment1/02 LAB_A DQN.ipynb": 9,
    "Assignment1/03 LAB_A SARSA.ipynb": 4,
    "Assignment1/04 LAB_A Bonus.ipynb": 3,
    "Assignment2/04 LAB_A Reinforce.ipynb": 6,
    "Assignment2/05 LAB_ A_Actor Critic.ipynb": 1,
    "Assignment2/06 LAB_A PPO.ipynb": 1,
}
REPLACEMENTS = {
    "total_episodes = 20000": "total_episodes = 5",
    "num_episodes = 600": "num_episodes = 3",
    "num_episodes = 500": "num_episodes = 3",
    "num_episodes = 2000": "num_episodes = 3",
    "num_episodes = 1000": "num_episodes = 3",
    "max_steps = 1000": "max_steps = 80",
    "max_steps = 500": "max_steps = 80",
    "BATCH_SIZE = 128": "BATCH_SIZE = 16",
    "n_steps=2048": "n_steps=64",
    "total_timesteps=25000": "total_timesteps=256",
}


def run_one(relative_path):
    import torch
    torch.set_num_threads(1)
    path = ROOT / relative_path
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    namespace = {"__name__": "__main__"}
    os.chdir(path.parent)
    for index, cell in enumerate(cells[:CASES[relative_path]]):
        source = cell["source"]
        code = "".join(source) if isinstance(source, list) else source
        lines = [line for line in code.splitlines()
                 if not line.lstrip().startswith(("!", "%"))]
        code = "\n".join(lines)
        for old, new in REPLACEMENTS.items():
            code = code.replace(old, new)
        if code.strip():
            exec(compile(code, f"{path.name}:code-{index}", "exec"), namespace)
    print(f"SMOKE OK: {relative_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", default=None)
    args = parser.parse_args()
    if args.single:
        run_one(args.single)
        return
    for relative_path in CASES:
        subprocess.run([sys.executable, __file__, "--single", relative_path],
                       check=True, timeout=240)


if __name__ == "__main__":
    main()
