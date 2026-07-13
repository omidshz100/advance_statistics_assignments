import json
import os
import numpy as np

notebook_path = "/Users/omidshojaeianzanjani/Documents/uni/AdvanceStatisticsLearning/Module A/projects/Assignment2/04 LAB_A Reinforce.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb['cells']

def find_cell(cells, substring):
    for i, cell in enumerate(cells):
        if 'source' in cell:
            src = ''.join(cell['source'])
            if substring in src:
                return i
    return -1

# T1. Policy Network
policy_idx = find_cell(cells, "# A neural network for policy")
if policy_idx != -1:
    policy_code = """# T1: Implementation of REINFORCE - Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
policy_net = PolicyNetwork(state_dim, action_dim).to(device)
"""
    cells[policy_idx]['source'] = [line + '\n' for line in policy_code.split('\n')[:-1]]

# T1. Hyperparameters
hyper_idx = find_cell(cells, "# Here, we will specify the hyperparameters")
if hyper_idx != -1:
    hyper_code = """# T1: Implementation of REINFORCE - Hyperparameters
learning_rate = 1e-3
num_episodes = 2000
gamma = 0.99
max_steps = 1000
hidden_dim = 128
"""
    cells[hyper_idx]['source'] = [line + '\n' for line in hyper_code.split('\n')[:-1]]

# T1 & T2. Training Loop with Variance Reduction
train_idx = find_cell(cells, "### Training")
if train_idx != -1:
    train_code_idx = train_idx + 1
    train_code = """# T1: Implementation of REINFORCE - Training Loop
# T2: Implement variance-reduction strategy (Baseline)

def select_action(state, policy):
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    probs = policy(state)
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action)

def train_reinforce(use_baseline=False):
    policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0] # Handle newer gym versions
            
        log_probs = []
        rewards = []
        
        for step in range(max_steps):
            action, log_prob = select_action(state, policy)
            res = env.step(action)
            if len(res) == 4:
                next_state, reward, done, _ = res
            else:
                next_state, reward, terminated, truncated, _ = res
                done = terminated or truncated
                
            log_probs.append(log_prob)
            rewards.append(reward)
            
            if done:
                break
            state = next_state
            
        # Calculate returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns).to(device)
        
        # T2: Variance Reduction Strategy (Standardization / Baseline)
        if use_baseline:
            # Using standardized returns as a baseline to reduce variance
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        
        ep_reward = sum(rewards)
        episode_rewards.append(ep_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1}\\tReward: {avg_reward:.2f}")
            if avg_reward >= 200:
                print(f"Environment solved in {episode+1} episodes!")
                break
                
    return episode_rewards

print("Training without baseline...")
rewards_no_baseline = train_reinforce(use_baseline=False)

print("\\nTraining with baseline...")
rewards_with_baseline = train_reinforce(use_baseline=True)
"""
    cells[train_code_idx]['source'] = [line + '\n' for line in train_code.split('\n')[:-1]]

    # T2 and T3 Markdown and Code cells
    markdown_t2_t3 = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## T2. Theoretical Questions and Coding Implementation\n",
            "\n",
            "**1. Why is REINFORCE considered a Monte Carlo method?**\n",
            "REINFORCE is considered a Monte Carlo method because it relies on sampling full trajectories (episodes) to estimate the expected return (the objective function gradient). It updates the policy weights based on the actual observed returns at the end of each episode rather than bootstrapping (updating based on estimated returns like in TD learning).\n",
            "\n",
            "**2. What is the main drawback that REINFORCE suffers from, and what is the role of a control variate (baseline) in this context?**\n",
            "The main drawback of REINFORCE is its high variance in gradient estimates. Since it relies on single full-trajectory sample returns, the updates can be noisy and lead to unstable learning. \n",
            "A control variate (baseline) is subtracted from the return to reduce this variance without adding bias to the gradient expectation. By subtracting a baseline (like the state-value or the mean return), the algorithm scales the policy gradient based on whether the action performed better or worse than average, leading to more stable and faster convergence.\n",
            "\n",
            "**3. Implement in your code at least one variance-reduction strategy and compare the learning curves with and without this modification. Briefly comment on the observed differences.**\n",
            "In the training loop above (`train_reinforce`), I implemented a variance-reduction strategy by standardizing the returns (`use_baseline=True`). We can compare the learning curves below."
        ]
    }

    code_t2_plot = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import matplotlib.pyplot as plt\n",
            "import numpy as np\n",
            "\n",
            "def moving_average(a, n=50):\n",
            "    ret = np.cumsum(a, dtype=float)\n",
            "    ret[n:] = ret[n:] - ret[:-n]\n",
            "    return ret[n - 1:] / n\n",
            "\n",
            "plt.figure(figsize=(10, 6))\n",
            "plt.plot(moving_average(rewards_no_baseline), label='Without Baseline')\n",
            "plt.plot(moving_average(rewards_with_baseline), label='With Baseline (Standardized Returns)')\n",
            "plt.xlabel('Episodes')\n",
            "plt.ylabel('Smoothed Reward')\n",
            "plt.title('T3: Learning Curve Analysis (REINFORCE)')\n",
            "plt.legend()\n",
            "plt.grid(True)\n",
            "plt.show()\n",
            "\n",
            "# Comment on differences: \n",
            "# The curve with the baseline (variance reduction) typically converges faster and exhibits \n",
            "# more stable learning (less fluctuation) compared to the standard REINFORCE algorithm."
        ]
    }

    markdown_t3 = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## T3. Learning Curve Analysis and Hyperparameter Summary\n",
            "\n",
            "**Hyperparameter Summary Table:**\n",
            "\n",
            "| Hyperparameter | Value |\n",
            "|---|---|\n",
            "| Learning Rate | `1e-3` |\n",
            "| Number of training episodes | `2000` |\n",
            "| Discount factor $\\gamma$ | `0.99` |\n",
            "| Maximum number of steps per episode | `1000` |\n",
            "| Batch size | `N/A` (updates at the end of each episode) |\n",
            "| Network architecture | Linear(8, 128) -> ReLU -> Linear(128, 4) -> Softmax |\n",
            "\n",
            "**Alternative Configurations Discussion:**\n",
            "- **Higher Learning Rate (e.g., `1e-2`)**: Causes the policy to update too aggressively, often resulting in instability, with the agent failing to converge or forgetting good policies quickly.\n",
            "- **Smaller Network Size (e.g., `hidden_dim = 32`)**: May struggle to learn complex representations required to balance the lander, leading to sub-optimal performance or slower convergence."
        ]
    }

    cells.insert(train_code_idx + 1, markdown_t2_t3)
    cells.insert(train_code_idx + 2, code_t2_plot)
    cells.insert(train_code_idx + 3, markdown_t3)

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Notebook updated successfully.")
