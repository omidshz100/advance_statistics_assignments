import json
import os

notebook_path = "/Users/omidshojaeianzanjani/Documents/uni/AdvanceStatisticsLearning/Module A/projects/Assignment2/04 LAB_A Reinforce.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb['cells']

for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # 1. Update train_reinforce to return the policy
        if 'def train_reinforce(use_baseline=False):' in source:
            source = source.replace('return episode_rewards', 'return episode_rewards, policy')
            source = source.replace('rewards_no_baseline = train_reinforce(use_baseline=False)', 'rewards_no_baseline, _ = train_reinforce(use_baseline=False)')
            source = source.replace('rewards_with_baseline = train_reinforce(use_baseline=True)', 'rewards_with_baseline, trained_policy = train_reinforce(use_baseline=True)')
            cell['source'] = [line + '\n' for line in source.split('\n')[:-1]]
            
        # 2. Update visualization cell
        if 'def show_video_of_model(env_name):' in source:
            # Handle Gym < 0.26 and >= 0.26 step return values in visualization
            if 'action, _ = ???' in source or 'action, _ = select_action' in source:
                new_source = []
                for line in source.split('\n'):
                    if 'action, _ = ' in line:
                        new_source.append('        action, _ = select_action(state, trained_policy) # T1/T3: Use trained policy')
                    elif 'next_state, reward, done, info = env.step(action)' in line or 'res = env.step(action)' in line:
                        new_source.append('        res = env.step(action)')
                        new_source.append('        if len(res) == 4:')
                        new_source.append('            next_state, reward, done, info = res')
                        new_source.append('        else:')
                        new_source.append('            next_state, reward, terminated, truncated, info = res')
                        new_source.append('            done = terminated or truncated')
                    elif 'state = env.reset()' in line:
                        new_source.append('    state = env.reset()')
                        new_source.append('    if isinstance(state, tuple): state = state[0]')
                    else:
                        new_source.append(line)
                cell['source'] = [line + '\n' for line in new_source[:-1]]

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Notebook visualization fixed.")
