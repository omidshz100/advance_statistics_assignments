import json
import os

notebook_path = "/Users/omidshojaeianzanjani/Documents/uni/AdvanceStatisticsLearning/Module A/projects/Assignment2/04 LAB_A Reinforce.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'def show_video_of_model(env_name):' in source:
            new_source = []
            for line in source.split('\n'):
                if 'action, _ = select_action(state, trained_policy)' in line:
                    new_source.append("        with torch.no_grad():")
                    new_source.append("            state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)")
                    new_source.append("            probs = trained_policy(state_t)")
                    new_source.append("            action = torch.argmax(probs).item() # Deterministic action for evaluation")
                else:
                    new_source.append(line)
            cell['source'] = [line + '\n' for line in new_source[:-1]]
            break

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Notebook evaluation fixed.")
