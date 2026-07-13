import json
import os

notebook_path = "/Users/omidshojaeianzanjani/Documents/uni/AdvanceStatisticsLearning/Module A/projects/Assignment2/04 LAB_A Reinforce.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = cell['source']
        if isinstance(source, list):
            new_source = []
            for line in source:
                new_source.append(line)
                if 'import numpy as np' in line:
                    new_source.append('np.bool8 = np.bool_ # Fix for gym compatibility with newer numpy\n')
            cell['source'] = new_source
        break # Only do it for the first code cell

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Notebook patched.")
