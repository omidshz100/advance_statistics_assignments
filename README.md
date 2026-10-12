# Advanced Statistical Learning and Modeling Assignments

Welcome to the repository for the **Advanced Statistical Learning and Modeling** course assignments. This project is divided into two main modules covering foundational and advanced concepts in Reinforcement Learning and Statistical Modeling.

## 🚀 Repository Structure

### 🧠 Module A: Reinforcement Learning (RL)
This module focuses on implementing classic and deep reinforcement learning algorithms from scratch using Python, PyTorch, and Gymnasium. The environments explored include **FrozenLake** and **LunarLander**.

The lab sessions and assignments cover:
- **Assignment 1:**
  - `Q-Learning` (FrozenLake) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/omidshz100/advance_statistics_assignments/blob/main/moduleA/Assignment1/01%20LAB_A%20Q_Learning.ipynb)
  - `Deep Q-Networks (DQN)` (LunarLander) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/omidshz100/advance_statistics_assignments/blob/main/moduleA/Assignment1/02%20LAB_A%20DQN.ipynb)
  - `SARSA` (FrozenLake) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/omidshz100/advance_statistics_assignments/blob/main/moduleA/Assignment1/03%20LAB_A%20SARSA.ipynb)
- **Assignment 2:**
  - `REINFORCE` (Policy Gradient) (LunarLander) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/omidshz100/advance_statistics_assignments/blob/main/moduleA/Assignment2/04%20LAB_A%20Reinforce.ipynb)
  - `Actor-Critic` methods (LunarLander) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/omidshz100/advance_statistics_assignments/blob/main/moduleA/Assignment2/05%20LAB_%20A_Actor%20Critic.ipynb)
  - `Proximal Policy Optimization (PPO)` (LunarLander) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/omidshz100/advance_statistics_assignments/blob/main/moduleA/Assignment2/06%20LAB_A%20PPO.ipynb)

*(Located in `moduleA/Assignment1` and `moduleA/Assignment2`)*

### 📊 Module B: Advanced Statistical Learning
Two completed university assignments under [Module B](moduleB/README_COMPLETED.md):
- **Assignment 1 — Generalized Linear Models:** three exercises covering (1) multinomial and ordinal logistic regression, (2) log-linear modeling and logistic equivalence, and (3) count regression using Poisson, Negative Binomial, ZIP, and ZINB. [Executed notebook](moduleB/Assignment1/assignment1.ipynb) · [English report](moduleB/Assignment1/assignment1_report.pdf)
- **Assignment 2 — Multilevel Educational Modeling:** mixed-effects regression of INVALSI grade-five mathematics scores with nested school and class effects. [Executed notebook](moduleB/Assignment2/project/assignment2.ipynb) · [English report](moduleB/Assignment2/assignment2_report.pdf)
- Persian report translations and bilingual exam revision guides are collected under [Exam_Review](moduleB/Exam_Review/). Use the English reports for official submission.

For Module B dependencies, run `pip install -r moduleB/requirements.txt`. The original datasets and the entire Module A tree are preserved.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/omidshz100/advance_statistics_assignments.git
   cd advance_statistics_assignments
   ```

2. **Create a virtual environment (Optional but recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: For RL visualization and PyTorch models, ensure you have the appropriate system packages (e.g., `swig`, `gymnasium[box2d]`) installed as outlined in the notebooks.*

## 💻 Usage

### Running Locally
Navigate to the respective module directory and launch Jupyter Notebook or JupyterLab to explore the implementations:
```bash
jupyter notebook
```
Open any `.ipynb` file to run the cells, view the training progress, statistical model summaries, and output visualisations/videos.

### Running in Google Colab
If you prefer to run the notebooks in Google Colab, you can click the **Open In Colab** badges provided in the Repository Structure section above.
For the Reinforcement Learning (Module A) notebooks, you may need to install additional dependencies in Colab. Simply add a new code cell at the top of the notebook and run the following command before executing the rest of the code:
```python
!pip install gymnasium[box2d] stable-baselines3 swig moviepy imageio-ffmpeg
```

## 👤 Author
**Omid Shojaeian Zanjani**

---
*Developed as part of the Advanced Statistical Learning and Modeling university coursework.*
