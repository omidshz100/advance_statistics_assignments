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
This module focuses on Generalized Linear Models (GLMs) and advanced statistical techniques to analyze real-world datasets (e.g., student performance, census data, and absenteeism at work) using `pandas` and `statsmodels`.

The assignments cover:
- **Assignment 1:** Categorical Data Analysis & Log-Linear Models [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/omidshz100/advance_statistics_assignments/blob/main/moduleB/Assignment1/assignment1.ipynb)
  - Multinomial and Ordinal Logistic Regression (Proportional Odds Model).
  - Analyzing multi-way contingency tables and their equivalence to Logistic Regression.
- **Assignment 2:** Count Data Modeling [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/omidshz100/advance_statistics_assignments/blob/main/moduleB/Assignment2/project/assignment2.ipynb)
  - Poisson and Negative Binomial Regression.
  - Handling overdispersion and zero-inflation using Zero-Inflated Poisson (ZIP) and Zero-Inflated Negative Binomial (ZINB) models.

*(Located in `moduleB/Assignment1` and `moduleB/Assignment2`)*

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
