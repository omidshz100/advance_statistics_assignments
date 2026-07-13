# Advanced Statistical Learning and Modeling Assignments

Welcome to the repository for the **Advanced Statistical Learning and Modeling** course assignments. This project is divided into two main modules covering foundational and advanced concepts in Reinforcement Learning and Statistical Modeling.

## 🚀 Repository Structure

### 🧠 Module A: Reinforcement Learning (RL)
This module focuses on implementing classic and deep reinforcement learning algorithms from scratch using Python, PyTorch, and Gymnasium. The environments explored include **FrozenLake** and **LunarLander**.

The lab sessions and assignments cover:
- **Tabular Methods:**
  - `Q-Learning` (FrozenLake)
  - `SARSA` (FrozenLake)
- **Deep Reinforcement Learning:**
  - `Deep Q-Networks (DQN)` (LunarLander)
  - `REINFORCE` (Policy Gradient) (LunarLander)
  - `Actor-Critic` methods (LunarLander)
  - `Proximal Policy Optimization (PPO)` (LunarLander)

*(Located in `moduleA/Assignment1` and `moduleA/Assignment2`)*

### 📊 Module B: Advanced Statistical Learning
This module focuses on Generalized Linear Models (GLMs) and advanced statistical techniques to analyze real-world datasets (e.g., student performance, census data, and absenteeism at work) using `pandas` and `statsmodels`.

The assignments cover:
- **Categorical Data Analysis:** 
  - Multinomial and Ordinal Logistic Regression (Proportional Odds Model).
- **Log-Linear Models:** 
  - Analyzing multi-way contingency tables and their equivalence to Logistic Regression.
- **Count Data Modeling:** 
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
Navigate to the respective module directory and launch Jupyter Notebook or JupyterLab to explore the implementations:
```bash
jupyter notebook
```
Open any `.ipynb` file to run the cells, view the training progress, statistical model summaries, and output visualisations/videos.

## 👤 Author
**Omid Shojaeian Zanjani**

---
*Developed as part of the Advanced Statistical Learning and Modeling university coursework.*
