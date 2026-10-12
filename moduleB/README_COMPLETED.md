# Module B — completed assignments (12 October 2026)

Two completed assignments for *Advanced Statistical Learning and Modeling* (Giulia Vannucci). This directory contains the updated, executed Jupyter notebooks, English PDF submissions, English/Persian exam guides, and selected numerical outputs.

## Assignment 1 — Generalized linear models

- [Executed notebook](Assignment1/assignment1.ipynb)
- [English PDF report](Assignment1/assignment1_report.pdf)
- [Individual-work note](Assignment1/group_note.txt)
- Exercise 1: multinomial vs. proportional-odds logistic regression on final grades (395 pupils). Separate parallel-slopes diagnostic and AIC/BIC comparison.
- Exercise 2: three nested log-linear count models, likelihood-ratio tests, formal fit assessment, and logistic/log-linear coefficient equivalence (32,561 adults).
- Exercise 3: Poisson, NB2, ZIP, ZINB2 for absent hours (740 observations); converged NB2 model has lowest AIC among tested candidates.

The supplied datasets `student-mat.csv`, `adult.csv` and `Absenteeism_at_work.csv` remain at their original locations in `Assignment1/`.

## Assignment 2 — multilevel INVALSI mathematics scores

- [Executed notebook](Assignment2/project/assignment2.ipynb)
- [English PDF report](Assignment2/assignment2_report.pdf)
- [Individual-work note](Assignment2/project/group_note.txt)
- Response: `WLE_MATH_200` (standardized grade-5 Italian math score).
- Hierarchy: students within classes within schools, with separate school and class variance terms. Dataset: 16,244 initial observations, 9,623 complete cases, 400 schools, 757 nested classes.
- ML comparison of a full and a reduced fixed-effects specification; REML final estimates, residual diagnostics and ICCs.
- Original `INVALSI_data_MAT_2324.xlsx` and `MAT_05-2023-24_datadescription.xlsx` remain in `Assignment2/` rather than being duplicated in `project/data/`.

**A2 paths:** The updated notebook searches its data in `Assignment2/` and `Assignment2/data/` (or the repository root), so launching Jupyter from `moduleB/Assignment2/project/` or the repository root is supported. Its outputs are written to `results/` relative to the working directory; checked outputs are committed to `Assignment2/project/results/`.

## Exam revision — not submission reports

- [English study guide](Exam_Review/Exam_Review_English.pdf)
- [Persian interactive study guide](Exam_Review/Exam_Review_Persian.html)
- [Assignment 1 Persian report](Exam_Review/Assignment1_Report_FA.pdf) / [HTML](Exam_Review/Assignment1_Report_FA.html)
- [Assignment 2 Persian report](Exam_Review/Assignment2_Report_FA.pdf) / [HTML](Exam_Review/Assignment2_Report_FA.html)

The Persian PDFs/HTML and exam guides are for personal revision; **submit the English assignment reports** to Microsoft Teams unless the lecturer requests another format.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r moduleB/requirements.txt
cd moduleB/Assignment1
jupyter notebook assignment1.ipynb
# or open moduleB/Assignment2/project/assignment2.ipynb
```

Open a Jupyter notebook and use **Run All**. Model fitting (particularly the multilevel mixed-effects model and zero-inflated models) can take time. Results/ includes files recorded from the verified 12 Oct 2026 runs.

## Statistical cautions

- Assignment 1, Exercise 2: model with all two-way interactions is best of three tested but has a significant goodness-of-fit statistic; interpret accordingly.
- Assignment 1, Exercise 3: apparent extra zeros alone do not establish a zero-inflated data-generating process. Zero-inflated results should be checked for identification, convergence and model comparison.
- Assignment 2: complete-case selection removed about 40.8% of observations, with differential retention by geography; models did not incorporate sampling weights, and effects are associative rather than causal.

The original notebooks and scripts can be retrieved from Git history if needed. This update does not modify Module A or the source data files.
