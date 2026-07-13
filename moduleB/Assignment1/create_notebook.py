import json
import os

cells = []

def add_markdown(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split('\n')]
    })

def add_code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split('\n')]
    })

add_markdown("""# Assignment 1: Generalized linear models
## ADVANCED STATISTICAL LEARNING AND MODELING - MODULE B
### Name: [Your Name Here]""")

add_code("""import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel
import statsmodels.discrete.discrete_model as discrete
from scipy import stats
import warnings
warnings.filterwarnings('ignore')""")

add_markdown("""---
## Exercise 1: Modeling Student Performance with Categorical Logistic Models
### 1. Data Preparation and Categorical Target Definition""")

add_code("""# Load dataset
df_student = pd.read_csv('student-mat.csv', sep=';')
if 'G3' not in df_student.columns:
    df_student = pd.read_csv('student-mat.csv', sep=',')
    
# Display first few rows
display(df_student.head())

# Categorize G3: Low (0-9), Medium (10-14), High (15-20)
# Justification: 0-9 is failing/poor, 10-14 is acceptable/average, 15-20 is good/excellent.
def categorize_grade(g):
    if g < 10:
        return 'Low'
    elif g < 15:
        return 'Medium'
    else:
        return 'High'

df_student['G3_cat'] = df_student['G3'].apply(categorize_grade)

# Convert to ordered categorical type
cat_type = pd.CategoricalDtype(categories=['Low', 'Medium', 'High'], ordered=True)
df_student['G3_cat'] = df_student['G3_cat'].astype(cat_type)

# Select predictors and encode them
# 'studytime', 'failures', 'absences' are numeric. We also use 'sex' and 'address' as categorical.
predictors = ['studytime', 'failures', 'absences']
X1 = df_student[predictors].copy()
X1['sex_M'] = (df_student['sex'] == 'M').astype(int)
X1['address_U'] = (df_student['address'] == 'U').astype(int)

X1 = sm.add_constant(X1)

y_multinomial = df_student['G3_cat']
y_ordinal = df_student['G3_cat'].cat.codes""")

add_markdown("""### 2. Modeling: Multinomial vs. Ordinal""")

add_code("""# Multinomial Logistic Regression
mn_model = sm.MNLogit(y_ordinal, X1)
mn_res = mn_model.fit(disp=False)
print("Multinomial Logistic Regression Results:")
print(mn_res.summary())

# Ordinal Logistic Regression (Proportional Odds Model)
ord_model = OrderedModel(y_ordinal, X1.drop(columns='const'), distr='logit')
ord_res = ord_model.fit(method='bfgs', disp=False)
print("\\nOrdinal Logistic Regression Results:")
print(ord_res.summary())""")

add_markdown("""### 3. Model Comparison and Interpretation""")

add_code("""# Likelihood Ratio Test for Proportional Odds Assumption
llf_ord = ord_res.llf
llf_mn = mn_res.llf
df_diff = mn_res.df_model - ord_res.df_model

lr_stat = -2 * (llf_ord - llf_mn)
p_value = stats.chi2.sf(lr_stat, df_diff)

print(f"Likelihood Ratio Test for Proportional Odds:")
print(f"LR Statistic: {lr_stat:.2f}")
print(f"Degrees of Freedom: {df_diff}")
print(f"P-value: {p_value:.4f}")

# Model Choice based on AIC/BIC
print("\\nModel Comparison:")
print(f"Multinomial Model - AIC: {mn_res.aic:.2f}, BIC: {mn_res.bic:.2f}")
print(f"Ordinal Model - AIC: {ord_res.aic:.2f}, BIC: {ord_res.bic:.2f}")""")

add_markdown("""---
## Exercise 2: Log-Linear Models and Equivalence with Logistic Regression
### 1. Data Preparation""")

add_code("""# Load Adult dataset
adult_cols = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 
              'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 
              'hours_per_week', 'native_country', 'income']
df_adult = pd.read_csv('adult.csv', names=adult_cols, skipinitialspace=True)

# Recode Education
def recode_edu(e):
    if e in ['Bachelors', 'Masters', 'Prof-school', 'Doctorate']:
        return 'Advanced'
    elif e in ['Some-college', 'Assoc-voc', 'Assoc-acdm']:
        return 'College'
    else:
        return 'HighSchool'

df_adult['Edu_cat'] = df_adult['education'].apply(recode_edu)

# Keep Sex, Race, Education
df_cat = df_adult[['sex', 'race', 'Edu_cat']].copy()

# Contingency Table
c_table = pd.crosstab(index=[df_cat['sex'], df_cat['race']], columns=df_cat['Edu_cat'])
display(c_table)

# Prepare dataframe for Poisson Regression (Log-Linear)
df_agg = df_cat.groupby(['sex', 'race', 'Edu_cat']).size().reset_index(name='count')
display(df_agg.head())""")

add_markdown("""### 2. Log-Linear Modeling""")

add_code("""# Fit nested log-linear models
# 1. Mutually independent model
ll_indep = smf.glm('count ~ sex + race + Edu_cat', data=df_agg, family=sm.families.Poisson()).fit(disp=False)

# 2. Conditional independence model (e.g., Sex and Race are independent given Education)
ll_cond = smf.glm('count ~ sex*Edu_cat + race*Edu_cat', data=df_agg, family=sm.families.Poisson()).fit(disp=False)

# 3. All two-way associations model
ll_twoway = smf.glm('count ~ sex*race + sex*Edu_cat + race*Edu_cat', data=df_agg, family=sm.families.Poisson()).fit(disp=False)

print(f"Independent Model Deviance: {ll_indep.deviance:.2f}")
print(f"Conditional Model Deviance: {ll_cond.deviance:.2f}")
print(f"Two-Way Associations Model Deviance: {ll_twoway.deviance:.2f}")

print("\\nTwo-Way Associations Model Summary:")
print(ll_twoway.summary())""")

add_markdown("""### 3. Equivalence Demonstration""")

add_code("""# Binary Logistic Regression: predicting Sex as a function of Race and Education
df_adult['sex_bin'] = (df_adult['sex'] == 'Male').astype(int)

# Logistic regression using formula
log_reg = smf.logit('sex_bin ~ race + Edu_cat', data=df_adult).fit(disp=False)

print("Binary Logistic Regression Summary:")
print(log_reg.summary())

print("\\nComparison of Coefficients:")
print("--- Logistic Regression (Sex ~ Race + Edu) ---")
print(log_reg.params)
print("\\n--- Log-Linear Two-Way (Parameters interacting with Sex[Male]) ---")
sex_params = ll_twoway.params.filter(like='sex[T.Male]')
print(sex_params)""")

add_markdown("""---
## Exercise 3: Count Data Modeling with Overdispersion and Zero-Inflation
### 1. Data Exploration and Diagnostics""")

add_code("""# Load Absenteeism dataset
df_abs = pd.read_csv('Absenteeism_at_work.csv', sep=';')
if 'Absenteeism time in hours' not in df_abs.columns:
    df_abs = pd.read_csv('Absenteeism_at_work.csv', sep=',')

y_abs = df_abs['Absenteeism time in hours']

# Calculate mean and variance
mean_y = y_abs.mean()
var_y = y_abs.var()
zero_counts = (y_abs == 0).sum()
total_counts = len(y_abs)

print(f"Mean of Absenteeism: {mean_y:.2f}")
print(f"Variance of Absenteeism: {var_y:.2f}")
print(f"Variance/Mean Ratio: {var_y/mean_y:.2f} (indicates overdispersion if >> 1)")
print(f"Number of zeros: {zero_counts} out of {total_counts} ({(zero_counts/total_counts)*100:.1f}%)")

# Select predictors
predictors_abs = ['Distance from Residence to Work', 'Age', 'Service time', 'Education']
X_abs = df_abs[predictors_abs]
X_abs = sm.add_constant(X_abs)""")

add_markdown("""### 2. Model Fitting and Selection""")

add_code("""# Poisson Regression
poisson_model = sm.GLM(y_abs, X_abs, family=sm.families.Poisson()).fit()
print(f"Poisson AIC: {poisson_model.aic:.2f}")

# Negative Binomial Regression
nb_model = sm.GLM(y_abs, X_abs, family=sm.families.NegativeBinomial()).fit()
print(f"Negative Binomial AIC: {nb_model.aic:.2f}")

# Zero-Inflated Poisson (ZIP)
try:
    zip_model = discrete.ZeroInflatedPoisson(y_abs, X_abs, exog_infl=X_abs, inflation='logit').fit(maxiter=100, disp=False)
    print(f"ZIP AIC: {zip_model.aic:.2f}")
except Exception as e:
    print(f"ZIP fitting failed: {e}")

# Zero-Inflated Negative Binomial (ZINB)
try:
    zinb_model = discrete.ZeroInflatedNegativeBinomialP(y_abs, X_abs, exog_infl=X_abs, inflation='logit').fit(maxiter=100, disp=False)
    print(f"ZINB AIC: {zinb_model.aic:.2f}")
except Exception as e:
    print(f"ZINB fitting failed: {e}")""")

add_markdown("""### 3. Model Diagnostics and Interpretation""")

add_code("""# Detailed summary of the best model (e.g., ZINB if it has lowest AIC)
try:
    print("Best Model Summary (ZINB):")
    print(zinb_model.summary())
except:
    pass""")

notebook = {
 "cells": cells,
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('assignment1.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
