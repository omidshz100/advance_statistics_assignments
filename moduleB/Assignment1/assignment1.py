import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel
from scipy import stats

print("### EXERCISE 1 ###")
# --- Exercise 1 ---
# student-mat.csv uses semicolon delimiter usually, let's check
df_student = pd.read_csv('student-mat.csv', sep=';')
if 'G3' not in df_student.columns:
    df_student = pd.read_csv('student-mat.csv', sep=',')
    
print(df_student.head())

# Categorize G3: Low (0-9), Medium (10-14), High (15-20)
def categorize_grade(g):
    if g < 10:
        return 'Low'
    elif g < 15:
        return 'Medium'
    else:
        return 'High'

df_student['G3_cat'] = df_student['G3'].apply(categorize_grade)
# Order categorical for ordinal
df_student['G3_cat'] = pd.Categorical(df_student['G3_cat'], categories=['Low', 'Medium', 'High'], ordered=True)

# Select predictors
predictors = ['studytime', 'failures', 'absences', 'Medu', 'Fedu']
X = df_student[predictors]
X = sm.add_constant(X)
y_multinomial = df_student['G3_cat']
y_ordinal = df_student['G3_cat'].cat.codes

# Multinomial
mn_model = sm.MNLogit(y_ordinal, X)
mn_res = mn_model.fit(disp=False)
print("Multinomial Result:")
print(mn_res.summary())

# Ordinal
# OrderedModel requires endog to be encoded 0, 1, 2...
ord_model = OrderedModel(y_ordinal, df_student[predictors], distr='logit')
ord_res = ord_model.fit(method='bfgs', disp=False)
print("Ordinal Result:")
print(ord_res.summary())

# AIC/BIC
print("MNLogit AIC:", mn_res.aic)
print("Ordinal AIC:", ord_res.aic)

# Proportional Odds Test (Brant test equivalent or comparing deviances)
# For simplicity, we compare log-likelihoods of MNLogit vs Ordinal.
# Since Ordinal is nested within Multinomial (with parallel slopes constraint),
# we can do a Likelihood Ratio test.
llf_ord = ord_res.llf
llf_mn = mn_res.llf
df_diff = mn_res.df_model - ord_res.df_model
lr_stat = -2 * (llf_ord - llf_mn)
p_value = stats.chi2.sf(lr_stat, df_diff)
print(f"LR Test for Proportional Odds: stat={lr_stat:.2f}, df={df_diff}, p-value={p_value:.4f}")

print("\n### EXERCISE 2 ###")
# --- Exercise 2 ---
adult_cols = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 
              'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 
              'hours_per_week', 'native_country', 'income']
df_adult = pd.read_csv('adult.csv', names=adult_cols, skipinitialspace=True)

# Recode Education
# HighSchool (HS-grad, 11th, 9th, 7th-8th, 12th, 10th, 5th-6th, 1st-4th, Preschool)
# College (Some-college, Assoc-voc, Assoc-acdm)
# Advanced (Bachelors, Masters, Prof-school, Doctorate)

def recode_edu(e):
    if e in ['Bachelors', 'Masters', 'Prof-school', 'Doctorate']:
        return 'Advanced'
    elif e in ['Some-college', 'Assoc-voc', 'Assoc-acdm']:
        return 'College'
    else:
        return 'HighSchool'

df_adult['Edu_cat'] = df_adult['education'].apply(recode_edu)

# Keep Sex, Race, Education
df_cat = df_adult[['sex', 'race', 'Edu_cat']]

# Contingency Table
c_table = pd.crosstab(index=[df_cat['sex'], df_cat['race']], columns=df_cat['Edu_cat']).reset_index()
print("Contingency Table:")
print(c_table)

# Melt for Poisson Regression
df_agg = df_cat.groupby(['sex', 'race', 'Edu_cat']).size().reset_index(name='count')
print(df_agg.head())

# Fit nested log-linear models
# Independent
ll_indep = smf.glm('count ~ sex + race + Edu_cat', data=df_agg, family=sm.families.Poisson()).fit(disp=False)
# Conditional
ll_cond = smf.glm('count ~ sex*Edu_cat + race*Edu_cat', data=df_agg, family=sm.families.Poisson()).fit(disp=False)
# Two-way associations
ll_twoway = smf.glm('count ~ sex*race + sex*Edu_cat + race*Edu_cat', data=df_agg, family=sm.families.Poisson()).fit(disp=False)

print("Log-Linear Indep Deviance:", ll_indep.deviance)
print("Log-Linear Cond Deviance:", ll_cond.deviance)
print("Log-Linear TwoWay Deviance:", ll_twoway.deviance)

# Equivalence with Binary Logistic
# Logistic: predict sex ~ race + Edu_cat
df_agg['sex_bin'] = (df_agg['sex'] == 'Male').astype(int)
# For logistic, we need individual data or count-weighted data.
df_adult['sex_bin'] = (df_adult['sex'] == 'Male').astype(int)
log_reg = smf.logit('sex_bin ~ race + Edu_cat', data=df_adult).fit(disp=False)
print("Logistic Regression Result:")
print(log_reg.summary())
print("Two-Way Log-Linear Interaction Params related to Sex:")
print(ll_twoway.params.filter(like='sex'))


print("\n### EXERCISE 3 ###")
# --- Exercise 3 ---
df_abs = pd.read_csv('Absenteeism_at_work.csv', sep=';')
if 'Absenteeism time in hours' not in df_abs.columns:
    df_abs = pd.read_csv('Absenteeism_at_work.csv', sep=',')

y = df_abs['Absenteeism time in hours']
print("Mean Y:", y.mean(), "Var Y:", y.var())
print("Zero counts:", (y == 0).sum())

import statsmodels.discrete.discrete_model as discrete

predictors_abs = ['Distance from Residence to Work', 'Age', 'Service time', 'Education']
X_abs = df_abs[predictors_abs]
X_abs = sm.add_constant(X_abs)

# Poisson
poisson_model = discrete.Poisson(y, X_abs).fit(disp=False)
print("Poisson AIC:", poisson_model.aic)

# Negative Binomial
nb_model = discrete.NegativeBinomial(y, X_abs).fit(disp=False)
print("NB AIC:", nb_model.aic)

# ZIP / ZINB
# Zero-Inflated models need exog and exog_infl. We use the same for both.
zip_model = discrete.ZeroInflatedPoisson(y, X_abs, exog_infl=X_abs, inflation='logit').fit(maxiter=100, disp=False)
print("ZIP AIC:", zip_model.aic)

zinb_model = discrete.ZeroInflatedNegativeBinomialP(y, X_abs, exog_infl=X_abs, inflation='logit').fit(maxiter=100, disp=False)
print("ZINB AIC:", zinb_model.aic)

print("ZINB Result:")
print(zinb_model.summary())
