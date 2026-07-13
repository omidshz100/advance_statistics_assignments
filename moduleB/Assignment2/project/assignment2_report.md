# Assignment 2: Multilevel Modeling of Standardized Math Scores
**Advanced Statistical Learning and Modeling - Module B**

## 1. Data Preparation

### Preprocessing and Variable Selection
The analysis uses the dataset `INVALSI_data_MAT_2324.xlsx`, which contains standardized math test scores for fifth-grade (G05) students in Italy. The dataset includes 16,244 observations.

**Rationale for Variable Selection:**
- **Demographics:** `gender`, `month`, `year`, `origin`.
- **Early Education:** `nursery_attendance`, `kindergarten_attendance`.
- **Socio-Economic Background:** `father_education`, `mother_education`, `father_occupation`, `mother_occupation`.
- **Indices:** `ESCS_student` and `ESCS_class` representing student and class socio-economic/cultural status.
- **Geography:** `GeoArea_5` representing the geographic macro-area.
- **Response Variable:** `WLE_MATH_200`.

**Missing Data Handling:**
Categorical variables indicating "Missing" or "Unknown" (typically coded as `9` or `99`) and missing ESCS codes were mapped to `NaN`. Observations with missing values were excluded to maintain a complete case dataset for the mixed-effects regression. The resulting dataset after cleaning contains 9,623 valid student observations.

## 2. Response Variable and Variable Classification

**Response Variable Selection:**
`WLE_MATH_200` was chosen as the continuous response variable. It represents the standardized math score estimated via Item Response Theory (IRT). It is highly suitable for linear mixed-effects modeling (Gaussian response).

**Variable Classification:**
- **Individual-level variables (Level 1):** `gender`, `month`, `year`, `origin`, `nursery_attendance`, `kindergarten_attendance`, `father_education`, `mother_education`, `father_occupation`, `mother_occupation`, `ESCS_student`.
- **School/Class-level variables (Level 2/3):** `ESCS_class`.
- **Contextual variables:** `GeoArea_5`.

## 3. Modeling

### Model Hierarchy and Structure
Given the multi-stage stratified nature of the data, students are nested within class sections, which are nested within school campuses.
To account for this hierarchical structure, a **Mixed Linear Model** was specified:
- **Level 1:** Students.
- **Level 2:** Classes (Variance component `school_class`).
- **Level 3:** Schools (`SCHOOL_ANONYMIZED`).

The model structure employs random intercepts for schools and classes within schools. This properly accounts for intra-class correlation (ICC), meaning that students within the same class/school are expected to have more similar scores due to shared unobserved characteristics (e.g., teaching quality, school environment).

## 4. Results and Interpretation

### Interpretation of Fixed Effects
The model was fitted using Restricted Maximum Likelihood (REML). Key findings from the fixed effects include:
- **Gender:** `gender = 2` (females) scores significantly lower (-10.12 points) on average than males, controlling for other variables (p < 0.001).
- **Origin:** Immigrant backgrounds (`origin = 2` and `origin = 3`) are associated with lower math scores compared to native students.
- **Socio-Economic Status (ESCS):** The student-level `ESCS_student` is highly significant and positive (+3.46, p < 0.001), indicating that higher socio-economic status predicts better math performance. The class-level `ESCS_class` is positive but not statistically significant at the 5% level (p = 0.188).
- **Parental Education:** Higher levels of parental education (especially mother's education levels 4 through 11) have a strong, significant positive association with the student's math score, often increasing the score by 13-15 points compared to baseline.
- **Geography:** `GeoArea_5 = 5` (South and Islands) has a significant negative effect on math scores (-5.89, p = 0.005) compared to the North-West baseline.

### Interpretation of Random Effects
- **Class Variance Component:** The estimated variance for the class level (`class Var`) is substantial (191.11). This justifies the use of a multilevel model, as a significant portion of the unexplained variance in math scores is attributable to differences between classroom environments.

## 5. Discussion

The findings highlight significant disparities in educational outcomes in Italian primary schools. Variables such as socio-economic background, parental education, and geographic location play a critical role in shaping a student's math performance. Furthermore, the strong class-level variance suggests that unobserved factors at the classroom level (such as teacher effectiveness or peer dynamics) significantly influence student achievement. Early education indicators, like nursery attendance, also showed some impact, suggesting that early interventions could have lasting effects.
