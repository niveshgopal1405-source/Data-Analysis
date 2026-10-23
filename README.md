# InternCircle Internship: Project 01 — Exploratory Data Analysis (EDA) on Titanic

**Track:** Data Science & AI  
**Level:** Beginner  
**Focus:** Master Pandas library, exploratory data cleaning, and grouping/aggregation analysis  
**Location of Project Files:** `C:\Users\user\Documents\internship`

---

## 🎯 Executive Summary of Findings

| Research Question | Factor Analyzed | Key Metric / Finding | Exact Figures |
| :--- | :--- | :--- | :--- |
| **Question 1** | **Gender (`sex`)** | Women had a massively higher survival rate than men due to the "women and children first" maritime evacuation protocol. | **74.20%** for females (233 / 314) vs **18.89%** for males (109 / 577) |
| **Question 2** | **Ticket Class (`pclass`)** | Higher socio-economic status dramatically increased survival chances; upper decks had faster lifeboat access. | **62.96%** (1st Class) vs **47.28%** (2nd Class) vs **24.24%** (3rd Class) |
| **Question 3** | **Age Band (`age_band`)** | Children experienced priority evacuation, while senior passengers had the lowest survival of any age demographic. | **57.97%** for Children (0–12 yrs) vs **22.73%** for Seniors (60+ yrs) |

---

## 🔍 The Three Research Questions & One-Sentence Answers

### 1. Gender Impact
* **Question:** *Did female passengers have a significantly higher survival rate than male passengers?*
* **Answer:** **Women survived at an overwhelming rate of 74.20% (233 out of 314 survived) compared with only 18.89% for men (109 out of 577 survived), demonstrating that gender was the single strongest predictor of survival.**

### 2. Socio-Economic Status Impact
* **Question:** *Did ticket class (socio-economic status) significantly impact a passenger's chance of survival?*
* **Answer:** **First-class passengers achieved a 62.96% survival rate (136 out of 216 survived), compared to 47.28% in second class (87 out of 184 survived), and just 24.24% in third class (119 out of 491 survived), proving that wealth and cabin proximity to the boat deck substantially improved survival odds.**

### 3. Age Demographic Impact
* **Question:** *Was age related to survival, specifically did young children have higher survival rates compared to other age bands?*
* **Answer:** **Children aged 12 and under recorded the highest age-group survival rate at 57.97% (40 out of 69 survived), whereas seniors over 60 had the lowest survival rate at 22.73% (5 out of 22 survived).**

---

## 🛠️ Step-by-Step Data Cleaning & Rationale

| Column | Missing Count / % | Cleaning Action Taken | Statistical Rationale |
| :--- | :--- | :--- | :--- |
| **`age`** | 177 (19.87%) | Imputed with **group median** by `pclass` and `sex` | Age distributions vary markedly across passenger class and gender (1st class passengers were older on average than 3rd class). Cohort-median imputation preserves intra-group variance without skewing overall distributions. |
| **`deck`** | 688 (77.22%) | **Dropped** column entirely | Over 77% of entries were null. Imputing more than three-quarters of a feature would introduce severe noise and subjective bias. |
| **`embarked` / `embark_town`** | 2 (0.22%) | Imputed with **mode** (`'S'` / `'Southampton'`) | With only 2 records missing (<0.25%), modal replacement is standard and reflects the port where over 72% of all passengers boarded. |

### Duplicate Rows Audit
- **Check:** `df.duplicated().sum()` detected **118** duplicate demographic profiles.
- **Decision:** As emphasized in the InternCircle guide ("*Do not drop rows blindly; explain every cleaning choice*"), the Seaborn Titanic dataset omits distinct identifiers (like `PassengerId`, `Name`, or `Ticket` numbers). Rows sharing identical age, class, sex, fare, and embarkation town represent **actual separate passengers traveling together** (e.g., steerage groups or couples on identical ticket prices), not redundant errors. Therefore, they are intentionally preserved.

---

## 📊 Grouping & Aggregation Analysis

### 1. Survival Rate by Gender
```text
        Total_Passengers  Survived_Count  Survival_Rate  Survival_Percentage
sex                                                                         
female               314             233       0.742038                74.20
male                 577             109       0.188908                18.89
```

### 2. Survival Rate by Ticket Class
```text
        Total_Passengers  Survived_Count  Survival_Rate  Survival_Percentage
pclass                                                                      
1                    216             136       0.629630                62.96
2                    184              87       0.472826                47.28
3                    491             119       0.242363                24.24
```

### 3. Survival Rate by Age Band (`pd.cut`)
```text
             Total_Passengers  Survived_Count  Survival_Rate  Survival_Percentage
age_band                                                                         
Child                      69              40       0.579710                57.97
Teen                       70              30       0.428571                42.86
Adult                     602             217       0.360465                36.05
Middle-aged               128              50       0.390625                39.06
Senior                     22               5       0.227273                22.73
```

### 4. Intersection Analysis: Ticket Class × Gender
```text
               Total_Passengers  Survived_Count  Survival_Rate  Survival_Percentage
pclass sex                                                                         
1      female                94              91       0.968085                96.81
       male                 122              45       0.368852                36.89
2      female                76              70       0.921053                92.11
       male                 108              17       0.157407                15.74
3      female               144              72       0.500000                50.00
       male                 347              47       0.135447                13.54
```
> **Key Insight:** Female passengers in First Class achieved a near-perfect survival rate of **96.81%** (91 of 94 survived), and Second Class females achieved **92.11%**. Conversely, male passengers in Third Class had the lowest survival rate of any cohort at **13.54%** (47 of 347 survived).

---

## 📈 Generated Visualizations (in `plots/`)

1. **`01_survival_by_gender.png`**: High-contrast bar plot illustrating the 74.2% vs 18.9% gender survival disparity.
2. **`02_survival_by_class.png`**: Bar plot showcasing the monotonic step-down from 1st Class (63.0%) to 3rd Class (24.2%).
3. **`03_survival_by_age_band.png`**: Multi-cohort age analysis highlighting Child protection (58.0%) vs Senior vulnerability (22.7%).
4. **`04_survival_by_class_and_gender.png`**: Grouped bar chart depicting the powerful interaction between socio-economic status and gender.
5. **`05_correlation_heatmap.png`**: Correlation matrix covering numeric attributes (`survived`, `pclass`, `age`, `sibsp`, `parch`, `fare`, `family_size`).

---

## ✅ Submission Checklist Verification

All criteria specified in the InternCircle Task Guide (Page 3) have been fully met:
- [x] **Three clear questions are written at the start**
- [x] **Missing values are handled and the reasons explained**
- [x] **Groupings compare survival by at least three factors (`sex`, `pclass`, `age_band`, `class × sex`)**
- [x] **Each question has a one-sentence answer with an exact number**
- [x] **The notebook runs from top to bottom without errors**
- [x] **Headings and notes make it easy to follow**

---

## 🚀 How to Run the Project Locally

### 1. Requirements Installation
```bash
pip install -r requirements.txt
```

### 2. Run the Full Python Script
```bash
python eda_titanic.py
```

### 3. Open the Jupyter Notebook or HTML Report
- Open `eda_titanic.ipynb` in VS Code / Jupyter Lab:
  ```bash
  jupyter notebook eda_titanic.ipynb
  ```
- Or open `eda_titanic.html` in any web browser to view the complete pre-computed outputs and charts offline!
