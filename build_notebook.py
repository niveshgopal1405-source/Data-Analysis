"""
Build and execute the complete, professional Jupyter Notebook for Titanic EDA project.
This script creates 'eda_titanic.ipynb' and exports 'eda_titanic.html'.
"""

import os

# Redirect directories away from read-only user root
BASE_DIR = 'C:/Users/user/Documents/internship'
for var, path in [
    ('IPYTHONDIR', f'{BASE_DIR}/.ipython'),
    ('JUPYTER_CONFIG_DIR', f'{BASE_DIR}/.jupyter'),
    ('JUPYTER_DATA_DIR', f'{BASE_DIR}/.local/share/jupyter'),
    ('JUPYTER_RUNTIME_DIR', f'{BASE_DIR}/.local/share/jupyter/runtime'),
]:
    os.makedirs(path, exist_ok=True)
    os.environ[var] = path

import nbformat as nbf
from nbclient import NotebookClient


nb = nbf.v4.new_notebook()

cells = []

# Cell 1: Markdown Title & Introduction
cells.append(nbf.v4.new_markdown_cell("""# PROJECT 01 | DATA SCIENCE & AI
# Exploratory Data Analysis (EDA) on Titanic

**Learner / Intern:** Data Science & AI Intern  
**Organization:** InternCircle  
**Difficulty:** Beginner | **Estimated Duration:** ~6 hours  
**Expected Outcome:** Master Pandas library, exploratory data cleaning, and statistical grouping techniques.

---

## 1. Project Overview & Research Questions
Before touching the data, we write down three core questions to maintain a focused and hypothesis-driven analysis:

1. **Question 1 (Gender & Survival):** Did female passengers have a significantly higher survival rate than male passengers?
2. **Question 2 (Socio-Economic Class & Survival):** Did ticket class (`pclass`) substantially influence a passenger's probability of survival?
3. **Question 3 (Age Dynamics & Survival):** Was passenger age related to survival—specifically, did young children experience higher survival rates compared to middle-aged passengers and seniors?
"""))

# Cell 2: Code - Imports and Global Configuration
cells.append(nbf.v4.new_code_cell("""# Import essential data science and visualization libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure presentation styling
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

print(f"Pandas version: {pd.__version__}")
print(f"Seaborn version: {sns.__version__}")
"""))

# Cell 3: Markdown - Step 2: Load and Look
cells.append(nbf.v4.new_markdown_cell("""---
## 2. Load and Look (Data Inspection)
We load the classic Titanic dataset using Seaborn and inspect its fundamental structure:
- `df.shape`: Dimensionality (rows and columns)
- `df.head()`: First 5 records
- `df.info()`: Data types and non-null counts
- `df.describe()`: Summary statistics for numerical and categorical variables
- Data dictionary explaining what each column means
"""))

# Cell 4: Code - Load Dataset & Shape
cells.append(nbf.v4.new_code_cell("""# Load the Titanic dataset
df = sns.load_dataset('titanic')

print(f"Dataset Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
df.head(10)
"""))

# Cell 5: Code - DataFrame Info & Types
cells.append(nbf.v4.new_code_cell("""# Inspect data types and structural integrity
df.info()
"""))

# Cell 6: Code - Describe Numerical and Categorical
cells.append(nbf.v4.new_code_cell("""# Descriptive statistical summary for numeric attributes
df.describe().round(2)
"""))

# Cell 7: Code - Describe Categorical
cells.append(nbf.v4.new_code_cell("""# Descriptive statistical summary for categorical/object attributes
df.describe(include=['category', 'object'])
"""))

# Cell 8: Markdown - Column Data Dictionary
cells.append(nbf.v4.new_markdown_cell("""### Feature / Column Dictionary

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `survived` | int64 | Target variable: `0` = Did not survive, `1` = Survived |
| `pclass` | int64 | Ticket class: `1` = 1st (Upper), `2` = 2nd (Middle), `3` = 3rd (Lower/Steerage) |
| `sex` | object | Biological sex (`male`, `female`) |
| `age` | float64 | Passenger age in fractional/integer years |
| `sibsp` | int64 | Number of siblings or spouses traveling aboard |
| `parch` | int64 | Number of parents or children traveling aboard |
| `fare` | float64 | Passenger ticket fare in British Pounds (£) |
| `embarked` | object | Port of embarkation code (`C` = Cherbourg, `Q` = Queenstown, `S` = Southampton) |
| `class` | category | Categorical text representation of `pclass` (First, Second, Third) |
| `who` | object | High-level demographic breakdown (`man`, `woman`, `child`) |
| `adult_male` | bool | Boolean flag indicating whether passenger is an adult male |
| `deck` | category | Cabin deck level (`A` through `G`) |
| `embark_town` | object | Full city name of embarkation port |
| `alive` | object | Text survival status (`yes`, `no`) |
| `alone` | bool | Flag indicating whether the passenger traveled without family |
"""))

# Cell 9: Markdown - Step 3: Find Missing Values
cells.append(nbf.v4.new_markdown_cell("""---
## 3. Identify Missing Values (Data Gaps)
Real-world datasets frequently contain missing values. We identify all columns with null entries and calculate their respective missing percentages.
"""))

# Cell 10: Code - Missing Values Audit
cells.append(nbf.v4.new_code_cell("""# Quantify missing values
missing_counts = df.isna().sum()
missing_pct = (missing_counts / len(df) * 100).round(2)

missing_df = pd.DataFrame({
    'Missing Values': missing_counts,
    'Percentage (%)': missing_pct
}).sort_values(by='Missing Values', ascending=False)

missing_df[missing_df['Missing Values'] > 0]
"""))

# Cell 11: Markdown - Step 4: Fix Missing Values with Reason
cells.append(nbf.v4.new_markdown_cell("""---
## 4. Handle Missing Values with Documented Rationale

Standard data cleaning requires explicit justification for every imputation or drop choice:

1. **`age` (177 missing / 19.87%):**
   - *Rationale:* Age is correlated with passenger class and gender (e.g., first-class passengers were older on average than third-class passengers). Imputing the global median would flatten this nuance. We use **conditional cohort median imputation** grouped by `pclass` and `sex` (`df.groupby(['pclass', 'sex'])['age'].transform('median')`).
2. **`deck` (688 missing / 77.22%):**
   - *Rationale:* Over 77% of entries are null. Imputing more than three-quarters of a column would introduce substantial synthetic bias and noise. Therefore, dropping the `deck` column is the most statistically sound strategy.
3. **`embarked` and `embark_town` (2 missing / 0.22%):**
   - *Rationale:* Only two passenger rows are missing port data. Southampton (`S`) is by far the most frequent port (>72% of all passengers). Mode imputation preserves dataset volume without biasing the distribution.
"""))

# Cell 12: Code - Execute Missing Values Fixes
cells.append(nbf.v4.new_code_cell("""# 1. Fill missing age values with median of the passenger's class and sex cohort
cohort_medians = df.groupby(['pclass', 'sex'])['age'].transform('median')
df['age'] = df['age'].fillna(cohort_medians)

# 2. Drop the mostly empty 'deck' column
df = df.drop(columns=['deck'])

# 3. Impute categorical port with the modal value
mode_port = df['embarked'].mode()[0]
mode_town = df['embark_town'].mode()[0]
df['embarked'] = df['embarked'].fillna(mode_port)
df['embark_town'] = df['embark_town'].fillna(mode_town)

# Verify zero remaining missing values
print("Verification of missing values post-cleaning:")
print(df.isna().sum())
"""))

# Cell 13: Markdown - Step 5: Check Duplicates and Data Types
cells.append(nbf.v4.new_markdown_cell("""---
## 5. Duplicate Rows Analysis & Type Verification

### Duplicate Rows:
- Running `df.duplicated().sum()` detects duplicate rows in this dataset.
- **Critical Caution:** In the Seaborn Titanic dataset, unique personal identifiers (`PassengerId`, `Name`, `Ticket`) are excluded. Consequently, rows with identical class, age, fare, and embarkation town represent **distinct passengers sharing demographic profiles** (e.g., steerage groups or couples on identical fares), rather than clerical duplicate errors.
- Therefore, we document and inspect these rows, but retain them so as not to discard authentic historical passenger data.
"""))

# Cell 14: Code - Duplicates and Types
cells.append(nbf.v4.new_code_cell("""# Check duplicate rows
dup_count = df.duplicated().sum()
print(f"Total duplicate rows detected: {dup_count}")

# Inspect a small sample of duplicate profiles
print("\\nSample duplicate records:")
display(df[df.duplicated(keep=False)].head(6))
"""))

# Cell 15: Code - Type Adjustments & Feature Engineering
cells.append(nbf.v4.new_code_cell("""# Ensure optimal data types
df['survived'] = df['survived'].astype(int)
df['pclass'] = df['pclass'].astype(int)
df['sex'] = df['sex'].astype('category')
df['embarked'] = df['embarked'].astype('category')

# Feature Engineering: Age Bands (pd.cut) as specified in project guide
# Bins: [0, 12, 18, 40, 60, 100] -> Child, Teen, Adult, Middle-aged, Senior
df['age_band'] = pd.cut(
    df['age'],
    bins=[0, 12, 18, 40, 60, 100],
    labels=['Child', 'Teen', 'Adult', 'Middle-aged', 'Senior']
)

# Feature Engineering: Family Size
df['family_size'] = df['sibsp'] + df['parch'] + 1

df[['age', 'age_band', 'sibsp', 'parch', 'family_size']].head()
"""))

# Cell 16: Markdown - Step 6: Grouping and Aggregations
cells.append(nbf.v4.new_markdown_cell("""---
## 6. Grouping and Aggregation Analysis
We now apply `groupby()` and aggregation functions (`mean`, `count`, `sum`) across three core factors:
1. **Survival by Sex**
2. **Survival by Ticket Class (`pclass`)**
3. **Survival by Age Band (`age_band`)**
4. **Interaction Analysis: Class × Sex**
"""))

# Cell 17: Code - Groupby Sex
cells.append(nbf.v4.new_code_cell("""# 1. Survival Rate by Sex
sex_agg = df.groupby('sex', observed=True)['survived'].agg(
    Total_Passengers='count',
    Survivors='sum',
    Survival_Rate='mean'
)
sex_agg['Survival_%'] = (sex_agg['Survival_Rate'] * 100).round(2)
sex_agg
"""))

# Cell 18: Code - Plot Sex
cells.append(nbf.v4.new_code_cell("""# Visualize Survival Rate by Gender
fig, ax = plt.subplots(figsize=(6, 4.5))
bars = ax.bar(
    sex_agg.index.astype(str).str.capitalize(),
    sex_agg['Survival_%'],
    color=['#e7298a', '#1f78b4'],
    width=0.45,
    edgecolor='black',
    linewidth=0.6
)
ax.set_title('Survival Rate by Gender', fontsize=13, fontweight='bold', pad=12)
ax.set_ylabel('Survival Rate (%)', fontsize=11)
ax.set_ylim(0, 100)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()
"""))

# Cell 19: Code - Groupby Class
cells.append(nbf.v4.new_code_cell("""# 2. Survival Rate by Ticket Class
class_agg = df.groupby('pclass', observed=True)['survived'].agg(
    Total_Passengers='count',
    Survivors='sum',
    Survival_Rate='mean'
)
class_agg['Survival_%'] = (class_agg['Survival_Rate'] * 100).round(2)
class_agg.index = ['1st Class', '2nd Class', '3rd Class']
class_agg
"""))

# Cell 20: Code - Plot Class
cells.append(nbf.v4.new_code_cell("""# Visualize Survival Rate by Ticket Class
fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(
    class_agg.index,
    class_agg['Survival_%'],
    color=['#2ca02c', '#ff7f0e', '#d62728'],
    width=0.45,
    edgecolor='black',
    linewidth=0.6
)
ax.set_title('Survival Rate by Ticket Class', fontsize=13, fontweight='bold', pad=12)
ax.set_ylabel('Survival Rate (%)', fontsize=11)
ax.set_ylim(0, 100)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()
"""))

# Cell 21: Code - Groupby Age Band
cells.append(nbf.v4.new_code_cell("""# 3. Survival Rate by Age Band
age_agg = df.groupby('age_band', observed=True)['survived'].agg(
    Total_Passengers='count',
    Survivors='sum',
    Survival_Rate='mean'
)
age_agg['Survival_%'] = (age_agg['Survival_Rate'] * 100).round(2)
age_agg
"""))

# Cell 22: Code - Plot Age Band
cells.append(nbf.v4.new_code_cell("""# Visualize Survival Rate by Age Band
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(
    age_agg.index.astype(str),
    age_agg['Survival_%'],
    color=sns.color_palette('Blues_r', n_colors=len(age_agg)),
    width=0.5,
    edgecolor='black',
    linewidth=0.6
)
ax.set_title('Survival Rate Across Age Bands', fontsize=13, fontweight='bold', pad=12)
ax.set_ylabel('Survival Rate (%)', fontsize=11)
ax.set_xlabel('Age Category', fontsize=11)
ax.set_ylim(0, 80)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()
"""))

# Cell 23: Code - Cross Tabulation Class x Sex
cells.append(nbf.v4.new_code_cell("""# 4. Multi-Variable Interaction: Pclass × Sex
cross_agg = df.groupby(['pclass', 'sex'], observed=True)['survived'].agg(
    Total_Passengers='count',
    Survivors='sum',
    Survival_Rate='mean'
)
cross_agg['Survival_%'] = (cross_agg['Survival_Rate'] * 100).round(2)
cross_agg
"""))

# Cell 24: Code - Plot Class x Sex
cells.append(nbf.v4.new_code_cell("""# Grouped Bar Plot: Class x Sex Survival Interaction
pivot_table = df.pivot_table(index='pclass', columns='sex', values='survived', aggfunc='mean') * 100

fig, ax = plt.subplots(figsize=(8, 5))
pivot_table.plot(kind='bar', ax=ax, color=['#e7298a', '#1f78b4'], edgecolor='black', linewidth=0.6, width=0.6)
ax.set_title('Survival Rate by Class & Gender Intersection', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Ticket Class', fontsize=11)
ax.set_ylabel('Survival Rate (%)', fontsize=11)
ax.set_ylim(0, 115)
ax.set_xticklabels(['1st Class', '2nd Class', '3rd Class'], rotation=0)
ax.legend(title='Gender', labels=['Female', 'Male'])

for container in ax.containers:
    for bar in container:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()
"""))

# Cell 25: Code - Correlation Heatmap
cells.append(nbf.v4.new_code_cell("""# Correlation Matrix of Numeric Features
fig, ax = plt.subplots(figsize=(7, 5))
numeric_df = df[['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare', 'family_size']]
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, square=True, ax=ax)
ax.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.show()
"""))

# Cell 26: Markdown - Step 7: Answers to Questions
cells.append(nbf.v4.new_markdown_cell("""---
## 7. Answers to Research Questions

As required by the guide, each question is answered in **one clear sentence featuring exact figures**:

> **Answer 1 (Gender):**  
> Women survived at an overwhelming rate of **74.20%** (233 survivors out of 314 female passengers) compared to only **18.89%** for men (109 survivors out of 577 male passengers), reflecting strict adherence to the maritime "women and children first" evacuation protocol.

> **Answer 2 (Class):**  
> Ticket class strongly dictated survival: First-class passengers achieved a **62.96%** survival rate, compared to **47.28%** in second class, and just **24.24%** in third class, showing that higher socio-economic status and cabin location closer to the upper boat deck significantly enhanced escape opportunities.

> **Answer 3 (Age):**  
> Children aged 12 and under experienced the highest age-group survival rate at **57.97%** (40 survivors out of 69 children), while senior passengers aged over 60 had the lowest survival rate at **22.73%** (5 survivors out of 22 seniors).

---
## 8. Submission Checklist Verification

- [x] **Three clear questions are written at the start**
- [x] **Missing values are handled and the reasons explained** (Cohort median for `age`, dropped `deck` due to 77.2% missingness, mode for `embarked`)
- [x] **Groupings compare survival by at least three factors** (`sex`, `pclass`, `age_band`, and `class × sex`)
- [x] **Each question has a one-sentence answer with an exact number**
- [x] **The notebook runs from top to bottom without errors**
- [x] **Headings, documentation, and notes make the analysis easy to follow**
"""))

nb['cells'] = cells

notebook_path = 'C:/Users/user/Documents/internship/eda_titanic.ipynb'
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Jupyter Notebook structure written to {notebook_path}")

# Execute notebook to pre-render all outputs
print("Executing notebook top-to-bottom...")
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb_to_run = nbf.read(f, as_version=4)

client = NotebookClient(nb_to_run, timeout=600, kernel_name='python3')
client.execute()

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb_to_run, f)

print("Notebook successfully executed and saved with all outputs!")

# Export to HTML using nbconvert
html_path = 'C:/Users/user/Documents/internship/eda_titanic.html'
try:
    from nbconvert import HTMLExporter
    html_exporter = HTMLExporter()
    (body, resources) = html_exporter.from_notebook_node(nb_to_run)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(body)
    print(f"Notebook successfully exported to HTML at: {html_path}")
except Exception as e:
    print(f"HTML Export note: {e}")
