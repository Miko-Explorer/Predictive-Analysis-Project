# Disease Outbreak Risk Prediction System

A machine learning-based predictive analysis system designed to forecast disease outbreak risks by analyzing epidemiological data across Philippine regions. This project implements both baseline and advanced machine learning models to predict high-risk disease locations.

---
## 📋 Project Overview

This project predicts whether disease cases will increase in the next week for specific regions and diseases. It analyzes historical disease data including:
- **Region** - Geographic location (NCR, IV-A, IV-B, VI, etc.)
- **PHI** - Public Health Information provider
- **Barangay** - Local administrative division
- **Disease** - Type of disease (Measles, Chikungunya, Typhoid, etc.)
- **No. of Cases** - Number of reported cases
- **Morbidity Week (MW)** - Week of reporting

The system aims to support public health officials in resource allocation and outbreak prevention.

---
## 🎯 Key Features

### Target Prediction
- **Binary Classification**: HIGH RISK (1) or LOW RISK (0)
- **Prediction Variable**: Will next week have above-average cases?
- **Average Threshold**: 4.7 cases per location
- **Dataset Size**: 17,755 records analyzed

### Data Characteristics
- **High Risk Locations**: 5,519 (31%)
- **Low Risk Locations**: 12,236 (69%)
- **Imbalanced Dataset**: Requires balanced class weights in models

---
## 🤖 Machine Learning Models

### 1. **Logistic Regression Model** (Baseline)
A simple binary classification model that serves as the baseline for comparison.

**Features Used:**
- No. of Cases
- Morbidity Week (MW)

**Model Configuration:**
- `random_state=42` - Reproducible results
- `max_iter=1000` - Convergence iterations
- `class_weight='balanced'` - Handles class imbalance

**Why These Features?**
- **No. of Cases** - Numeric data allows sophisticated analysis (trends, rates, comparisons)
- **Morbidity Week** - Captures essential seasonal disease patterns and reporting cycles

**Parameters Explained:**
- **penalty**: Controls regularization (default: 'l2' - gentle enforcement)
- **C**: Flexibility knob (higher = more flexible)
- **solver**: Mathematical optimization method
- **fit_intercept**: Includes baseline in the model

**Performance Metrics Calculated:**
- Accuracy
- Precision
- Recall
- F1-Score

### 2. **Random Forest Classifier** (Advanced Model)
An ensemble learning method using multiple decision trees for improved accuracy.

**Features Used:**
- All 4,553 features (after one-hot encoding)
- Region, PHI, Barangay, Disease (categorical)
- No. of Cases, MW (numerical)

**One-Hot Encoding Applied To:**
- Region
- PHI (Public Health Information)
- Barangay
- Disease

**Why All Features for Random Forest?**
- Random Forest utilizes all available features because its ensemble decision-tree architecture can handle high dimensionality and identify complex patterns
- Categorical encoding creates sparse feature space which Random Forest handles well

**Model Configuration:**
- `n_estimators=100` - Number of trees
- `random_state=42` - Reproducibility
- `max_depth=None` - Full tree growth
- `min_samples_split=2` - Node splitting threshold
- `min_samples_leaf=1` - Leaf node threshold

**Advantages:**
- Handles high-dimensional categorical data
- Captures complex non-linear relationships
- Provides feature importance scores
- Better performance on imbalanced datasets

---
## 📊 Model Evaluation Metrics

### Accuracy
- **Formula**: (Correct Predictions) / (Total Predictions)
- **Use Case**: When classes are balanced
- **Caveat**: Can be misleading with imbalanced data
- **Syntax**: `accuracy = accuracy_score(y_true, y_pred)`

### Precision
- **Formula**: TP / (TP + FP)
- **Meaning**: Of predicted HIGH risk, how many are actually correct?
- **Best For**: When false alarms are costly
- **Syntax**: `precision = precision_score(y_true, y_pred)`

### Recall (Sensitivity)
- **Formula**: TP / (TP + FN)
- **Meaning**: Of actual HIGH risk cases, how many did we catch?
- **Best For**: When missing positives is dangerous (outbreak prevention)
- **Syntax**: `recall = recall_score(y_true, y_pred)`

### F1-Score
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)
- **Use Case**: Balanced metric between precision and recall
- **Best For**: Imbalanced datasets
- **Syntax**: `f1 = f1_score(y_true, y_pred)`

---
## 📁 Project Structure

```
Predictive-Analysis-Project/
├── README.md                          # This file
├── Machine Learning Model.ipynb       # Main analysis notebook
├── Project Manuscript.pdf             # Detailed project report
├── Datasets/                          # Disease outbreak data
│   └── GROUP#3_2DSA2_FINAL_DATASET.csv
└── Application/                       # Application code
```

---
## 🔧 Technologies & Libraries

```python
# Data Processing
pandas                # Data manipulation
numpy                 # Numerical computing

# Visualization
matplotlib.pyplot     # Static plots
seaborn              # Statistical visualization

# Machine Learning
sklearn.model_selection  # Train-test split, stratification
sklearn.linear_model     # LogisticRegression
sklearn.ensemble         # RandomForestClassifier
sklearn.metrics          # Accuracy, Precision, Recall, F1

# Environment
Google Colab          # Cloud execution
```

---
## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Jupyter Notebook or Google Colab
- Required libraries (see below)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Miko-Explorer/Predictive-Analysis-Project.git
cd Predictive-Analysis-Project
```

2. **Install dependencies:**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

3. **Open the notebook:**
```bash
jupyter notebook "Machine Learning Model.ipynb"
```

Or use **Google Colab** for cloud-based execution (recommended for large datasets).

---
## 📖 How to Use

### Step 1: Load and Preview Data
```python
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
df = pd.read_csv("path/to/GROUP#3_2DSA2_FINAL_DATASET.csv")
print(df)  # 17,928 rows x 7 columns
```

### Step 2: Data Preparation
```python
import numpy as np

# Sort by Region, Disease, and Morbidity Week
df = df.sort_values(['Region', 'Disease', 'MW'])

# Create target variable: Will NEXT WEEK have high cases?
df['next_week_cases'] = df.groupby(['Region', 'Disease'])['No. of Cases'].shift(-1)

# Calculate average cases
average_cases = df['No. of Cases'].mean()  # 4.7

# Create binary target: 1 if above average, 0 otherwise
df['risk_prediction'] = np.where(df['next_week_cases'] > average_cases, 1, 0)

# Remove last week for each group
df = df.dropna(subset=['risk_prediction', 'next_week_cases'])

print(f"High risk locations: {sum(df['risk_prediction']==1)}")
print(f"Low risk locations: {sum(df['risk_prediction']==0)}")
```

### Step 3: Choose a Model

**Option A - Logistic Regression (Quick Baseline):**
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Select features
log_features = ['No. of Cases', 'MW']
X_log = df[log_features]
y_log = df['risk_prediction']

# Train-test split (70-30)
X_train, X_test, y_train, y_test = train_test_split(
    X_log, y_log, test_size=0.3, random_state=42, stratify=y_log
)

# Create and train model
model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
print(f"Precision: {precision_score(y_test, predictions):.2%}")
print(f"Recall: {recall_score(y_test, predictions):.2%}")
print(f"F1-Score: {f1_score(y_test, predictions):.2%}")
```

**Option B - Random Forest (Better Accuracy):**
```python
from sklearn.ensemble import RandomForestClassifier

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=['Region', 'PHI', 'Barangay', 'Disease'], drop_first=True)

# Prepare features
X_rf = df_encoded.drop('risk_prediction', axis=1)
y_rf = df_encoded['risk_prediction']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_rf, y_rf, test_size=0.3, random_state=42, stratify=y_rf
)

# Create and train model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=None)
rf_model.fit(X_train, y_train)

# Evaluate
predictions = rf_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
print(f"Precision: {precision_score(y_test, predictions):.2%}")
print(f"Recall: {recall_score(y_test, predictions):.2%}")
print(f"F1-Score: {f1_score(y_test, predictions):.2%}")
```

---
## 📈 Expected Results

| Metric | Logistic Regression | Random Forest |
|--------|-------------------|-----------------|
| Accuracy | ~65-70% | ~75-85% |
| Precision | Varies | Generally Higher |
| Recall | Varies | Generally Higher |
| F1-Score | Moderate | Strong |

*Note: Actual results depend on data split and random state*

---
## 💡 Key Insights

1. **Class Imbalance**: Dataset is imbalanced (31% HIGH, 69% LOW)
   - Solution: Use `class_weight='balanced'` in Logistic Regression
   - Random Forest handles this better naturally

2. **Feature Engineering**: One-hot encoding converts categorical variables
   - Creates 4,553 features (high dimensionality)
   - Random Forest handles sparse features well

3. **Temporal Patterns**: Morbidity Week captures seasonal disease patterns
   - Essential for identifying outbreak cycles

4. **Geographic Variation**: Region and Barangay show location-specific risks

---
## 🔬 Model Justification

### Why Logistic Regression?
- ✅ Simple baseline for comparison
- ✅ Interpretable coefficients
- ✅ Fast training
- ✅ Requires minimal features
- ❌ Limited to linear relationships

### Why Random Forest?
- ✅ Handles non-linear patterns
- ✅ Works with high-dimensional categorical data
- ✅ Provides feature importance
- ✅ Better for imbalanced data
- ✅ No feature scaling required
- ❌ Less interpretable (black box)

---
## 📊 Dataset Information

**Source**: Philippine Disease Outbreak Data
**Time Period**: Multiple morbidity weeks
**Original Records**: 17,928
**Processed Records**: 17,755
**Diseases Tracked**: Measles, Chikungunya, Typhoid, etc.
**Geographic Scope**: All Philippine regions (NCR, I-XIII, CAR, BARMM, NIR, CARAGA)

**Preprocessing Steps**:
1. Sorted by Region, Disease, Morbidity Week
2. Created target variable (next week prediction)
3. Removed rows with missing future data
4. Applied one-hot encoding for categorical variables
5. Stratified train-test split (70-30)

---
## 🤝 Contributing

Contributions are welcome! To improve this project:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/improvement`)
3. **Commit** your changes (`git commit -am 'Add enhancement'`)
4. **Push** to the branch (`git push origin feature/improvement`)
5. **Submit** a Pull Request

### Suggested Improvements
- Test additional models (SVM, Gradient Boosting, Neural Networks)
- Implement hyperparameter tuning with GridSearchCV
- Add k-fold cross-validation analysis
- Create visualizations (confusion matrices, ROC curves)
- Deploy as web service (Flask, FastAPI)
- Add SHAP explainability analysis

---
## 📝 Model Parameters Reference

### Logistic Regression Parameters
| Parameter | Default | Recommended | Effect |
|-----------|---------|-------------|--------|
| `penalty` | 'l2' | 'l2' | 'l1': strict, 'l2': gentle, 'elasticnet': mixed |
| `C` | 1.0 | 1.0 | Lower = stricter, Higher = flexible |
| `solver` | 'lbfgs' | 'lbfgs' | Algorithm choice for optimization |
| `max_iter` | 100 | 1000 | Increase if convergence fails |
| `class_weight` | None | 'balanced' | 'balanced' for imbalanced data |
| `random_state` | None | 42 | For reproducibility |
| `fit_intercept` | True | True | Include baseline in model |

### Random Forest Parameters
| Parameter | Default | Recommended | Effect |
|-----------|---------|-------------|--------|
| `n_estimators` | 100 | 100-500 | More trees = better but slower |
| `max_depth` | None | None | Deeper = more overfitting risk |
| `min_samples_split` | 2 | 2-5 | Higher = less overfitting |
| `min_samples_leaf` | 1 | 1-2 | Higher = less overfitting |
| `random_state` | None | 42 | For reproducibility |

---
## 🧮 Data Distribution

**Regional Distribution:**
- NCR (National Capital Region)
- Regions I through XIII
- CAR (Cordillera Administrative Region)
- BARMM (Bangsamoro Autonomous Region)
- NIR (Northern Mindanao region)
- CARAGA (Caraga region)

**Disease Types:**
- Measles
- Chikungunya
- Typhoid
- Other reportable diseases

**Temporal Coverage:**
- Morbidity Weeks 1-52 (full calendar year cycle)

---
## 📄 License

This project is open source and available under the MIT License.

---
## 👨‍💻 Author & Contributors
### Contributors
- Benedict Caliba - UST | BS Data Science and Analytics
- Thom Daniel Yutuc - UST | BS Data Science and Analytics
### Author
- Miko-Explorer 
- [GitHub Profile](https://github.com/Miko-Explorer)

---
## 📞 Support & Questions

For questions or issues:
1. Check the **Project Manuscript.pdf** for detailed documentation
2. Review comments in **Machine Learning Model.ipynb**
3. Open an issue on GitHub
4. Contact the project maintainer

---
## 🎓 Learning Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Logistic Regression Explained](https://en.wikipedia.org/wiki/Logistic_regression)
- [Random Forest Guide](https://en.wikipedia.org/wiki/Random_forest)
- [Machine Learning Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Class Imbalance Handling](https://imbalanced-learn.org/)
  
---
**Last Updated**: June 2026
**Status**: Active Development
**Language Composition**: Jupyter Notebook (98.8%), Python (1.2%)
**Dataset Size**: 17,755 records | **Model Comparison**: Baseline vs Advanced
