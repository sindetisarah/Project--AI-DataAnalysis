# 🧠 AI Data Analysis – Advertising Click Prediction

This project utilizes a synthetic advertising dataset to predict whether a user will click on an advertisement, based on behavioral and demographic features.

---

## 📊 Data Description

The dataset includes the following features:

- **Daily Time Spent on Site**: Time spent on the site (in minutes)  
- **Age**: Age of the consumer  
- **Area Income**: Average income in the consumer’s area  
- **Daily Internet Usage**: Daily average internet usage (in minutes)  
- **Ad Topic Line**: Title of the advertisement  
- **City**: Consumer’s city  
- **Male**: Whether the user is male (1 = Male, 0 = Female)  
- **Country**: Country of the consumer  
- **Timestamp**: Time the ad was clicked or window was closed  
- **Clicked on Ad**: Target variable (1 = clicked, 0 = not clicked)

---

## 📁 Project Structure

### 1. Data Loading & Exploration
- Reads `advertising.csv` into a DataFrame
- Basic exploration with `.head()`, `.info()`, and `.describe()`
- Visualizes patterns with histograms, joint plots, and pair plots
- Analyzes feature correlations

### 2. Feature Engineering
- Converts `Timestamp` to datetime
- Extracts `Hour` and `DayofWeek` from the timestamp
- Creates new interaction features:  
  - `TimeSpent_Age`  
  - `Income_InternetUsage`
- Groups numerical features into categories:  
  - `AgeGroup`, `IncomeGroup`
- One-hot encodes categorical features
- Drops original `Timestamp` column

### 3. Model Training
- Splits dataset using `train_test_split`
- Applies feature scaling (MinMaxScaler, StandardScaler)
- Trains:
  - Logistic Regression
  - Random Forest Classifier
- Evaluates performance using:
  - Accuracy score
  - Classification report

### 4. Model Deployment
- Saves models using `joblib`:  
  - `logistic_regression_model.joblib`  
  - `random_forest_model.joblib`
- Also saves the column transformer:  
  - `column_transformer.joblib`

### 5. Model Testing
- Loads saved models and transformer
- Preprocesses new sample input
- Predicts ad click outcome

---

## 🚀 Usage

1. Clone this repo and open the notebook
2. Run all cells sequentially
3. Trained models and transformers are saved using `joblib`
4. Load them via:

```python
from joblib import load

model = load("random_forest_model.joblib")
transformer = load("column_transformer.joblib")
```

5. Pass in new data, preprocess, and predict.

---

## 📦 Dependencies

- `pandas`  
- `numpy`  
- `seaborn`  
- `matplotlib`  
- `scikit-learn`  
- `joblib`  

### 📥 To install:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn joblib
```
