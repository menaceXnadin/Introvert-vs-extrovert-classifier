# Introvert vs Extrovert Classifier

This project is a machine learning pipeline for classifying individuals as introverts or extroverts based on survey data. The workflow includes data preprocessing, exploratory data analysis, feature engineering, model selection, and evaluation.

## Features
- Data cleaning and preprocessing
- Exploratory data analysis with visualizations
- Handling missing values and outliers
- Feature encoding and scaling
- Model training with multiple algorithms (Logistic Regression, SVM, Random Forest, Gradient Boosting, KNN, Decision Tree, XGBoost)
- Hyperparameter tuning with cross-validation
- Performance evaluation and comparison

## Dataset
The dataset (`personality_dataset.csv`) contains survey responses with features such as:
- Time spent alone
- Social event attendance
- Going outside
- Friends circle size
- Post frequency
- Stage fear
- Drained after socializing
- Personality (target: Introvert/Extrovert)

## Usage
1. **Clone the repository**
2. **Install dependencies** (see below)
3. **Open `introvertvsextrovert.ipynb` in Jupyter Notebook**
4. **Run all cells** to reproduce the analysis and results

## Requirements
- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- xgboost
- missingno

Install dependencies with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost missingno
```

## Results
The notebook compares several machine learning models and reports their cross-validation F1 scores. The best model and its parameters are displayed at the end.

## License
This project is for educational purposes.

