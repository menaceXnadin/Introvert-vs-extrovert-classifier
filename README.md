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

## Making Predictions on New Data
After training the model, you can make predictions on new data using the following steps:

1. **Run the entire notebook** to train and save the best model as `best_model.pkl`
2. **Use the prediction functions** provided at the end of the notebook:

```python
# Load the saved model
import joblib
loaded_model = joblib.load("best_model.pkl")

# Example of making a prediction
new_sample = {
    "Time_spent_Alone": 4.0,
    "Social_event_attendance": 4.0,
    "Going_outside": 6.0,
    "Friends_circle_size": 13.0,
    "Post_frequency": 5.0,
    "Stage_fear": False,
    "Drained_after_socializing": False
}

# Use the predict_personality function
personality, confidence = predict_personality(new_sample)
print(f"Predicted personality: {personality}")
print(f"Confidence: {confidence:.2f}")
```

The `predict_personality()` function handles all the necessary preprocessing steps to ensure your new data is processed in the same way as the training data.

### Required Features
When making predictions, you must provide values for all these features:
- Time_spent_Alone (float): Hours spent alone per day
- Social_event_attendance (float): Number of social events attended per month
- Going_outside (float): Number of times going outside per week
- Friends_circle_size (float): Number of close friends
- Post_frequency (float): Number of social media posts per week
- Stage_fear (boolean): Whether the person has stage fear (True/False)
- Drained_after_socializing (boolean): Whether the person feels drained after socializing (True/False)

## License
This project is for educational purposes.
