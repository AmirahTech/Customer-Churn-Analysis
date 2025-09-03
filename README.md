# 📊 Customer Churn Prediction

This project implements a customer churn prediction model for a telecom company. The goal is to predict which customers are likely to leave the service using a Random Forest Classifier with SMOTE balancing and a full preprocessing pipeline. Predictions can be exported to CSV and visualized in Power BI dashboards.

---

---
## Preprocess.py
This script handles the initial data cleaning and feature preprocessing required for churn prediction. It performs the following key tasks:

- Load and Clean Data: Reads a CSV file into a Pandas DataFrame, converts the TotalCharges column to numeric (handling errors by coercing invalid values to NaN), drops rows with missing values, and removes the customerID column since it’s not useful for prediction.

- Build Preprocessor: Identifies numeric features (SeniorCitizen, tenure, MonthlyCharges, TotalCharges) and categorical features (all columns with object type). It then creates a ColumnTransformer that applies:

Standard Scaling to numeric columns to normalize values.

One-Hot Encoding to categorical columns, ensuring unseen categories are handled gracefully during prediction.

The build_preprocessor() function returns a transformer pipeline that is later used in model training and predictions to ensure consistent preprocessing across datasets. This ensures the machine learning model receives well-formatted and standardized data.

---
---
## Train.py

This script is responsible for training the customer churn prediction model. It combines preprocessing, class balancing, model training, and evaluation into one workflow. The main steps include:

- Data Loading and Cleaning: Reads the telecom churn dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv), converts all numeric columns (SeniorCitizen, tenure, MonthlyCharges, TotalCharges) to proper numeric types, and fills any missing numeric values with zero.

- Feature and Target Separation: Defines X as all customer attributes except Churn, and y as the target variable mapped to binary (Yes → 1, No → 0).

- Preprocessing Pipeline: Builds a ColumnTransformer that standardizes numeric features using StandardScaler and applies OneHotEncoder to categorical features, ensuring unknown categories are handled during predictions.

- Data Splitting: Splits data into training (80%) and testing (20%) sets using stratification to maintain the original churn distribution.

- Class Balancing with SMOTE: Applies Synthetic Minority Oversampling Technique (SMOTE) to the training set to balance the churn class, preventing the model from being biased toward non-churn customers.

- Model Training (Random Forest): Trains a tuned Random Forest Classifier with 300 trees, controlled depth and splitting rules, and class_weight='balanced' to further address any residual imbalance.

- Model Evaluation: Evaluates performance on the test set using Accuracy, Confusion Matrix, Precision/Recall/F1 classification report, and ROC-AUC score to assess both overall accuracy and churn detection capability.

- Pipeline Saving: Combines the preprocessor and model into a single Pipeline object and saves it as models/churn_model.pkl using Joblib, allowing easy re-use during prediction.

Running this file (python train.py) trains the model end-to-end, outputs evaluation metrics in the terminal, and saves the trained pipeline for later use in prediction scripts.
---
---
## Predict.py

This script is used to generate churn predictions for new customer data using the trained pipeline. It reads in customer records, applies the preprocessing and model automatically, and saves results as a CSV file with churn scores. The key steps are:

- Model Loading: Loads the pre-trained pipeline (models/churn_model.pkl). If the model is missing, the script will stop and prompt you to train it first.

- Data Input: Accepts a CSV file of customer information. The file must contain the same features used during training (excluding the target column Churn).

- Prediction with Threshold: Calculates the churn probability for each customer (churn_probability from 0 to 1) and assigns a binary churn label (churn_prediction of 1 for churn, 0 for no churn). A custom decision threshold can be set via command line (default is 0.4).

- Save Results: Adds the prediction columns to the original dataset and exports the full results to a specified CSV file (predictions_full.csv by default).

Quick Summary: After predictions, the script displays how many customers are predicted to churn and the percentage of total customers.
---
## Project Structure


---

## Dataset
- **Source:** Public Telco Customer Churn dataset (7,032 rows, 20 features)  
- **Target:** `Churn` (whether the customer left or stayed)  

---

##  Workflow (in simple terms)
1. **Prep:** cleaned data, fixed missing/invalid values, converted `TotalCharges` to numeric, one-hot encoded categoricals, scaled numeric features.  
2. **EDA:** found churn patterns — higher for **month-to-month**, **fiber optic**, **high charges**; lower for **long tenure** and **1–2 year contracts**; **Tech support / Security add-ons** reduce churn.  
3. **Models tried:**  
   - Logistic Regression  
   - Random Forest  
   - Gradient Boosting  
   - Tuned Random Forest (GridSearchCV)  
   - Random Forest + SMOTE (to handle imbalance)  
4. **Evaluation:** compared accuracy, precision, recall, F1 score.  

---

## Results (Test Set Performance)

| Model                | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|----------------------|---------:|----------------:|-------------:|---------:|
| Logistic Regression   | 0.787    | 0.73            | 0.70         | 0.71     |
| Random Forest         | 0.793    | 0.74            | 0.69         | 0.71     |
| Gradient Boosting     | 0.790    | 0.73            | 0.69         | 0.71     |
| **Tuned RF**          | **0.795**| 0.74            | 0.70         | 0.71     |
| RF + **SMOTE**        | 0.755    | 0.70            | **0.74**     | 0.71     |

- **Best accuracy → Tuned Random Forest (~0.80)**  
- **Best churn recall → RF + SMOTE (~0.71)** → catches more at-risk customers  

---

##  Key Insights
- Higher **TotalCharges** and **MonthlyCharges** → churn more likely  
- Short **tenure** → churn risk high  
- **Month-to-month** contracts & **Fiber optic** service → higher churn  
- **TechSupport** / **OnlineSecurity** → reduce churn risk  
- Encourage **1–2 year contracts** to stabilize customer base  

---

## Business Takeaways
- Focus retention efforts on **new** and **month-to-month** customers, especially on **fiber** and **high-charge** plans.  
- Upsell **support/security services** to reduce churn.  
- Incentivize customers to switch to **1–2 year contracts** for improved retention.  

---

## Reproduce
```bash
pip install -r requirements.txt
# Open the notebook and run cells
# If dataset not included, download Telco Customer Churn dataset and update the path
