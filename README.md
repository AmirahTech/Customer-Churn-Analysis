# 📊 Customer Analytics: Churn Prediction & Business Intelligence

This project implements a customer churn prediction model for a telecom company. The goal is to predict which customers are likely to leave the service using a Random Forest Classifier with SMOTE balancing and a full preprocessing pipeline. Predictions can be exported to CSV and visualized in Power BI dashboards.

---
🛠 Technologies Used

Python 3.9+

Pandas – data manipulation

Scikit-learn – preprocessing, modeling, evaluation

Imbalanced-learn (SMOTE) – class balancing

Joblib – save/load model pipeline

Power BI – dashboard visualization
---
---
## Dataset
- **Source:** Public Telco Customer Churn dataset (7,032 rows, 20 features)  
- **Target:** `Churn` (whether the customer left or stayed)
---
---
##  Workflow (in simple terms)
- Prep: cleaned telecom data, removed missing/invalid rows, converted TotalCharges to numeric, scaled numeric features, and one-hot encoded categorical columns.

- Model Training: split data into train/test, applied SMOTE to fix class imbalance, trained a tuned Random Forest model (300 trees, controlled depth, balanced weights).

- Evaluation: measured accuracy, precision, recall, F1 score, and ROC-AUC to ensure churners were detected effectively.

- Prediction: used saved pipeline to score new customers with churn probability and label (0 = stay, 1 = churn) at a chosen threshold.

- Visualization: loaded predictions into Power BI to build dashboards showing churn risk, revenue impact, and high-priority customers for retention.
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



---


---
### Key Insights
- **Overall accuracy ~75%** — the model correctly predicts churn in about three out of four cases.  
- **High precision for non-churners (0.89)** — the model is very reliable when predicting customers who will stay.  
- **Good recall for churners (0.74)** — the model identifies most customers who are likely to churn.  
- **Balanced performance (F1 for churn = 0.61)** — captures churners without too many false alarms.  
- **ROC-AUC ~0.82** — strong ability to distinguish between churn and non-churn customers.  
- **Class imbalance handled well with SMOTE** — improved detection of minority churn class compared to baseline.  
---

---

### Business Takeaways
- **Target high-risk customers early** — focus retention efforts on those flagged with high churn probability.  
- **Improve customer experience for vulnerable segments** — customers on month-to-month contracts or with high monthly charges are more likely to leave.  
- **Retention programs can reduce revenue loss** — prioritize outreach (discounts, loyalty perks, better support) for high-value accounts at risk.  
- **Upsell protective services** — features like tech support and security add-ons are linked to lower churn rates.  
- **Use dashboards for ongoing monitoring** — Power BI makes it easy to track churn risk and adjust strategy in real time.  


---
### Model Evaluation Results

```text
Accuracy: 0.7466

Confusion Matrix:
[[775 260]
 [ 97 277]]

Classification Report:
               precision    recall  f1-score   support
           0       0.89      0.75      0.81      1035
           1       0.52      0.74      0.61       374

    accuracy                           0.75      1409
   macro avg       0.70      0.74      0.71      1409
weighted avg       0.79      0.75      0.76      1409

---

