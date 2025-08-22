# Customer-Churn-Analysis

End-to-end machine learning project predicting **customer churn** and extracting **actionable business insights**.

---

## Project Structure
- `Churn_Prediction.ipynb` — full workflow (EDA → modeling → tuning → evaluation)
- *(Optional)* `requirements.txt` — install dependencies quickly
- *(Optional)* `data/` — dataset if licensing allows (see “Data” section below)

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
