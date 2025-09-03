import pandas as pd
import numpy as np
import os

# Number of test customers you want
n_samples = 120

# Random seed for reproducibility
np.random.seed(42)

# Create synthetic test customers (features only)
df_test = pd.DataFrame({
    "customerID": [f"CUST-{i:04d}" for i in range(1, n_samples+1)],
    "gender": np.random.choice(["Male", "Female"], size=n_samples),
    "SeniorCitizen": np.random.choice([0, 1], size=n_samples),
    "Partner": np.random.choice(["Yes", "No"], size=n_samples),
    "Dependents": np.random.choice(["Yes", "No"], size=n_samples),
    "PhoneService": np.random.choice(["Yes", "No"], size=n_samples),
    "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], size=n_samples),
    "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], size=n_samples),
    "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], size=n_samples),
    "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], size=n_samples),
    "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], size=n_samples),
    "TechSupport": np.random.choice(["Yes", "No", "No internet service"], size=n_samples),
    "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], size=n_samples),
    "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], size=n_samples),
    "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples),
    "PaperlessBilling": np.random.choice(["Yes", "No"], size=n_samples),
    "PaymentMethod": np.random.choice([
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ], size=n_samples),
    "tenure": np.random.randint(0, 72, size=n_samples),
    "MonthlyCharges": np.round(np.random.uniform(20, 120, size=n_samples), 2),
})

# TotalCharges = tenure * MonthlyCharges
df_test["TotalCharges"] = np.round(df_test["tenure"] * df_test["MonthlyCharges"], 2)

# Save CSV for predictions
os.makedirs("data", exist_ok=True)
df_test.to_csv("data/test_customers.csv", index=False)

print("✅ Test customers CSV created: data/test_customers.csv")
