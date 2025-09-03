import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    df = df.drop('customerID', axis=1)
    return df

def build_preprocessor(df):
    X = df.drop("Churn", axis=1)
    numeric_features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )
    return preprocessor
