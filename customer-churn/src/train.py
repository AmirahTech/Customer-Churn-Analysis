import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

def train_tuned_model():
    # 1. Load data
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # 2. Clean numeric columns
    for col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(0)

    # 3. Separate features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({'Yes':1, 'No':0})

    # 4. Define numeric and categorical features
    numeric_features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    # 5. Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    # 6. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 7. Preprocess features
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # 8. Apply SMOTE to training set
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_transformed, y_train)

    # 9. Tuned Random Forest
    model = RandomForestClassifier(
        n_estimators=300,      # more trees
        max_depth=12,          # deeper trees
        min_samples_split=4,   # tune splitting
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'  # handle any remaining imbalance
    )
    model.fit(X_train_bal, y_train_bal)

    # 10. Evaluate
    y_pred = model.predict(X_test_transformed)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test_transformed)[:,1]))

    # 11. Save pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(models_dir, "churn_model.pkl"))
    print("Model saved to models/churn_model.pkl")

    return pipeline

if __name__ == "__main__":
    train_tuned_model()
