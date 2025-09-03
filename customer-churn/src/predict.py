import os
import argparse
import pandas as pd
import joblib
import sys

# Locate project root and models folder
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(project_root, "models", "churn_model.pkl")

def predict(input_csv, output_csv="predictions_full.csv", threshold=0.4):
    # Load model
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        sys.exit(f"❌ Model not found at {model_path}. Train it first.")

    # Load input data
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        sys.exit(f"❌ Failed to load input CSV: {e}")

    # Predict probabilities
    try:
        churn_prob = model.predict_proba(df)[:, 1]  # probability of churn (class 1)
    except Exception as e:
        sys.exit(f"❌ Prediction failed. Ensure CSV columns match training data.\nError: {e}")

    # Apply custom threshold to create churn prediction
    churn_label = (churn_prob >= threshold).astype(int)

    # Add predictions as new columns
    df["churn_prediction"] = churn_label
    df["churn_probability"] = churn_prob

    # Save full dataset with predictions
    df.to_csv(output_csv, index=False)
    print(f"✅ Predictions saved to {output_csv} with threshold {threshold}")

    # Quick summary
    churn_count = (churn_label == 1).sum()
    total_count = len(df)
    print(f"📊 Predicted churn customers: {churn_count}/{total_count} "
          f"({churn_count/total_count:.1%})")

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict customer churn")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to customer data CSV")
    parser.add_argument("--output_csv", type=str, default="predictions_full.csv", help="Path to save predictions CSV")
    parser.add_argument("--threshold", type=float, default=0.4, help="Churn decision threshold (default=0.4)")
    args = parser.parse_args()

    predict(args.input_csv, args.output_csv, args.threshold)
