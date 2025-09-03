
from predict import predict

# Input test CSV and output predictions CSV
input_csv = "data/test_customers.csv"
output_csv = "data/predictions_full.csv"

# Run prediction
df_predictions = predict(input_csv, output_csv)

# Optional: see the first few rows
print(df_predictions.head())
