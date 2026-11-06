from pathlib import Path

import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
DATA_PATH = BASE_DIR / "data" / "processed" / "student-performance-processed.csv"

print("SAVED MODEL TESTING")

model = joblib.load(MODEL_PATH)

df = pd.read_csv(DATA_PATH)

student_data = df.drop(columns=["risk"]).iloc[[0]]

actual_risk = int(df.iloc[0]["risk"])

prediction = int(model.predict(student_data)[0])

print("\nPrediction Completed!")

if prediction == 1:
    print("Predicted Result: At Risk")
else:
    print("Predicted Result: Not At Risk")

print("Actual Result:", actual_risk)
print("\nMODEL TEST COMPLETED SUCCESSFULLY!")