from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_DIR / "data" / "raw" / "student-mat.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "student-performance-processed.csv"

df = pd.read_csv(INPUT_PATH, sep=";")

print("Original Dataset Shape:", df.shape)

df = df.drop_duplicates()

if df.isnull().sum().sum() > 0:
    raise ValueError("Missing values found in dataset.")

df["risk"] = (df["G3"] < 10).astype(int)

df = df.drop(columns=["G1", "G2", "G3"])

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_PATH, index=False)

print("\nProcessed Dataset Shape:", df.shape)
print("\nTarget Distribution:")
print(df["risk"].value_counts())

print("\nMissing Values:", df.isnull().sum().sum())

print("\nSaved File:", OUTPUT_PATH)
print("\nDATA PREPARATION COMPLETED SUCCESSFULLY!")