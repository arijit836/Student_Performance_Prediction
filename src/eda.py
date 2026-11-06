from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "student-performance-processed.csv"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

sns.set_theme(style="whitegrid")

print("========== EXPLORATORY DATA ANALYSIS ==========")

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nRisk Distribution:")
print(df["risk"].value_counts())

print("\nRisk Percentage:")
print(df["risk"].value_counts(normalize=True) * 100)

# 1. Risk Distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="risk")
plt.title("Student Academic Risk Distribution")
plt.xlabel("Risk (0 = Not At Risk, 1 = At Risk)")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "risk_distribution.png")
plt.show()

# 2. Age Distribution
plt.figure(figsize=(7, 5))
sns.histplot(data=df, x="age", bins=10, kde=True)
plt.title("Student Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "age_distribution.png")
plt.show()

# 3. Study Time vs Academic Risk
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="studytime", hue="risk")
plt.title("Study Time vs Academic Risk")
plt.xlabel("Weekly Study Time")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "studytime_vs_risk.png")
plt.show()

# 4. Previous Failures vs Academic Risk
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="failures", hue="risk")
plt.title("Previous Failures vs Academic Risk")
plt.xlabel("Number of Previous Failures")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "failures_vs_risk.png")
plt.show()

# 5. Absences vs Academic Risk
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="risk", y="absences")
plt.title("Absences vs Academic Risk")
plt.xlabel("Risk (0 = Not At Risk, 1 = At Risk)")
plt.ylabel("Number of Absences")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "absences_vs_risk.png")
plt.show()

# 6. Correlation Heatmap
numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(12, 9))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Numerical Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
plt.show()

print("\nEDA COMPLETED SUCCESSFULLY!")
print("Charts saved in:", OUTPUT_DIR)