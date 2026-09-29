import pandas as pd

# ----------------------------
# Load Dataset
# ----------------------------

df = pd.read_csv("placement_predict_50k Dataset.csv")

print("=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

# ----------------------------
# Missing Values
# ----------------------------

missing_columns = [
    "Workshops",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "CodingTestScore",
    "MockInterviewScore"
]

for column in missing_columns:
    df[column] = df[column].fillna(df[column].median())

print("\nMissing Values After Cleaning\n")
print(df.isnull().sum())

# ----------------------------
# Remove Unnecessary Columns
# ----------------------------

df = df.drop(columns=["StudentID", "IsAnomaly"])

print("\nRemaining Columns")
print(df.columns.tolist())

# ----------------------------
# Save Clean Dataset
# ----------------------------

df.to_csv("clean_dataset.csv", index=False)

print("\nClean dataset saved successfully.")
print("Shape :", df.shape)