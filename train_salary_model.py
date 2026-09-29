import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

print("=" * 60)
print("SALARY PREDICTION MODEL")
print("=" * 60)

# ------------------------
# Load Dataset
# ------------------------

df = pd.read_csv("clean_dataset.csv")

# ------------------------
# Encode Categorical Data
# ------------------------

categorical_columns = [
    "Gender",
    "City",
    "CollegeTier",
    "Stream",
    "Specialisation",
    "Hostel",
    "HistoryOfBacklogs",
    "CGPA_Tier"
]

for column in categorical_columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])

# ------------------------
# Use only placed students
# ------------------------

df = df[df["PlacementStatus"] == 1]

# ------------------------
# Features
# ------------------------

X = df.drop(columns=["Salary Package", "PlacementStatus"])

y = df["Salary Package"]

# ------------------------
# Train/Test Split
# ------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ------------------------
# Train Model
# ------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("R² Score :", round(r2_score(y_test, prediction), 3))
print("MAE      :", round(mean_absolute_error(y_test, prediction), 3), "LPA")

joblib.dump(model, "model/salary_model.pkl")

print("\nSalary model saved successfully.")