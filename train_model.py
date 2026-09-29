import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

print("=" * 60)
print("PLACEMENT MODEL TRAINING")
print("=" * 60)

# ---------------------------------------------------
# Load Clean Dataset
# ---------------------------------------------------

df = pd.read_csv("clean_dataset.csv")

# ---------------------------------------------------
# Encode Categorical Columns
# ---------------------------------------------------

label_encoders = {}

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
    label_encoders[column] = encoder

# Save encoders
joblib.dump(label_encoders, "model/label_encoders.pkl")

# ---------------------------------------------------
# Features and Target
# ---------------------------------------------------

X = df.drop(columns=["PlacementStatus", "Salary Package"])

y = df["PlacementStatus"]

# ---------------------------------------------------
# Split Dataset
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

# ---------------------------------------------------
# Models
# ---------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}

best_model = None
best_accuracy = 0

print("\nMODEL ACCURACIES\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    print(f"{name:<25} : {accuracy*100:.2f}%")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# ---------------------------------------------------
# Save Best Model
# ---------------------------------------------------

joblib.dump(best_model, "model/placement_model.pkl")

print("\nBest Accuracy :", round(best_accuracy*100,2), "%")

print("\nModel saved successfully.")