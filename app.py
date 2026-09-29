from flask import Flask, render_template

import pandas as pd
import joblib

app = Flask(__name__)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_csv("clean_dataset.csv")

# --------------------------------------------------
# Load Models
# --------------------------------------------------

placement_model = joblib.load("model/placement_model.pkl")
salary_model = joblib.load("model/salary_model.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")

# --------------------------------------------------
# Dashboard Statistics
# --------------------------------------------------

TOTAL_STUDENTS = len(df)

PLACED_STUDENTS = len(df[df["PlacementStatus"] == 1])

NOT_PLACED = TOTAL_STUDENTS - PLACED_STUDENTS

PLACEMENT_PERCENTAGE = round(
    (PLACED_STUDENTS / TOTAL_STUDENTS) * 100,
    2
)

AVERAGE_SALARY = round(
    df[df["PlacementStatus"] == 1]["Salary Package"].mean(),
    2
)

HIGHEST_SALARY = round(
    df["Salary Package"].max(),
    2)

AVERAGE_CGPA = round(
    df["CGPA"].mean(),
    2)

# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.route("/")
def dashboard():

    metrics = {

        "total_students": TOTAL_STUDENTS,

        "placed_students": PLACED_STUDENTS,

        "not_placed": NOT_PLACED,

        "placement_percentage": PLACEMENT_PERCENTAGE,

        "average_salary": AVERAGE_SALARY,

        "highest_salary": HIGHEST_SALARY,

        "average_cgpa": AVERAGE_CGPA

    }

    recent_students = df.head(10).to_dict(orient="records")

    return render_template(
        "dashboard.html",
        metrics=metrics,
        recent_students=recent_students
    )


@app.route("/students")
def students():

    students = df.head(100).to_dict(orient="records")

    return render_template(
        "students.html",
        students=students
    )


@app.route("/predict")
def predict():

    return render_template("predict.html")


@app.route("/analytics")
def analytics():

    return render_template("analytics.html")


if __name__ == "__main__":
    app.run(debug=True)