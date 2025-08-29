from flask import Flask, render_template, request
import pandas as pd, joblib, json
from pathlib import Path

# --- Setup ---
ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
META = json.loads((MODELS / "meta.json").read_text())

app = Flask(__name__)

# Load models if present
reg_model = joblib.load(MODELS / "reg_model.pkl") if (MODELS / "reg_model.pkl").exists() else None
cls_model = joblib.load(MODELS / "cls_model.pkl") if (MODELS / "cls_model.pkl").exists() else None

FIELDS = META["used_feature_cols"]
NUMERIC = set(META["numeric_cols"])
CATEG  = set(META["categorical_cols"])

def cast_value(col, val):
    if col in NUMERIC:
        try:
            return float(val)
        except:
            return 0.0
    return val

# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        row = {c: cast_value(c, request.form.get(c, "")) for c in FIELDS}
        X = pd.DataFrame([row])
        outputs = {}

        if reg_model:
            g3 = float(reg_model.predict(X)[0])
            outputs["reg"] = g3

        if cls_model:
            grade = cls_model.predict(X)[0]
            outputs["cls"] = str(grade)

        # Coaching Tips
        tips = []
        if "StudyTimeWeekly" in X.columns and X.loc[0, "StudyTimeWeekly"] < 8:
            tips.append("Increase study time to at least 8–10 hrs/week.")
        if "Absences" in X.columns and X.loc[0, "Absences"] > 10:
            tips.append("Reduce absences; keep attendance consistent.")
        if "Tutoring" in X.columns and str(X.loc[0, "Tutoring"]) in ["0", "no", "No"]:
            tips.append("Consider tutoring for difficult subjects.")
        if "ParentalSupport" in X.columns:
            try:
                if float(X.loc[0, "ParentalSupport"]) <= 1:
                    tips.append("Plan weekly study check-ins for accountability.")
            except:
                pass

        return render_template("result.html", outputs=outputs, meta=META, tips=tips)

    field_specs = [(c, "number" if c in NUMERIC else "text") for c in FIELDS]
    return render_template("index.html", fields=field_specs, meta=META)

if __name__ == "__main__":
    app.run(debug=True)
