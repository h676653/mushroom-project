from flask import Flask, render_template, request, jsonify
import joblib
import json
from pathlib import Path
import numpy as np
import pandas as pd

MODEL_PATH = Path("model/model.joblib")
META_PATH = Path("metadata/features.json")

app = Flask(__name__, template_folder="templates", static_folder="static")

model = joblib.load(MODEL_PATH)
with open(META_PATH, 'r', encoding='utf-8') as f:
    features_meta = json.load(f)

@app.route("/")
def index():
   
    return render_template("index.html", features_meta=features_meta)

@app.route("/predict", methods=["POST"])
def predict():
    
    data = request.json
    
    cols = list(features_meta.keys())
    row = {col: data.get(col, "") for col in cols}
    df = pd.DataFrame([row])
    
    pred = model.predict(df)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[0].tolist()
    label = "poisonous" if int(pred) == 1 else "edible"
    return jsonify({"label": label, "probability": proba})

if __name__ == "__main__":
    app.run(debug=True, port=5000)