# app_gradio.py
import joblib
import json
import pandas as pd
import gradio as gr

# paths
MODEL_PATH = "model/model.joblib"
META_PATH = "metadata/features.json"

# load model + feature metadata
model = joblib.load(MODEL_PATH)
with open(META_PATH, "r") as f:
    features_meta = json.load(f)

# get feature names in correct order
feature_names = list(features_meta.keys())

def predict(*inputs):
    # inputs is a tuple, map to dataframe
    x = pd.DataFrame([inputs], columns=feature_names)
    y_proba = model.predict_proba(x)[0]
    y_pred = model.predict(x)[0]

    label = "Edible" if y_pred == 0 else "Poisonous"
    proba_str = f"Edible: {y_proba[0]:.2f}, Poisonous: {y_proba[1]:.2f}"

    return f"{label} ({proba_str})"

# build gradio UI
inputs = [
    gr.Dropdown(choices=features_meta[col], label=col)
    for col in feature_names
]

output = gr.Textbox(label="Prediction Result")

demo = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=output,
    title="🍄 Mushroom Classifier",
    description="Select mushroom features and see whether it's edible or poisonous."
)

demo.launch()