# train.py
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# --- CONFIG ---
DATA_PATH = Path("data/mushrooms.csv")   # put dataset here
OUT_MODEL = Path("model/model.joblib")
OUT_META = Path("metadata/features.json")
RANDOM_STATE = 42

# --- load data ---
# Expect the UCI mushroom CSV. If you have different columns adjust accordingly.
# UCI dataset uses first column 'class' with values 'e' and 'p' (edible, poisonous).
df = pd.read_csv(DATA_PATH, header=None)  # if CSV has no header, provide names=...
# Assign column names (UCI dataset has 23 columns total)
columns = [
    "class", "cap_shape", "cap_surface", "cap_color", "bruises", "odor",
    "gill_attachment", "gill_spacing", "gill_size", "gill_color",
    "stalk_shape", "stalk_root", "stalk_surface_above_ring",
    "stalk_surface_below_ring", "stalk_color_above_ring",
    "stalk_color_below_ring", "veil_type", "veil_color", "ring_number",
    "ring_type", "spore_print_color", "population", "habitat"
]

df.columns = columns
print(f"Loaded {len(df)} rows with {len(df.columns)} columns")


# target
y = df['class'].map({'e': 0, 'p': 1})  # 0 = eatableble, 1 = poisonous
X = df.drop(columns=['class'])

# collect categorical columns (should be all)
cat_cols = X.columns.tolist()

# Save feature metadata (unique values for each column) for building HTML select options
features_meta = {col: sorted(X[col].dropna().unique().tolist()) for col in cat_cols}
OUT_META.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_META, 'w', encoding='utf-8') as f:
    json.dump(features_meta, f, ensure_ascii=False, indent=2)

# --- train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

# --- pipeline ---
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)]
)

pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1))
])

# Optional: quick hyperparameter tuning (small grid)
param_grid = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [None, 10, 20]
}

search = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1)
print("Starting GridSearchCV...")
search.fit(X_train, y_train)
print("Best params:", search.best_params_)
best = search.best_estimator_

# --- evaluation ---
y_pred = best.predict(X_test)
print("--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=["eatable", "poisonous"]))
print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

# Save model
OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(best, OUT_MODEL)
print(f"Saved model to {OUT_MODEL}")