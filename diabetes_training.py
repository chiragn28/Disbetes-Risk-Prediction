###############################################################################
# Diabetes Risk Prediction Prototype (Corrected & Robust)
###############################################################################

import pathlib, warnings, json, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import shap

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# 1. Load Dataset
def load_pima(path: str | pathlib.Path | None = None) -> pd.DataFrame:
    if path is None:
        url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
        return pd.read_csv(url)
    return pd.read_csv(path)

df = load_pima("diabetes.csv")  # Change path if needed
print("Dataset shape:", df.shape)

# 2. Train/Test Split
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

# 3. Preprocessing Pipeline
numeric_features = X.columns.tolist()
preprocess = ColumnTransformer(
    [("num", StandardScaler(), numeric_features)], remainder="drop"
)

# 4. Model Definitions & Hyper-tuning
models = {
    "log_reg": {
        "estimator": LogisticRegression(max_iter=200, solver="lbfgs"),
        "param_grid": {"log_reg__C": [0.1, 1, 10], "log_reg__penalty": ["l2"]}
    },
    "rf": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
        "param_grid": {"rf__n_estimators": [100, 300], "rf__max_depth": [None, 5, 10]}
    }
}

results = {}
for name, cfg in models.items():
    pipe = Pipeline([("pre", preprocess), (name, cfg["estimator"])])
    gs = GridSearchCV(
        pipe, param_grid=cfg["param_grid"], cv=5, scoring="f1", n_jobs=-1, verbose=0
    )
    gs.fit(X_train, y_train)
    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }
    results[name] = {"model": best_model, "metrics": metrics, "y_pred": y_pred}
    print(f"\n{name.upper()}  → best params:", gs.best_params_)
    print("  → metrics:", json.dumps(metrics, indent=2))

# 5. Bar chart comparison
model_names = [k.upper() for k in results]
x = np.arange(len(model_names))
width = 0.2
plt.figure(figsize=(8,5))
plt.bar(x - 1.5*width, [results[k]["metrics"]["accuracy"] for k in results], width, label='Accuracy')
plt.bar(x - 0.5*width, [results[k]["metrics"]["precision"] for k in results], width, label='Precision')
plt.bar(x + 0.5*width, [results[k]["metrics"]["recall"] for k in results], width, label='Recall')
plt.bar(x + 1.5*width, [results[k]["metrics"]["f1"] for k in results], width, label='F1')
plt.xticks(x, model_names)
plt.ylim(0, 1)
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.legend()
plt.tight_layout()
plt.show()

# 6. Choose Champion & Print Report
champion_key = max(results, key=lambda k: results[k]["metrics"]["f1"])
champion = results[champion_key]["model"]
print(f"\nSelected champion model: {champion_key}")

cm = confusion_matrix(y_test, results[champion_key]["y_pred"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=champion.classes_)
disp.plot(cmap="Blues")
plt.title(f"{champion_key.upper()} – Confusion Matrix")
plt.show()

print("\nDetailed report:\n", classification_report(y_test, results[champion_key]["y_pred"]))

# 7. SHAP Feature Importance (robust shape check)
try:
    preprocessor = champion.named_steps['pre']
    model = champion.named_steps[champion_key]
    X_test_transformed = preprocessor.transform(X_test)
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()
    shap_explainer = shap.TreeExplainer(model)
    shap_values = shap_explainer.shap_values(X_test_transformed)
    print("shap_values shape:", np.array(shap_values).shape)
    # shap_values will now be (154, 8, 2)

    # Use only class 1 scores for explanation (i.e., diabetes positive)
    shap_values_class1 = shap_values[:, :, 1]  # shape: (154, 8)

    shap.summary_plot(
        shap_values_class1,
        X_test_transformed,
        feature_names=numeric_features,
        show=True
    )
except Exception as e:
    print("SHAP explainability failed:", e)

# 8. Persist Model
model_path = "diabetes_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(champion, f)
print(f"\n🎉 Saved champion model → {model_path}")
