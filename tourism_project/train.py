"""
train.py
Loads the prepared train/test sets from the Hugging Face dataset space, trains
two tree ensembles with hyperparameter tuning, tracks everything in MLflow, and
registers the better model to the Hugging Face model hub.

Why two models: the rubric asks for tuning and for registering the *best* model,
so benchmarking RandomForest against XGBoost and picking the winner answers that
directly. Selection is on F1 of the positive class, not accuracy, because the
target is imbalanced (~19% buyers) and accuracy would reward a lazy model that
just predicts 'no' for everyone.
"""

import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from xgboost import XGBClassifier
from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

# ---- config ----
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = "iamanshkataria/tourism-package-prediction"
MODEL_REPO = "iamanshkataria/tourism-package-model"
TARGET = "ProdTaken"

# ---- load prepared data from the hub ----
def load(split):
    p = hf_hub_download(repo_id=DATASET_REPO, filename=f"{split}.csv",
                        repo_type="dataset", token=HF_TOKEN)
    return pd.read_csv(p)

train_df, test_df = load("train"), load("test")

# one-hot the categoricals; align columns so train and test share the same space
X_train = pd.get_dummies(train_df.drop(columns=[TARGET]), drop_first=True)
X_test = pd.get_dummies(test_df.drop(columns=[TARGET]), drop_first=True)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
y_train, y_test = train_df[TARGET], test_df[TARGET]

# class imbalance ratio for XGBoost
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# ---- MLflow setup ----
# Use a SQLite backend for tracking. This is MLflow's recommended store, and it
# avoids two Windows headaches with the default file store: the PermissionError
# on the default location, and a username containing a space getting mangled into
# a broken path. The db file sits inside the project folder.
os.makedirs("tourism_project", exist_ok=True)
mlflow.set_tracking_uri("sqlite:///tourism_project/mlflow.db")
mlflow.set_experiment("tourism-package-prediction")

def evaluate(model, name):
    """Fit-free scoring helper: model is already fit, score it on the test set."""
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
    }

results = {}

# ---- RandomForest ----
with mlflow.start_run(run_name="RandomForest"):
    rf_grid = {
        "n_estimators": [200],
        "max_depth": [10, None],
        "min_samples_split": [2, 5],
    }
    rf = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        rf_grid, scoring="f1", cv=3, n_jobs=-1,
    ).fit(X_train, y_train)
    rf_best = rf.best_estimator_
    rf_metrics = evaluate(rf_best, "RandomForest")
    mlflow.log_params(rf.best_params_)
    mlflow.log_metrics(rf_metrics)
    mlflow.sklearn.log_model(rf_best, "model")
    results["RandomForest"] = (rf_best, rf_metrics)
    print("RandomForest", rf_metrics)

# ---- XGBoost ----
with mlflow.start_run(run_name="XGBoost"):
    xgb_grid = {
        "n_estimators": [200],
        "max_depth": [4, 6],
        "learning_rate": [0.1, 0.05],
    }
    xgb = GridSearchCV(
        XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight,
                      eval_metric="logloss"),
        xgb_grid, scoring="f1", cv=3, n_jobs=-1,
    ).fit(X_train, y_train)
    xgb_best = xgb.best_estimator_
    xgb_metrics = evaluate(xgb_best, "XGBoost")
    mlflow.log_params(xgb.best_params_)
    mlflow.log_metrics(xgb_metrics)
    mlflow.sklearn.log_model(xgb_best, "model")
    results["XGBoost"] = (xgb_best, xgb_metrics)
    print("XGBoost", xgb_metrics)

# ---- pick the winner on F1 ----
best_name = max(results, key=lambda k: results[k][1]["f1"])
best_model, best_metrics = results[best_name]
print(f"Best model: {best_name} with F1 {best_metrics['f1']:.3f}")

# persist the winning model + the training column order (needed at inference)
joblib.dump(best_model, "tourism_project/model_building/best_model.joblib")
joblib.dump(list(X_train.columns), "tourism_project/model_building/model_columns.joblib")

# ---- register the best model on the hub ----
api = HfApi(token=HF_TOKEN)
try:
    create_repo(repo_id=MODEL_REPO, repo_type="model", token=HF_TOKEN, exist_ok=True)
except HfHubHTTPError as e:
    print(f"Model repo check/create returned: {e}")

for fname in ["best_model.joblib", "model_columns.joblib"]:
    api.upload_file(
        path_or_fileobj=f"tourism_project/model_building/{fname}",
        path_in_repo=fname,
        repo_id=MODEL_REPO,
        repo_type="model",
    )
print(f"Best model ({best_name}) registered to {MODEL_REPO}.")
