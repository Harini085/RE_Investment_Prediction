"""Train a classifier for the explicitly limited relative-value proxy."""
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .data_preprocessing import make_preprocessor, prepare_classification
from .utils import DATA_PATH, MODEL_DIR, RANDOM_STATE, REPORT_DIR, get_mlflow, load_data

# These are the concrete types used by our locally trained sklearn pipelines;
# do not add arbitrary types reported by skops for externally supplied models.
SKOPS_TRUSTED_MODEL_TYPES = ["numpy.dtype", "sklearn.tree._tree.Tree"]


def train():
    mlflow = get_mlflow()
    raw = load_data()
    X, y = prepare_classification(raw)
    if y.nunique() < 2 or y.value_counts().min() < 2:
        raise ValueError("Peer-median proxy does not provide at least two classes with two examples each.")
    stratify = y if y.value_counts().min() >= 2 else None
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=RANDOM_STATE, stratify=y_trainval
    )
    candidates = [
        ("LogisticRegression", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        ("RandomForestClassifier", RandomForestClassifier(n_estimators=100, max_depth=18,
         min_samples_leaf=8, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE)),
    ]
    MODEL_DIR.mkdir(exist_ok=True); REPORT_DIR.mkdir(exist_ok=True)
    figures = REPORT_DIR / "figures"; figures.mkdir(parents=True, exist_ok=True)
    results = []
    fitted_candidates = []
    best = None
    for name, estimator in candidates:
        pipeline = Pipeline([("preprocess", make_preprocessor(X_train)), ("model", estimator)])
        with mlflow.start_run(run_name=f"classification-{name}") as run:
            pipeline.fit(X_train, y_train)
            pred = pipeline.predict(X_val)
            metrics = {
                "validation_accuracy": float(accuracy_score(y_val, pred)),
                "validation_precision": float(precision_score(y_val, pred, zero_division=0)),
                "validation_recall": float(recall_score(y_val, pred, zero_division=0)),
                "validation_f1": float(f1_score(y_val, pred, zero_division=0)),
            }
            mlflow.log_params({"model": name, "random_state": RANDOM_STATE,
                "validation_rows": len(X_val), "training_rows": len(X_train), "test_rows": len(X_test),
                "proxy": "Price_per_SqFt below median within City and Property_Type",
                "source_file": DATA_PATH.name})
            if hasattr(estimator, "get_params"):
                mlflow.log_params({f"model_{k}": v for k, v in estimator.get_params(deep=False).items()
                                   if isinstance(v, (str, int, float, bool, type(None)))})
            mlflow.log_metrics(metrics)
            cm = confusion_matrix(y_val, pred, labels=[0, 1])
            fig, ax = plt.subplots(figsize=(5, 4)); ax.imshow(cm, cmap="Blues"); ax.set(title=f"{name}: validation proxy confusion matrix", xlabel="Predicted", ylabel="Observed", xticks=[0,1], yticks=[0,1]);
            for (i, j), value in np.ndenumerate(cm): ax.text(j, i, str(value), ha="center", va="center")
            fig.tight_layout(); cm_path = figures / f"confusion_matrix_{name}.png"; fig.savefig(cm_path, dpi=130); plt.close(fig)
            mlflow.log_artifact(str(cm_path), artifact_path="evaluation")
            report_path = REPORT_DIR / f"classification_report_{name}_validation.txt"
            report_path.write_text(classification_report(y_val, pred, zero_division=0), encoding="utf-8")
            mlflow.log_artifact(str(report_path), artifact_path="evaluation")
            try:
                signature = mlflow.models.infer_signature(X_train.head(5), pipeline.predict(X_train.head(5)))
                mlflow.sklearn.log_model(pipeline, name="model", signature=signature,
                                         input_example=X_train.head(3),
                                         skops_trusted_types=SKOPS_TRUSTED_MODEL_TYPES)
            except (TypeError, AttributeError) as exc:
                warnings.warn(f"Model signature logging skipped for this MLflow version: {exc}")
                mlflow.sklearn.log_model(pipeline, artifact_path="model", signature=signature,
                                         input_example=X_train.head(3))
            result = {"model": name, **metrics, "run_id": run.info.run_id}
            results.append(result)
            fitted_candidates.append((name, pipeline))
            if best is None or metrics["validation_f1"] > best["validation_f1"]:
                best = {**result, "pipeline": pipeline}
    logistic = next((candidate for candidate in results if candidate["model"] == "LogisticRegression"), None)
    if logistic and best["validation_f1"] - logistic["validation_f1"] <= 0.01:
        # Prefer the simpler, much smaller pipeline when validation F1 is effectively tied.
        best = {**logistic, "pipeline": next(p for n, p in fitted_candidates if n == "LogisticRegression")}
    best["pipeline"].fit(X_trainval, y_trainval)
    test_pred = best["pipeline"].predict(X_test)
    test_metrics = {
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "f1": float(f1_score(y_test, test_pred, zero_division=0)),
    }
    table = pd.DataFrame(results).drop(columns="run_id")
    table.to_csv(REPORT_DIR / "classification_model_comparison.csv", index=False)
    with mlflow.start_run(run_name="classification-selected-model"):
        mlflow.log_params({"selected_model": best["model"], "selection_metric": "validation_f1",
                           "selection_rule": "highest validation F1; prefer Logistic Regression when within 0.01 F1",
                           "selected_candidate_run_id": best["run_id"], "dataset_rows": len(raw),
                           "final_training_rows": len(X_trainval), "final_test_rows": len(X_test)})
        mlflow.log_metrics({f"final_test_{key}": value for key, value in test_metrics.items()})
        mlflow.log_artifact(str(REPORT_DIR / "classification_model_comparison.csv"))
        selected_cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4)); ax.imshow(selected_cm, cmap="Blues"); ax.set(title="Selected model: final test confusion matrix", xlabel="Predicted", ylabel="Observed", xticks=[0,1], yticks=[0,1])
        for (i, j), value in np.ndenumerate(selected_cm): ax.text(j, i, str(value), ha="center", va="center")
        fig.tight_layout(); selected_cm_path = figures / "confusion_matrix_selected_test.png"; fig.savefig(selected_cm_path, dpi=130); plt.close(fig)
        mlflow.log_artifact(str(selected_cm_path), artifact_path="evaluation")
        try:
            signature = mlflow.models.infer_signature(X_trainval.head(5), best["pipeline"].predict(X_trainval.head(5)))
            mlflow.sklearn.log_model(best["pipeline"], name="model", signature=signature,
                                     input_example=X_trainval.head(3),
                                     skops_trusted_types=SKOPS_TRUSTED_MODEL_TYPES)
        except (TypeError, AttributeError) as exc:
            warnings.warn(f"Selected model signature logging skipped: {exc}")
            mlflow.sklearn.log_model(best["pipeline"], artifact_path="model", signature=signature,
                                     input_example=X_trainval.head(3))
    final_model = best["pipeline"].named_steps["model"]
    if hasattr(final_model, "feature_importances_") or hasattr(final_model, "coef_"):
        try:
            names = best["pipeline"].named_steps["preprocess"].get_feature_names_out()
            weights = (np.asarray(final_model.feature_importances_) if hasattr(final_model, "feature_importances_")
                       else np.mean(np.abs(np.asarray(final_model.coef_)), axis=0))
            top = pd.Series(weights, index=names).nlargest(20).sort_values()
            fig, ax = plt.subplots(figsize=(9, 7)); top.plot.barh(ax=ax); ax.set(title=f"Top model features: {best['model']}", xlabel="Importance (RF) / absolute coefficient (Logistic Regression)"); fig.tight_layout()
            importance_path = figures / "feature_importance.png"; fig.savefig(importance_path, dpi=140); plt.close(fig)
            with mlflow.start_run(run_name="classification-feature-importance", nested=True):
                mlflow.log_artifact(str(importance_path), artifact_path="interpretation")
        except (AttributeError, ValueError) as exc:
            warnings.warn(f"Feature importance could not be produced: {exc}")
    joblib.dump({"pipeline": best["pipeline"], "features": X.columns.tolist(),
        "target_name": "Below peer median current price per sq ft (relative-value proxy)",
        "target_definition": "Price_per_SqFt below median among records sharing City and Property_Type",
        "metrics": test_metrics,
        "model_name": best["model"], "random_state": RANDOM_STATE}, MODEL_DIR / "classification_pipeline.joblib")
    (REPORT_DIR / "classification_metrics.json").write_text(json.dumps({"proxy_only": True,
        "selection_metric": "validation_f1",
        "selection_rule": "Highest validation F1; prefer Logistic Regression when within 0.01 F1",
        "selected_model": best["model"], "results": results,
        "final_test_metrics": test_metrics,
        "class_balance": {str(k): int(v) for k, v in y.value_counts().items()}}, indent=2), encoding="utf-8")
    print(table.to_string(index=False)); print(f"Saved {MODEL_DIR / 'classification_pipeline.joblib'}")
    return results


if __name__ == "__main__":
    train()
