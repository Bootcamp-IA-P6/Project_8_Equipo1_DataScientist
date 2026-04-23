import mlflow
import mlflow.sklearn
from pathlib import Path
from src.models.plots import log_confusion_matrix

import warnings

def log_to_mlflow(config, metrics, pipeline, y_test, y_pred, y_prob,
                  model_name="model", experiment_name="Ictus_Project"):

    warnings.filterwarnings('ignore')
    mlflow.set_experiment(experiment_name)

    run_name = f"{config['model_name']}_final"

    with mlflow.start_run(run_name=run_name):

        # ── METRICS ──
        mlflow.log_metric("auc", metrics["auc"])
        mlflow.log_metric("pr_auc", metrics["pr_auc"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("f1", metrics["f1"])

        # ── CONFUSION MATRIX ──
        cm_path = log_confusion_matrix(y_test, y_pred, run_name=run_name)
        mlflow.log_artifact(cm_path)

        # ── THRESHOLD ──
        mlflow.log_metric("threshold", config["threshold"])

        # ── PARAMS ──
        for k, v in config["params"].items():
            mlflow.log_param(k, v)

        mlflow.log_param("model_type", config["model_type"])
        mlflow.log_param("use_smote", config["use_smote"])
        mlflow.log_param("add_features", config["add_features"])

        if "scale_pos_weight" in config:
            mlflow.log_param("scale_pos_weight", config["scale_pos_weight"])

        # ── MODELO ──
        name_model = f"{config['model_name']}"

        mlflow.sklearn.log_model(sk_model= pipeline, name=name_model,     
                                 pip_requirements=["scikit-learn==1.7.2",
                                                   "xgboost", "imbalanced-learn",
                                                   "mlflow"]
                                                   )

        print("📊 Logged to MLflow")