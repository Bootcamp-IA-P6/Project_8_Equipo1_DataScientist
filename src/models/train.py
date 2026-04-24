from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.data.dataset import get_dataset
from src.features.build_features import build_preprocessor
from src.models.pipeline import build_pipeline
from src.models.evaluate import evaluate_model
from src.models.save import save_artifacts
from src.config.model_config import MODEL_CONFIG
from src.models.mlflow_logger import log_to_mlflow

RANDOM_STATE = 42
TEST_SIZE = 0.2


def build_model():
    if MODEL_CONFIG["model_type"] == "xgboost":
        params = MODEL_CONFIG["params"].copy()
        params["scale_pos_weight"] = MODEL_CONFIG["scale_pos_weight"]
        print(f"⚖️ scale_pos_weight (config): {params['scale_pos_weight']}")

        return XGBClassifier(**params)
    else:
        raise ValueError("Modelo no soportado aún")

def train():

    print("\n🚀 INICIANDO TRAIN PIPELINE")
    print("="*50)

    # ── DATA ──
    df = get_dataset(add_features=MODEL_CONFIG["add_features"])

    X = df.drop("stroke", axis=1)
    y = df["stroke"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print(f"📊 Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"⚖️ Positivos train: {y_train.sum()}")

    # ── MODEL ──
    model = build_model()
    preprocessor = build_preprocessor(X_train, model_ty="tree")

    pipeline = build_pipeline(
        model,
        preprocessor,
        use_smote=MODEL_CONFIG["use_smote"]
    )

    print("\n🧠 Entrenando modelo...")
    pipeline.fit(X_train, y_train)


    # ── FEATURE INFO (para Streamlit) ──
    print("\n🧩 Generando feature_info...")

    feature_info = {}

    for col in X_train.columns:
        if X_train[col].dtype == "object":
            feature_info[col] = {
                "type": "categorical",
                "values": sorted(list(X_train[col].dropna().unique()))
            }
        else:
            feature_info[col] = {
                "type": "numeric",
                "min": float(X_train[col].min()),
                "max": float(X_train[col].max())
            }

    print("✔ feature_info generado")

    # ── EVALUACIÓN ──
    print("\n📈 Evaluación en TEST")
    metrics, y_pred, y_prob = evaluate_model(
        pipeline,
        X_test,
        y_test,
        threshold=MODEL_CONFIG["threshold"]
    )

    for k, v in metrics.items():
        print(f"{k:12s}: {v:.4f}")

    # ── MLFlow ──
    
    log_to_mlflow(config = MODEL_CONFIG,
    metrics = metrics, pipeline = pipeline,
    y_test = y_test, y_pred = y_pred,
    y_prob = y_prob)
    
    # ── SAVE ──
    save_artifacts(
        pipeline,
        threshold=MODEL_CONFIG["threshold"],
        metrics=metrics,
        config=MODEL_CONFIG,
        feature_info=feature_info        
    )

    print("\n✅ Pipeline completo")
    return pipeline