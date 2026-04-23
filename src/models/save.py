import joblib
import json
from pathlib import Path


def save_artifacts(pipeline, threshold, metrics, config):

    MODEL_DIR = Path("models")
    MODEL_DIR.mkdir(exist_ok=True)

    model_path = MODEL_DIR / f"{config['model_name']}_final.joblib"
    threshold_path = MODEL_DIR / "threshold.txt"
    metadata_path = MODEL_DIR / "metadata.json"

    print("\n💾 Guardando artefactos...")

    # Modelo
    joblib.dump(pipeline, model_path)

    # Threshold
    threshold_path.write_text(str(threshold))

    # Metadata (clave para UI)
    metadata = {
        "model_name": config["model_name"],
        "threshold": threshold,
        "metrics": metrics,
        "features": list(pipeline.feature_names_in_),
        "scale_pos_weight": config["scale_pos_weight"]
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"✔ Modelo: {model_path}")
    print(f"✔ Threshold: {threshold}")
    print(f"✔ Metadata: {metadata_path}")