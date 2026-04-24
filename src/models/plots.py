import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path


def log_confusion_matrix(y_test, y_pred, run_name="model"):

    # Crear figura
    fig, ax = plt.subplots(figsize=(5, 4))

    ConfusionMatrixDisplay(
        confusion_matrix(y_test, y_pred),
        display_labels=["No Stroke", "Stroke"]
    ).plot(ax=ax)

    ax.set_title(f"Confusion Matrix - {run_name}")

    # Guardar imagen
    output_dir = Path("assets")
    output_dir.mkdir(exist_ok=True)

    file_path = output_dir / f"confusion_matrix_{run_name}_final.png"
    plt.savefig(file_path, dpi=100, bbox_inches="tight")
    plt.close()

    return file_path