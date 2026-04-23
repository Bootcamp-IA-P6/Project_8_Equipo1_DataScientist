from src.models.train import train
import logging
import warnings

# ── MLflow logs ──
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("mlflow.sklearn").setLevel(logging.ERROR)

# ── warnings generales ──
warnings.filterwarnings("ignore")

def main():
    train()

if __name__ == "__main__":
    main()