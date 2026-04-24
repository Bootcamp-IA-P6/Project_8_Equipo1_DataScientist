import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data/raw/stroke_dataset.csv"

def get_dataset(version='full', add_features=False):

    df = pd.read_csv(DATA_PATH)

    df.loc[df['work_type'] == 'children', 'work_type'] = 'not_applied'

    if version == 'adults':
        df = df[df['age'] >= 17].copy()

    if add_features:
        df['age_group'] = pd.cut(
            df['age'],
            bins=[0, 17, 40, 60, 100],
            labels=['child', 'young', 'adult', 'senior']
        ).astype(str)

        df['high_risk'] = (
            (df['hypertension'] == 1) |
            (df['heart_disease'] == 1)
        ).astype(int)

        df['age_glucose'] = df['age'] * df['avg_glucose_level']

    return df