"""
conftest.py — Fixtures compartidos para todos los tests.

pytest los inyecta automáticamente por nombre de parámetro.
Se ejecutan una sola vez por sesión (scope='session') para no
repetir carga de datos ni entrenamiento en cada test.
"""

import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.data.dataset import get_dataset
from src.features.build_features import build_preprocessor
from src.models.pipeline import build_pipeline
from src.config.model_config import MODEL_CONFIG

RANDOM_STATE = 42
TEST_SIZE    = 0.2


@pytest.fixture(scope='session')
def dataset() -> pd.DataFrame:
    """Dataset completo cargado y limpio (sin split)."""
    return get_dataset(
        add_features=MODEL_CONFIG['add_features']
    )


@pytest.fixture(scope='session')
def splits(dataset):
    """Train/test split estratificado — mismo seed que train.py."""
    X = dataset.drop('stroke', axis=1)
    y = dataset['stroke']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        stratify=y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope='session')
def fitted_pipeline(splits):
    """
    Pipeline completo entrenado sobre X_train.
    Compartido por test_features y test_model — se entrena una sola vez.
    """
    X_train, _, y_train, _ = splits

    params = MODEL_CONFIG['params'].copy()
    params['scale_pos_weight'] = MODEL_CONFIG['scale_pos_weight']

    model        = XGBClassifier(**params)
    preprocessor = build_preprocessor(X_train, model_ty='tree')
    pipeline     = build_pipeline(
        model,
        preprocessor,
        use_smote=MODEL_CONFIG['use_smote']
    )
    pipeline.fit(X_train, y_train)
    return pipeline
