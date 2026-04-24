"""
test_data.py — Validación del dataset.

Verifica que el CSV carga correctamente, que la limpieza
se aplica y que la estructura es la esperada por el pipeline.
"""

import pandas as pd
from src.data.dataset import get_dataset

EXPECTED_COLUMNS = [
    'gender', 'age', 'hypertension', 'heart_disease',
    'ever_married', 'work_type', 'Residence_type',
    'avg_glucose_level', 'bmi', 'smoking_status', 'stroke'
]


def test_dataset_loads(dataset):
    """El dataset carga sin errores y devuelve un DataFrame no vacío."""
    assert isinstance(dataset, pd.DataFrame)
    assert len(dataset) > 0, 'El dataset está vacío'


def test_expected_columns(dataset):
    """Las 11 columnas originales existen en el dataset."""
    missing = [c for c in EXPECTED_COLUMNS if c not in dataset.columns]
    assert missing == [], f'Columnas faltantes: {missing}'


def test_no_nulls(dataset):
    """No hay valores nulos en ninguna columna."""
    null_counts = dataset.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    assert cols_with_nulls.empty, (
        f'Columnas con NaN: {cols_with_nulls.to_dict()}'
    )


def test_target_is_binary(dataset):
    """La columna stroke solo contiene 0 y 1."""
    unique_vals = set(dataset['stroke'].unique())
    assert unique_vals.issubset({0, 1}), (
        f'Valores inesperados en stroke: {unique_vals}'
    )


def test_work_type_cleaning(dataset):
    """El valor "children" en work_type fue reemplazado por "not_applied"."""
    assert 'children' not in dataset['work_type'].values, (
        'work_type aún contiene "children" — limpieza no aplicada'
    )
