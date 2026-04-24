"""
test_features.py — Validación del preprocesador.

Verifica que build_preprocessor construye correctamente el
ColumnTransformer y que la transformación no introduce errores
ni NaNs en los datos.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from src.features.build_features import build_preprocessor


def test_preprocessor_builds(splits):
    """build_preprocessor devuelve un ColumnTransformer sin errores."""
    X_train, _, _, _ = splits
    preprocessor = build_preprocessor(X_train, model_ty='linear')
    assert isinstance(preprocessor, ColumnTransformer)


def test_output_shape(splits):
    """La salida transformada tiene el mismo número de filas que el input."""
    X_train, X_test, _, _ = splits
    preprocessor = build_preprocessor(X_train, model_ty='tree')
    preprocessor.fit(X_train)

    X_transformed = preprocessor.transform(X_test)
    assert X_transformed.shape[0] == X_test.shape[0], (
        f'Filas entrada: {X_test.shape[0]} — '
        f'Filas salida: {X_transformed.shape[0]}'
    )


def test_no_nan_after_transform(splits):
    """El preprocesado no introduce NaNs en la salida."""
    X_train, X_test, _, _ = splits
    preprocessor = build_preprocessor(X_train, model_ty='tree')
    preprocessor.fit(X_train)

    X_transformed = preprocessor.transform(X_test)

    # toarray() por si la salida es sparse matrix (OHE)
    if hasattr(X_transformed, 'toarray'):
        X_transformed = X_transformed.toarray()

    assert not np.isnan(X_transformed).any(), (
        'El preprocesado introdujo NaNs en la salida'
    )


def test_linear_uses_scaler_tree_uses_passthrough(splits):
    """
    model_ty='linear' aplica StandardScaler en numéricas.
    model_ty='tree' usa passthrough (sin escalar).
    """
    X_train, _, _, _ = splits

    prep_linear = build_preprocessor(X_train, model_ty='linear')
    prep_tree   = build_preprocessor(X_train, model_ty='tree')

    # Transformer de numéricas es el primero (índice 0)
    _, linear_transformer, _ = prep_linear.transformers[0]
    _, tree_transformer, _   = prep_tree.transformers[0]

    assert isinstance(linear_transformer, StandardScaler), (
        f'model_ty="linear" debería usar StandardScaler, '
        f'encontrado: {type(linear_transformer)}'
    )
    assert tree_transformer == 'passthrough', (
        f'model_ty="tree" debería usar passthrough, '
        f'encontrado: {tree_transformer}'
    )
