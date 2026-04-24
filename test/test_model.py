"""
test_model.py — Validación del pipeline entrenado.

Verifica que el pipeline predice correctamente, que las
probabilidades están en rango válido y que el threshold
produce salida binaria coherente.
"""

import numpy as np
from src.config.model_config import MODEL_CONFIG


def test_pipeline_predicts(fitted_pipeline, splits):
    """predict_proba no lanza excepciones sobre el test set."""
    _, X_test, _, _ = splits
    try:
        fitted_pipeline.predict_proba(X_test)
    except Exception as e:
        raise AssertionError(f'predict_proba falló: {e}')


def test_proba_range(fitted_pipeline, splits):
    """Todas las probabilidades de la clase positiva están en [0, 1]."""
    _, X_test, _, _ = splits
    y_prob = fitted_pipeline.predict_proba(X_test)[:, 1]

    assert (y_prob >= 0).all() and (y_prob <= 1).all(), (
        f'Probabilidades fuera de rango — '
        f'min={y_prob.min():.4f}, max={y_prob.max():.4f}'
    )


def test_proba_shape(fitted_pipeline, splits):
    """La salida de predict_proba tiene la misma longitud que el test set."""
    _, X_test, _, _ = splits
    y_prob = fitted_pipeline.predict_proba(X_test)[:, 1]

    assert len(y_prob) == len(X_test), (
        f'Shape mismatch — y_prob: {len(y_prob)}, X_test: {len(X_test)}'
    )


def test_threshold_produces_binary_output(fitted_pipeline, splits):
    """
    Aplicar el threshold del config produce una salida estrictamente binaria.
    Solo debe contener 0 y 1 — sin otros valores.
    """
    _, X_test, _, _ = splits
    threshold = MODEL_CONFIG['threshold']

    y_prob = fitted_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    unique_preds = set(np.unique(y_pred))
    assert unique_preds.issubset({0, 1}), (
        f'y_pred contiene valores inesperados: {unique_preds}'
    )
    assert len(unique_preds) == 2, (
        f'y_pred solo contiene una clase ({unique_preds}) — '
        'threshold puede ser demasiado extremo'
    )
