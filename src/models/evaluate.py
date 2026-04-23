from sklearn.metrics import (
    roc_auc_score, recall_score, f1_score,
    precision_score, average_precision_score
)


def evaluate_model(pipeline, X_test, y_test, threshold):

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }


    return metrics, y_pred, y_prob