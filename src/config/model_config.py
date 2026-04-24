
MODEL_CONFIG = {
    "model_name": "XGB_optuna",

    "model_type": "xgboost",

    "params": {
        "n_estimators": 196,
        "max_depth": 3,
        "learning_rate": 0.01374894555331037,
        "subsample": 0.7390533721986331,
        "colsample_bytree": 0.6999018540590647,
        "reg_alpha": 6.764752472270359e-05,
        "reg_lambda": 0.025574091808629934,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1
    },
    
    "use_smote": False,
    "add_features": False,
    "threshold": 0.5517,
    "scale_pos_weight": 19.1
}