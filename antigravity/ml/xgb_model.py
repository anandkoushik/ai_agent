"""
antigravity.ml.xgb_model — XGBoost Prediction Pipeline

Provides train, predict, save, and load utilities for XGBoost models
on structured/tabular datasets (CSV, DataFrame).

Usage:
    from antigravity.ml.xgb_model import train_xgb, predict, save_model, load_model

    model, metrics = train_xgb("data.csv", target_column="label")
    predictions    = predict(model, X_new)
    save_model(model, "my_model.json")
    model          = load_model("my_model.json")
"""

import os
import logging
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import — keeps startup fast
# ---------------------------------------------------------------------------
_xgb = None


def _get_xgb():
    global _xgb
    if _xgb is None:
        try:
            import xgboost as xgb
            _xgb = xgb
        except ImportError as e:
            raise ImportError(
                "xgboost is not installed. Run: pip install xgboost"
            ) from e
    return _xgb


# ============================================================
#  TRAIN
# ============================================================

def train_xgb(
    data_source: Union[str, pd.DataFrame],
    target_column: str = "label",
    test_size: float = 0.2,
    params: Optional[Dict[str, Any]] = None,
    num_boost_round: int = 100,
    early_stopping_rounds: int = 10,
    task: str = "auto",
) -> Tuple[Any, Dict[str, float]]:
    """
    Train an XGBoost model on tabular data.

    Args:
        data_source:            Path to a CSV file or a pandas DataFrame.
        target_column:          Name of the target / label column.
        test_size:              Fraction held out for validation (0-1).
        params:                 XGBoost booster params (auto-detected if None).
        num_boost_round:        Maximum boosting rounds.
        early_stopping_rounds:  Stop if val metric doesn't improve for N rounds.
        task:                   'classification', 'regression', or 'auto'.

    Returns:
        (trained_booster, metrics_dict)
    """
    xgb = _get_xgb()
    from sklearn.model_selection import train_test_split

    # --- Load data ---
    if isinstance(data_source, str):
        if not os.path.exists(data_source):
            raise FileNotFoundError(f"Dataset not found: {data_source}")
        df = pd.read_csv(data_source)
        logger.info(f"[xgb_model] Loaded CSV with {len(df)} rows, {len(df.columns)} cols")
    elif isinstance(data_source, pd.DataFrame):
        df = data_source.copy()
    else:
        raise TypeError("data_source must be a file path (str) or pandas DataFrame")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # --- Split ---
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Auto-detect task type
    if task == "auto":
        n_unique = y.nunique()
        task = "classification" if n_unique <= 20 else "regression"
        logger.info(f"[xgb_model] Auto-detected task: {task} ({n_unique} unique target values)")

    # Encode categoricals
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        logger.info(f"[xgb_model] One-hot encoded {len(cat_cols)} categorical columns")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # --- Default params ---
    if params is None:
        if task == "classification":
            n_classes = y.nunique()
            if n_classes == 2:
                params = {
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "seed": 42,
                }
            else:
                params = {
                    "objective": "multi:softprob",
                    "num_class": n_classes,
                    "eval_metric": "mlogloss",
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "seed": 42,
                }
        else:
            params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "seed": 42,
            }

    # --- Train ---
    evals = [(dtrain, "train"), (dval, "val")]
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=10,
    )

    # --- Metrics ---
    metrics: Dict[str, float] = {
        "best_iteration": booster.best_iteration,
        "best_score": booster.best_score,
        "task": task,
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "features": len(X.columns),
    }

    if task == "classification":
        from sklearn.metrics import accuracy_score, f1_score

        y_pred_prob = booster.predict(dval)
        if y.nunique() == 2:
            y_pred = (y_pred_prob > 0.5).astype(int)
        else:
            y_pred = np.argmax(y_pred_prob, axis=1)

        metrics["accuracy"] = float(accuracy_score(y_val, y_pred))
        metrics["f1_score"] = float(f1_score(y_val, y_pred, average="weighted"))
    else:
        from sklearn.metrics import mean_squared_error, r2_score

        y_pred = booster.predict(dval)
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        metrics["r2"] = float(r2_score(y_val, y_pred))

    logger.info(f"[xgb_model] Training complete — metrics: {metrics}")
    return booster, metrics


# ============================================================
#  PREDICT
# ============================================================

def predict(
    model: Any,
    data: Union[str, pd.DataFrame, np.ndarray],
) -> np.ndarray:
    """
    Run predictions using a trained XGBoost booster.

    Args:
        model:  A trained xgb.Booster.
        data:   CSV path, DataFrame, or numpy array.

    Returns:
        numpy array of predictions.
    """
    xgb = _get_xgb()

    if isinstance(data, str):
        data = pd.read_csv(data)
    if isinstance(data, pd.DataFrame):
        # Encode categoricals the same way as training
        cat_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            data = pd.get_dummies(data, columns=cat_cols, drop_first=True)
        data = xgb.DMatrix(data)
    elif isinstance(data, np.ndarray):
        data = xgb.DMatrix(data)

    return model.predict(data)


# ============================================================
#  SAVE / LOAD
# ============================================================

def save_model(model: Any, path: str) -> str:
    """Save an XGBoost booster to a JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    model.save_model(path)
    logger.info(f"[xgb_model] Model saved to {path}")
    return path


def load_model(path: str) -> Any:
    """Load an XGBoost booster from a JSON file."""
    xgb = _get_xgb()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    booster = xgb.Booster()
    booster.load_model(path)
    logger.info(f"[xgb_model] Model loaded from {path}")
    return booster


# ============================================================
#  FEATURE IMPORTANCE
# ============================================================

def feature_importance(model: Any, importance_type: str = "weight") -> Dict[str, float]:
    """
    Get feature importance scores from a trained booster.

    Args:
        model:            Trained xgb.Booster.
        importance_type:  'weight', 'gain', or 'cover'.

    Returns:
        dict mapping feature names to importance scores, sorted descending.
    """
    scores = model.get_score(importance_type=importance_type)
    sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    return sorted_scores
