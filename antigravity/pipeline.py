"""
antigravity.pipeline — Unified Intelligent Response Pipeline

Combines XGBoost structured-data predictions with Qwen LLM reasoning
into a single `intelligent_response()` call.

Usage:
    from antigravity.pipeline import intelligent_response

    # Pure LLM question
    result = intelligent_response("Explain batch normalization.")

    # Tabular prediction + LLM explanation
    result = intelligent_response(
        "Predict the output and explain why.",
        tabular_data="test.csv",
        model_path="trained_model.json",
        target_column="price",
    )
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def intelligent_response(
    query: str,
    tabular_data: Optional[Union[str, pd.DataFrame]] = None,
    model_path: Optional[str] = None,
    target_column: str = "label",
    qwen_model_id: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Unified AI pipeline that intelligently routes between:
      1. XGBoost prediction  (if tabular_data + model_path are provided)
      2. Qwen LLM reasoning  (always runs to answer or explain)

    Args:
        query:           Natural-language question from the user.
        tabular_data:    CSV path or DataFrame for XGBoost prediction (optional).
        model_path:      Path to a saved XGBoost model .json file (optional).
        target_column:   Target column name (used in context only).
        qwen_model_id:   Override Qwen model ID.
        max_new_tokens:  Max tokens for Qwen generation.
        temperature:     Sampling temperature for Qwen.
        history:         Optional conversation history.

    Returns:
        dict with keys:
          - query:        Original user question
          - prediction:   XGBoost output (if applicable, else None)
          - explanation:  Qwen's natural-language response
          - pipeline:     Which stages ran ('xgb+qwen' or 'qwen')
    """
    result: Dict[str, Any] = {
        "query": query,
        "prediction": None,
        "explanation": None,
        "pipeline": "qwen",
    }

    xgb_context = ""

    # ------------------------------------------------------------------
    # Stage 1: XGBoost Prediction (optional)
    # ------------------------------------------------------------------
    if tabular_data is not None and model_path is not None:
        try:
            from antigravity.ml.xgb_model import predict, load_model

            booster = load_model(model_path)

            if isinstance(tabular_data, str):
                df = pd.read_csv(tabular_data)
            else:
                df = tabular_data.copy()

            predictions = predict(booster, df)
            result["prediction"] = predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
            result["pipeline"] = "xgb+qwen"

            # Build context for Qwen
            n_rows = len(df)
            sample_preds = predictions[:5].tolist() if len(predictions) > 5 else predictions.tolist()
            xgb_context = (
                f"\n\n[XGBoost Prediction Context]\n"
                f"The XGBoost model was run on {n_rows} rows of tabular data.\n"
                f"Target column: '{target_column}'\n"
                f"Sample predictions (first 5): {sample_preds}\n"
                f"Prediction stats: min={float(predictions.min()):.4f}, "
                f"max={float(predictions.max()):.4f}, "
                f"mean={float(predictions.mean()):.4f}\n"
            )

            logger.info(f"[pipeline] XGBoost prediction complete — {n_rows} rows")

        except Exception as e:
            logger.error(f"[pipeline] XGBoost stage failed: {e}")
            xgb_context = f"\n\n[XGBoost Error: {str(e)}]\n"

    # ------------------------------------------------------------------
    # Stage 2: Qwen LLM Reasoning (always runs)
    # ------------------------------------------------------------------
    try:
        from antigravity.llm.qwen_client import ask_qwen

        system_prompt = (
            "You are Antigravity AI, an expert ML engineering assistant. "
            "You specialize in AI, Machine Learning, Deep Learning, and Python. "
            "When XGBoost prediction context is provided, analyze and explain the results clearly. "
            "Be concise, technical, and avoid filler."
        )

        enriched_query = query
        if xgb_context:
            enriched_query = query + xgb_context

        explanation = ask_qwen(
            prompt=enriched_query,
            system_prompt=system_prompt,
            model_id=qwen_model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            history=history,
        )

        result["explanation"] = explanation
        logger.info("[pipeline] Qwen reasoning complete")

    except Exception as e:
        logger.error(f"[pipeline] Qwen stage failed: {e}")
        result["explanation"] = f"LLM reasoning unavailable: {str(e)}"

    return result


# ============================================================
#  TRAIN + EXPLAIN — End-to-end XGBoost with LLM summary
# ============================================================

def train_and_explain(
    data_source: Union[str, pd.DataFrame],
    target_column: str = "label",
    save_path: Optional[str] = None,
    qwen_model_id: Optional[str] = None,
    **train_kwargs,
) -> Dict[str, Any]:
    """
    Train an XGBoost model and generate a Qwen-powered summary of the results.

    Args:
        data_source:    CSV path or DataFrame.
        target_column:  Target column name.
        save_path:      Where to save the trained model (optional).
        qwen_model_id:  Override Qwen model ID.
        **train_kwargs: Extra args forwarded to train_xgb().

    Returns:
        dict with keys: model, metrics, summary, model_path.
    """
    from antigravity.ml.xgb_model import train_xgb, save_model

    # Train
    booster, metrics = train_xgb(
        data_source=data_source,
        target_column=target_column,
        **train_kwargs,
    )

    # Save
    model_path = None
    if save_path:
        model_path = save_model(booster, save_path)

    # Explain with Qwen
    summary = None
    try:
        from antigravity.llm.qwen_client import ask_qwen

        metrics_str = "\n".join(f"  • {k}: {v}" for k, v in metrics.items())
        explain_prompt = (
            f"I just trained an XGBoost model on a tabular dataset.\n\n"
            f"Training metrics:\n{metrics_str}\n\n"
            f"Give me a concise analysis: Is this model performing well? "
            f"What do these metrics indicate? Any suggestions for improvement?"
        )

        summary = ask_qwen(
            prompt=explain_prompt,
            model_id=qwen_model_id,
            max_new_tokens=400,
            temperature=0.5,
        )
    except Exception as e:
        summary = f"LLM summary unavailable: {str(e)}"

    return {
        "model": booster,
        "metrics": metrics,
        "summary": summary,
        "model_path": model_path,
    }
