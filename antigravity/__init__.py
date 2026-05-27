"""
antigravity — Unified ML Training, Inference & Intelligence Package

Modules:
    port       — YOLO / Whisper / LLM / Vision training & inference
    ml         — XGBoost tabular prediction pipeline
    llm        — Qwen local LLM inference client
    pipeline   — Unified intelligent_response() combining XGBoost + Qwen

Quick start:
    from antigravity import train, infer, health_check
    from antigravity import intelligent_response, predict, ask_qwen
"""

from . import port
from .pipeline import intelligent_response, train_and_explain
from .ml.xgb_model import predict, train_xgb, load_model, save_model, feature_importance
from .llm.qwen_client import ask_qwen, ask_qwen_stream, unload_model

# Re-export port-level functions
train = port.train
infer = port.infer
export_model = port.export_model
health_check = port.health_check

SUPPORTED_TRAINING_TYPES = port.SUPPORTED_TRAINING_TYPES
SUPPORTED_INFERENCE_TYPES = port.SUPPORTED_INFERENCE_TYPES

__all__ = [
    # port (deep learning)
    "port",
    "train",
    "infer",
    "export_model",
    "health_check",
    "SUPPORTED_TRAINING_TYPES",
    "SUPPORTED_INFERENCE_TYPES",
    # ml (XGBoost)
    "predict",
    "train_xgb",
    "load_model",
    "save_model",
    "feature_importance",
    # llm (Qwen)
    "ask_qwen",
    "ask_qwen_stream",
    "unload_model",
    # pipeline (unified)
    "intelligent_response",
    "train_and_explain",
]
