"""
antigravity — Unified ML Training & Inference Package

Quick start:
    from antigravity import port

    # Check system readiness
    port.health_check()

    # Train a model
    port.train(model_type="yolo", dataset_path="./data", epochs=50)

    # Run inference
    port.infer(model_type="whisper", model_path="./models/whisper", input_file="audio.wav")

    # Export trained model as ZIP
    port.export_model(model_type="llm", model_name="tinyllama", workspace_dir="./ws")
"""

from . import port

# Re-export top-level functions for convenience:
#   from antigravity import train, infer, health_check
train = port.train
infer = port.infer
export_model = port.export_model
health_check = port.health_check

SUPPORTED_TRAINING_TYPES = port.SUPPORTED_TRAINING_TYPES
SUPPORTED_INFERENCE_TYPES = port.SUPPORTED_INFERENCE_TYPES

__all__ = [
    "port",
    "train",
    "infer",
    "export_model",
    "health_check",
    "SUPPORTED_TRAINING_TYPES",
    "SUPPORTED_INFERENCE_TYPES",
]
