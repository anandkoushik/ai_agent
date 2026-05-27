"""
antigravity.port — Unified ML Training & Inference Port

Provides a single, clean entry point for all supported model families:
  • YOLO   (object detection)
  • Whisper (speech-to-text)
  • LLM    (text generation / fine-tuning)
  • Vision (image classification)

Usage:
    from antigravity import port

    # Training
    result = port.train(model_type="yolo", dataset_path="./data", epochs=50)

    # Inference
    output = port.infer(model_type="whisper", model_path="./models/whisper", input_file="audio.wav")

    # Health check
    status = port.health_check()
"""

import os
import sys
import logging
import importlib
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — keeps startup fast, only loads heavy ML libs when needed
# ---------------------------------------------------------------------------

_training_module = None
_inference_module = None


def _get_training_module():
    """Lazy-load training.py from project root."""
    global _training_module
    if _training_module is None:
        try:
            _training_module = importlib.import_module("training")
            logger.info("[port] training module loaded successfully")
        except ImportError as e:
            logger.error(f"[port] Failed to import training module: {e}")
            raise ImportError(
                "training.py not found. Ensure it exists in the project root."
            ) from e
    return _training_module


def _get_inference_module():
    """Lazy-load inference.py from project root."""
    global _inference_module
    if _inference_module is None:
        try:
            _inference_module = importlib.import_module("inference")
            logger.info("[port] inference module loaded successfully")
        except ImportError as e:
            logger.error(f"[port] Failed to import inference module: {e}")
            raise ImportError(
                "inference.py not found. Ensure it exists in the project root."
            ) from e
    return _inference_module


# ============================================================
#  SUPPORTED MODELS
# ============================================================

SUPPORTED_TRAINING_TYPES = ("yolo", "whisper", "llm", "vision")
SUPPORTED_INFERENCE_TYPES = ("yolo", "whisper", "llm", "vision")


# ============================================================
#  TRAIN — Unified training entry point
# ============================================================

def train(
    model_type: str,
    dataset_path: str,
    epochs: int = 10,
    workspace_dir: Optional[str] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Train a model of the given type on the provided dataset.

    Args:
        model_type:    One of 'yolo', 'whisper', 'llm', 'vision'.
        dataset_path:  Absolute path to the dataset directory or ZIP.
        epochs:        Number of training epochs (auto-tuned if omitted).
        workspace_dir: Working directory for model output (defaults to dataset_path).
        batch_size:    Override auto-selected batch size.
        learning_rate: Override auto-selected learning rate.
        **kwargs:      Extra arguments forwarded to the underlying trainer.

    Returns:
        dict with keys: status, model_path, metrics (when available).
    """
    model_type = model_type.lower().strip()

    if model_type not in SUPPORTED_TRAINING_TYPES:
        return {
            "status": "error",
            "message": f"Unsupported model type '{model_type}'. "
                       f"Supported: {', '.join(SUPPORTED_TRAINING_TYPES)}",
        }

    if not os.path.exists(dataset_path):
        return {
            "status": "error",
            "message": f"Dataset path does not exist: {dataset_path}",
        }

    workspace_dir = workspace_dir or dataset_path
    training = _get_training_module()

    try:
        if model_type == "yolo":
            result = training.train_yolo(
                data_path=dataset_path,
                epochs=epochs,
                workspace_dir=workspace_dir,
                **kwargs,
            )
        elif model_type == "whisper":
            result = training.train_whisper(
                data_path=dataset_path,
                epochs=epochs,
                workspace_dir=workspace_dir,
                model_type=kwargs.get("whisper_size", "whisper-small"),
                batch_size=batch_size,
                learning_rate=learning_rate,
            )
        elif model_type == "llm":
            result = training.train_llm(
                data_path=dataset_path,
                epochs=epochs,
                workspace_dir=workspace_dir,
                model_type=kwargs.get("llm_model", "tinyllama"),
            )
        elif model_type == "vision":
            result = training.train_vision_classifier(
                data_path=dataset_path,
                epochs=epochs,
                workspace_dir=workspace_dir,
                model_type=kwargs.get("vision_model", "resnet18"),
            )
        else:
            return {"status": "error", "message": "Unknown model type"}

        return {
            "status": "success",
            "model_type": model_type,
            "model_path": result if isinstance(result, str) else str(result),
            "epochs": epochs,
        }

    except Exception as e:
        logger.exception(f"[port.train] Training failed for {model_type}")
        return {
            "status": "error",
            "model_type": model_type,
            "message": str(e),
        }


# ============================================================
#  INFER — Unified inference entry point
# ============================================================

def infer(
    model_type: str,
    model_path: str,
    input_file: str,
    workspace_dir: Optional[str] = None,
    workspace_name: str = "default",
    **kwargs,
) -> Dict[str, Any]:
    """
    Run inference using a trained model.

    Args:
        model_type:     One of 'yolo', 'whisper', 'llm', 'vision'.
        model_path:     Path to the trained model weights / directory.
        input_file:     Path to the input file (image, audio, text).
        workspace_dir:  Working directory for output artifacts.
        workspace_name: Human-friendly workspace label.
        **kwargs:       Extra arguments forwarded to the inference function.

    Returns:
        dict with keys: status, prediction / transcript / response.
    """
    model_type = model_type.lower().strip()

    if model_type not in SUPPORTED_INFERENCE_TYPES:
        return {
            "status": "error",
            "message": f"Unsupported inference type '{model_type}'. "
                       f"Supported: {', '.join(SUPPORTED_INFERENCE_TYPES)}",
        }

    if not os.path.exists(model_path):
        return {
            "status": "error",
            "message": f"Model path does not exist: {model_path}",
        }

    if not os.path.exists(input_file):
        return {
            "status": "error",
            "message": f"Input file does not exist: {input_file}",
        }

    inference_mod = _get_inference_module()
    workspace_dir = workspace_dir or os.path.dirname(model_path)

    try:
        if model_type == "yolo":
            # YOLO inference via ultralytics
            from ultralytics import YOLO
            model = YOLO(model_path)
            results = model.predict(source=input_file, save=True, **kwargs)
            detections = []
            for r in results:
                for box in r.boxes:
                    detections.append({
                        "class": r.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "bbox": box.xyxy[0].tolist(),
                    })
            return {
                "status": "success",
                "model_type": "yolo",
                "detections": detections,
                "num_objects": len(detections),
            }

        elif model_type == "whisper":
            import torch
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            import librosa

            processor = WhisperProcessor.from_pretrained(model_path)
            model = WhisperForConditionalGeneration.from_pretrained(model_path)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)

            speech, sr = librosa.load(input_file, sr=16000)
            input_features = processor.feature_extractor(
                speech, sampling_rate=16000
            ).input_features
            input_features = torch.tensor(input_features).to(device)

            with torch.no_grad():
                predicted_ids = model.generate(input_features)
            transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            return {
                "status": "success",
                "model_type": "whisper",
                "transcript": transcript,
            }

        elif model_type == "llm":
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)

            # Read the input text
            with open(input_file, "r", encoding="utf-8") as f:
                prompt_text = f.read().strip()

            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            max_new = kwargs.get("max_new_tokens", 256)

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=max_new)
            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            return {
                "status": "success",
                "model_type": "llm",
                "response": response,
            }

        elif model_type == "vision":
            import torch
            from torchvision import transforms
            from PIL import Image
            import json

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = torch.load(
                os.path.join(model_path, "model.pth"), map_location=device
            )
            model.eval()

            labels_path = os.path.join(model_path, "labels.json")
            labels = {}
            if os.path.exists(labels_path):
                with open(labels_path, "r") as f:
                    labels = json.load(f)

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            img = Image.open(input_file).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(tensor)
                _, predicted = torch.max(output, 1)

            class_idx = predicted.item()
            class_name = labels.get(str(class_idx), f"class_{class_idx}")

            return {
                "status": "success",
                "model_type": "vision",
                "prediction": class_name,
                "class_index": class_idx,
            }

        else:
            return {"status": "error", "message": "Unknown model type"}

    except Exception as e:
        logger.exception(f"[port.infer] Inference failed for {model_type}")
        return {
            "status": "error",
            "model_type": model_type,
            "message": str(e),
        }


# ============================================================
#  EXPORT — Package trained models into downloadable ZIPs
# ============================================================

def export_model(
    model_type: str,
    model_name: str,
    workspace_dir: str,
    workspace_name: str = "default",
) -> Optional[str]:
    """
    Export a trained model into a deployment-ready ZIP.

    Returns the absolute path to the ZIP file, or None on failure.
    """
    inference_mod = _get_inference_module()
    model_type = model_type.lower().strip()

    try:
        if model_type == "yolo":
            return inference_mod.make_model_zip(
                ws_dir=workspace_dir,
                model_type="yolo",
                model_name=model_name,
                workspace_name=workspace_name,
            )
        elif model_type == "whisper":
            return inference_mod.make_whisper_inference_zip(
                ws_dir=workspace_dir,
                model_name=model_name,
                workspace_name=workspace_name,
            )
        elif model_type == "llm":
            return inference_mod.make_llm_inference_zip(
                ws_dir=workspace_dir,
                model_name=model_name,
                workspace_name=workspace_name,
            )
        elif model_type == "vision":
            return inference_mod.make_vision_inference_zip(
                ws_dir=workspace_dir,
                model_name=model_name,
                workspace_name=workspace_name,
            )
        else:
            logger.error(f"[port.export_model] Unsupported type: {model_type}")
            return None
    except Exception as e:
        logger.exception(f"[port.export_model] Failed for {model_type}/{model_name}")
        return None


# ============================================================
#  HEALTH CHECK
# ============================================================

def health_check() -> Dict[str, Any]:
    """
    Verify that all critical ML dependencies are importable and a GPU is
    reachable (if CUDA is expected).

    Returns a dict summarizing system readiness.
    """
    status: Dict[str, Any] = {
        "training_module": False,
        "inference_module": False,
        "torch": False,
        "cuda_available": False,
        "gpu_name": None,
        "ultralytics": False,
        "transformers": False,
        "litellm": False,
    }

    # Core modules
    try:
        _get_training_module()
        status["training_module"] = True
    except ImportError:
        pass

    try:
        _get_inference_module()
        status["inference_module"] = True
    except ImportError:
        pass

    # PyTorch + CUDA
    try:
        import torch
        status["torch"] = True
        status["cuda_available"] = torch.cuda.is_available()
        if status["cuda_available"]:
            status["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    # Ultralytics (YOLO)
    try:
        import ultralytics
        status["ultralytics"] = True
    except ImportError:
        pass

    # Transformers (Whisper / LLM)
    try:
        import transformers
        status["transformers"] = True
    except ImportError:
        pass

    # LiteLLM (Groq cloud inference)
    try:
        import litellm
        status["litellm"] = True
    except ImportError:
        pass

    all_ok = all([
        status["training_module"],
        status["inference_module"],
        status["torch"],
        status["ultralytics"],
        status["transformers"],
    ])

    status["ready"] = all_ok

    return status
