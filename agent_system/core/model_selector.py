import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ModelSelector:
    """Hardware-aware model selection logic."""
    
    @staticmethod
    def select_model(dataset_manifest: Dict[str, Any], vram_gb: float = 8.0) -> str:
        """
        Provider-Agnostic Model Registry.
        Selects model family and size based on telemetry and dataset.
        """
        modality = dataset_manifest.get("dataset_type", "unknown")
        
        logger.info(f"[ModelSelector] Selecting model for modality: {modality} with {vram_gb}GB VRAM")
        
        if modality == "yolo":
            if vram_gb < 4.0:
                return "yolov8n"
            elif vram_gb < 10.0:
                return "yolov8m"
            else:
                return "yolov8l"
                
        elif modality == "whisper":
            if vram_gb < 4.0:
                return "whisper-tiny"
            elif vram_gb < 8.0:
                return "whisper-small"
            else:
                return "whisper-large-v3"
                
        elif modality == "llm":
            if vram_gb < 8.0:
                return "phi-2" # local fallback
            else:
                return "mistral-7b"
                
        else:
            return "unknown-model"
