import logging
from typing import Dict, Any
from agent_system.config import Config

logger = logging.getLogger(__name__)

class HyperparameterEngine:
    """
    Smart Auto-Selection Engine.
    Maps DatasetAnalyzer outputs strictly to the boundaries defined in Config.
    """
    
    @staticmethod
    def estimate_epochs(model_type: str, analysis_report: Dict[str, Any]) -> int:
        size_mb = analysis_report.get("total_size_mb", 0)
        imbalance_ratio = analysis_report.get("imbalance_ratio", 1.0)
        
        # Select appropriate strict dictionary from config
        if model_type == "yolo":
            constraints = Config.YOLO_EPOCHS
        elif model_type == "whisper":
            constraints = Config.WHISPER_EPOCHS
        elif model_type == "vision":
            constraints = Config.VISION_EPOCHS
        elif model_type == "llm":
            constraints = Config.LLM_EPOCHS
        else:
            raise ValueError(f"Unknown model_type for epoch estimation: {model_type}")
            
        # Smart Auto-Selection based on Size
        if size_mb < constraints["small_mb"]:
            base_epochs = constraints["small_range"][0] # e.g. 80
        elif size_mb < constraints["medium_mb"]:
            base_epochs = constraints["medium_range"][0] # e.g. 50
        else:
            base_epochs = constraints["large_range"][0] # e.g. 30
            
        # Dynamic Adjustment: High imbalance -> needs more epochs to learn minority classes
        if imbalance_ratio > 5.0:
            logger.info("High class imbalance detected. Dynamically increasing epochs.")
            base_epochs += 15
            
        # Strict enforcement of absolute bounds
        base_epochs = max(constraints["absolute_min"], base_epochs)
        base_epochs = min(constraints["absolute_max"], base_epochs)
        
        return base_epochs

    @staticmethod
    def generate_training_plan(model_type: str, analysis_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates the full hyperparameter payload for the DAG job_metadata.
        """
        epochs = HyperparameterEngine.estimate_epochs(model_type, analysis_report)
        imbalance = analysis_report.get("imbalance_ratio", 1.0)
        
        # If highly imbalanced, lower learning rate slightly to prevent catastrophic forgetting of minority classes
        lr = 0.001 if imbalance < 5.0 else 0.0005
        
        return {
            "model_type": model_type,
            "estimated_epochs": epochs,
            "learning_rate": lr,
            "batch_size": 16 if analysis_report.get("total_size_mb", 0) < 1000 else 32,
            "imbalance_handling": "focal_loss" if imbalance > 5.0 else "standard"
        }
