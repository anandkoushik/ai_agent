import os
from typing import Dict, Any, List
from agent_system.uploads.modality_router import ModalityRouter

class ManifestBuilder:
    """Compiles unified dataset metadata."""
    
    @staticmethod
    def build_manifest(workspace_dir: str) -> Dict[str, Any]:
        files = []
        total_size = 0
        
        for root, _, filenames in os.walk(workspace_dir):
            for f in filenames:
                path = os.path.join(root, f)
                files.append(path)
                total_size += os.path.getsize(path)
                
        modality = ModalityRouter.infer_modality(files)
        size_mb = total_size / (1024 * 1024)
        
        return {
            "dataset_type": modality,
            "num_files": len(files),
            "total_size_mb": round(size_mb, 2),
            "contains_labels": any("labels" in f or f.endswith(".csv") for f in files),
            "modalities": [modality] if modality != "unknown" else [],
            "recommended_model": "yolov8m" if modality == "yolo" else ("whisper-small" if modality == "whisper" else "gpt2")
        }
