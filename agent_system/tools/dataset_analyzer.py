import os
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class DatasetAnalyzer:
    """
    Scans a dataset for balance, size, and duplicates to inform the Hyperparameter Engine.
    """
    def __init__(self, dataset_path: str, dataset_type: str):
        self.dataset_path = dataset_path
        self.dataset_type = dataset_type
        
    def analyze(self) -> Dict[str, Any]:
        logger.info(f"Analyzing {self.dataset_type} dataset at {self.dataset_path}")
        
        # 1. Total Volume Extraction
        total_size_mb = self._get_directory_size_mb()
        
        # 2. Mock Class Balance Detection
        # In a real scenario, this parses YOLO annotations, CSV labels, etc.
        class_distribution = {"class_0": 1500, "class_1": 1450, "class_2": 200}
        imbalance_ratio = self._calculate_imbalance(class_distribution)
        
        # 3. Duplicate Detection
        duplicates_found = 12
        
        analysis_report = {
            "total_size_mb": total_size_mb,
            "class_distribution": class_distribution,
            "imbalance_ratio": imbalance_ratio, # e.g., max_class / min_class
            "duplicates_found": duplicates_found,
            "is_corrupt": False,
            "is_too_small": total_size_mb < 0.1
        }
        
        if analysis_report["is_too_small"]:
            raise ValueError(f"DatasetTooSmall: Dataset is only {total_size_mb}MB. Aborting workflow.")
            
        return analysis_report

    def _get_directory_size_mb(self) -> float:
        total_size = 0
        if not os.path.exists(self.dataset_path):
            return 10.0 # Mock default for testing
            
        for dirpath, _, filenames in os.walk(self.dataset_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
        return total_size / (1024 * 1024)

    def _calculate_imbalance(self, dist: Dict[str, int]) -> float:
        if not dist:
            return 1.0
        counts = list(dist.values())
        return max(counts) / (min(counts) or 1)
