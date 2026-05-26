from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class ToolSchema:
    name: str
    description: str
    args_schema: Dict[str, Any]
    required_files: List[str] = field(default_factory=list)
    gpu_required: bool = False
    estimated_vram_gb: float = 0.0
    allowed_retries: int = 2
    category: str = "general"
    failure_patterns: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "args_schema": self.args_schema,
            "required_files": self.required_files,
            "gpu_required": self.gpu_required,
            "estimated_vram_gb": self.estimated_vram_gb,
            "allowed_retries": self.allowed_retries,
            "category": self.category,
            "failure_patterns": self.failure_patterns
        }

# Pre-defined tool schemas
WHISPER_SCHEMA = ToolSchema(
    name="train_whisper",
    description="Trains a Whisper audio transcription model.",
    args_schema={"epochs": "int", "batch_size": "int"},
    required_files=["dataset.zip", "transcript.csv"],
    gpu_required=True,
    estimated_vram_gb=4.0,
    allowed_retries=2,
    category="training",
    failure_patterns=["CUDA out of memory", "missing transcript"]
)

LLM_SCHEMA = ToolSchema(
    name="train_llm",
    description="Fine-tunes a Large Language Model on instruction data.",
    args_schema={"epochs": "int", "batch_size": "int", "learning_rate": "float"},
    required_files=["dataset.txt"],
    gpu_required=True,
    estimated_vram_gb=8.0,
    allowed_retries=2,
    category="training",
    failure_patterns=["CUDA out of memory", "OOM", "tensor size mismatch"]
)

MODEL_ZIP_SCHEMA = ToolSchema(
    name="make_model_zip",
    description="Packages trained model weights into a ZIP archive for deployment.",
    args_schema={"model_dir": "str", "output_zip": "str"},
    gpu_required=False,
    estimated_vram_gb=0.0,
    allowed_retries=1,
    category="inference"
)

ANALYZE_DATASET_SCHEMA = ToolSchema(
    name="analyze_dataset",
    description="Analyzes the uploaded dataset structure to determine model type and epochs.",
    args_schema={"workspace_dir": "str"},
    gpu_required=False,
    estimated_vram_gb=0.0,
    allowed_retries=1,
    category="dataset"
)

# A registry dictionary of schemas
TOOL_SCHEMAS = {
    "train_whisper": WHISPER_SCHEMA,
    "train_llm": LLM_SCHEMA,
    "make_model_zip": MODEL_ZIP_SCHEMA,
    "analyze_dataset": ANALYZE_DATASET_SCHEMA,
}
