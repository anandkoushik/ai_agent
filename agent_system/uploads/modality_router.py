import os
from typing import Dict, Any, List

class ModalityRouter:
    """Maps dataset structures to the correct model family."""
    
    @staticmethod
    def infer_modality(files: List[str]) -> str:
        # Detect YOLO
        if any('images' in p for p in files) or any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
            return "yolo"
            
        # Detect Whisper (Audio files)
        if any(f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')) for f in files):
            return "whisper"
            
        # Detect LLM (Text/JSON files)
        if any(f.lower().endswith(('.txt', '.csv', '.json', '.jsonl')) for f in files):
            return "llm"
            
        # Default to LLM if files are present but extension is generic
        if len(files) > 0:
            return "llm"
            
        return "unknown"
