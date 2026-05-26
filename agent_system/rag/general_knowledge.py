import logging
from typing import List, Dict, Any
import hashlib
import time

logger = logging.getLogger(__name__)

class KnowledgeCache:
    """Versioned cache for educational snippets to prevent stale documentation."""
    def __init__(self):
        self.version = "1.0.0"  # Increment to invalidate stale caches
        self.cache = {}
        
    def get(self, query: str) -> Dict[str, Any]:
        key = hashlib.md5(f"{self.version}_{query}".encode()).hexdigest()
        return self.cache.get(key)
        
    def set(self, query: str, data: Dict[str, Any]):
        key = hashlib.md5(f"{self.version}_{query}".encode()).hexdigest()
        self.cache[key] = {
            "data": data,
            "timestamp": time.time()
        }

# Global cache instance
_cache = KnowledgeCache()

class LocalKnowledgeProvider:
    """
    Swappable interface seeded with curated ML documentation.
    Ranks official documentation higher than community content.
    """
    
    # Mock seeded database of ML concepts
    SEED_DATA = {
        "overfitting": [
            {"source": "PyTorch Official Docs", "text": "Overfitting occurs when a model learns the detail and noise in the training data to the extent that it negatively impacts the performance of the model on new data.", "type": "official"},
            {"source": "ML Community Forum", "text": "Just add dropout and more data if it overfits.", "type": "community"}
        ],
        "batch size": [
            {"source": "Deep Learning Book", "text": "Larger batch sizes provide a more accurate estimate of the gradient, but with less than linear returns. Smaller batch sizes offer a regularizing effect.", "type": "official"},
            {"source": "Reddit r/MachineLearning", "text": "I usually just use 32 because it fits in my VRAM.", "type": "community"}
        ],
        "epoch": [
            {"source": "PyTorch Official Docs", "text": "An epoch in machine learning means one complete pass of the entire training dataset through the model during training. Multiple epochs are used because models gradually improve by repeatedly adjusting weights after seeing the data multiple times.", "type": "official"},
            {"source": "ML Community Forum", "text": "Too few epochs can cause underfitting. Too many epochs can cause overfitting.", "type": "community"}
        ],
        "yolo vs cnn": [
            {"source": "Ultralytics Docs", "text": "YOLO (You Only Look Once) is a single-stage object detector that predicts bounding boxes and class probabilities simultaneously, making it much faster than two-stage CNNs like Faster R-CNN.", "type": "official"}
        ]
    }
    
    @classmethod
    def retrieve(cls, query: str) -> List[Dict[str, Any]]:
        """Simulates retrieving semantically relevant snippets."""
        query_lower = query.lower()
        results = []
        for key, snippets in cls.SEED_DATA.items():
            if key in query_lower:
                results.extend(snippets)
        return results

class GeneralKnowledgeRAG:
    """Web-Augmented Knowledge Retrieval (using Local Provider currently)"""
    
    @staticmethod
    def _validate_relevance(query: str, snippet: str) -> bool:
        """Retrieval Safety Validation: Ensures snippet is actually relevant."""
        # Simple heuristic: at least one significant word from query must be in snippet
        query_words = set(query.lower().split()) - {"what", "is", "how", "why", "does", "the", "a", "an"}
        if not query_words:
            return True
        return any(w in snippet.lower() for w in query_words)
        
    @staticmethod
    def _score_confidence(snippet: Dict[str, Any]) -> float:
        """Source Confidence Scoring: prioritizes official docs."""
        if snippet["type"] == "official":
            return 0.95
        elif snippet["type"] == "community":
            return 0.60
        return 0.50

    @classmethod
    def query_concept(cls, query: str) -> Dict[str, Any]:
        """Fetches, scores, validates, and synthesizes ML knowledge."""
        cached = _cache.get(query)
        if cached:
            logger.info(f"Cache hit for query: {query}")
            return cached["data"]
            
        logger.info(f"Querying knowledge provider for: {query}")
        raw_snippets = LocalKnowledgeProvider.retrieve(query)
        
        valid_snippets = []
        for s in raw_snippets:
            if cls._validate_relevance(query, s["text"]):
                score = cls._score_confidence(s)
                valid_snippets.append({"score": score, **s})
                
        # Sort by confidence score descending
        valid_snippets.sort(key=lambda x: x["score"], reverse=True)
        
        result = {
            "query": query,
            "snippets": valid_snippets,
            "synthesized": bool(valid_snippets)
        }
        
        _cache.set(query, result)
        return result
