import logging
import os
import json
from typing import Tuple, Dict, List, Any
from agent_system.config import Config
try:
    import litellm
except ImportError:
    litellm = None

logger = logging.getLogger(__name__)

class HybridRouter:
    """Intelligently routes intents between conversation, RAG, and workflow execution using an LLM."""
    
    @staticmethod
    def classify_intent(user_prompt: str, files: list = None) -> Dict[str, Any]:
        """
        Returns a dict:
        {
            "primary_intent": str,
            "intents": List[str],
            "confidence": float,
            "cleaned_prompt": str
        }
        """
        logger.info(f"Classifying intent for prompt using LLM: {user_prompt[:50]}...")
        
        has_files = files is not None and len(files) > 0
        prompt_lower = user_prompt.lower().strip()
        
        # Hardcode bypass for UI buttons so they never fail
        if prompt_lower in ["train", "run inference", "execute workflow"]:
            return {
                "primary_intent": "workflow_execution",
                "intents": ["workflow_execution"],
                "confidence": 1.0,
                "cleaned_prompt": user_prompt
            }
        
        # Fallback to heuristic if litellm is not installed or API key is missing
        if litellm:
            try:
                model = Config.LLM_MODEL
                
                system_prompt = f"""You are an advanced intent classification system for an AI Machine Learning Engineering Assistant.
The user is interacting with you. They may have uploaded files: {has_files}.

PRIORITIZE classification in this exact order:
1. 'workflow_execution': The user wants to train, infer, or analyze a dataset/model (e.g., 'train this dataset', 'run detection').
2. 'troubleshooting': The user is reporting an error, crash, or bug.
3. 'hyperparameter_guidance': The user is asking for advice on epochs, batch size, or which model to use.
4. 'concept_explanation': The user is asking for a definition or explanation of an ML concept (e.g., 'what is a CNN', 'explain transformers'). Use this for general ML knowledge, NOT RAG.
5. 'rag_document_query': The user is explicitly asking to search external documentation or wants highly specific, recent factual information that requires web search/retrieval.
6. 'conversation': The user is saying hi, thanks, making small talk, or general conversational remarks.

Output a valid JSON object EXACTLY in this format:
{{
    "primary_intent": "the most prominent intent",
    "intents": ["list", "of", "all", "detected", "intents"],
    "confidence": 0.95
}}

CRITICAL:
- If the user says "hi", "hello", "how are you", "tell me a joke", the primary_intent MUST BE "conversation". Do NOT classify small talk as "rag_document_query".
- Do NOT use "rag_document_query" for general educational questions like "What is an epoch". Use "concept_explanation" instead.
- If the user provides a dataset and says "train it", the primary_intent MUST BE "workflow_execution".
"""
                response = litellm.completion(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={ "type": "json_object" },
                    temperature=0.0
                )
                
                result_str = response.choices[0].message.content
                result = json.loads(result_str)
                confidence = float(result.get("confidence", 0.9))
                primary = result.get("primary_intent", "conversation")
                
                # Confidence-Aware Routing: Fallback to simple conversation if unsure
                if confidence < 0.65 and primary not in ["workflow_execution", "troubleshooting"]:
                    logger.info(f"Low confidence ({confidence}) for {primary}. Falling back to conversation.")
                    primary = "conversation"
                    
                return {
                    "primary_intent": primary,
                    "intents": result.get("intents", ["conversation"]),
                    "confidence": confidence,
                    "cleaned_prompt": user_prompt
                }
            except Exception as e:
                logger.error(f"LLM Routing failed, falling back to heuristic: {str(e)}")
        
        # Fallback heuristic logic
        prompt_lower = user_prompt.lower()
        
        scores = {
            "workflow_execution": 0.0,
            "concept_explanation": 0.0,
            "hyperparameter_guidance": 0.0,
            "troubleshooting": 0.0,
            "rag_document_query": 0.0,
            "conversation": 0.1 # base fallback score
        }
        
        if has_files:
            if "train" in prompt_lower or "detect" in prompt_lower:
                scores["workflow_execution"] += 0.9
            if "summarize" in prompt_lower or "document" in prompt_lower:
                scores["rag_document_query"] += 0.9
                
        if "what is" in prompt_lower or "explain" in prompt_lower or "definition" in prompt_lower:
            scores["concept_explanation"] += 0.8
            
        if "how many" in prompt_lower or "recommend" in prompt_lower or "best" in prompt_lower or "should i use" in prompt_lower:
            scores["hyperparameter_guidance"] += 0.8
            
        if "error" in prompt_lower or "failed" in prompt_lower or "crash" in prompt_lower:
            scores["troubleshooting"] += 0.8
            
        active_intents = [k for k, v in scores.items() if v >= 0.7]
        
        if not active_intents:
            return {
                "primary_intent": "conversation",
                "intents": ["conversation"],
                "confidence": 1.0,
                "cleaned_prompt": user_prompt
            }
            
        active_intents.sort(key=lambda x: scores[x], reverse=True)
        primary_intent = "multi_intent" if len(active_intents) > 1 else active_intents[0]
        
        return {
            "primary_intent": primary_intent,
            "intents": active_intents,
            "confidence": scores[active_intents[0]],
            "cleaned_prompt": user_prompt
        }
