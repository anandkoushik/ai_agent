import json
import logging
from typing import List

class PlanningEngine:
    def __init__(self, available_tools: List[str]):
        self.available_tools = available_tools
        self._init_llm()

    def _init_llm(self):
        # Placeholder for LLM initialization.
        # In a real setup, we might load AutoModelForCausalLM here 
        # or use a remote client (e.g. OpenAI/Anthropic SDK).
        pass

    def generate_plan(self, query: str) -> dict:
        """
        Takes a user query and returns a strict JSON execution plan.
        We simulate the LLM generation for testing purposes here,
        but in a real environment it would prompt the LLM.
        """
        system_prompt = f"""You are a strict Planning Engine.
You MUST output ONLY valid JSON in the following format:
{{"pipeline": [{{"tool_name": "...", "kwargs": {{}}}}]}}
Available tools: {', '.join(self.available_tools)}
Do NOT output any markdown blocks or conversational text.
"""
        
        # Here we mock the LLM parsing logic for safety and speed in isolated testing.
        # A real implementation would pass `system_prompt` and `query` to `model.generate()`.
        
        try:
            # Simulated naive routing based on keywords
            plan = {"pipeline": []}
            if "analyze" in query.lower() or "dataset" in query.lower():
                plan["pipeline"].append({"tool_name": "analyze_dataset", "kwargs": {"workspace_dir": "."}})
            if "whisper" in query.lower():
                plan["pipeline"].append({"tool_name": "train_whisper", "kwargs": {"epochs": 5}})
            if "llm" in query.lower():
                plan["pipeline"].append({"tool_name": "train_llm", "kwargs": {"epochs": 3}})
                
            # Fallback if no keywords hit
            if not plan["pipeline"]:
                plan["pipeline"].append({"tool_name": "analyze_dataset", "kwargs": {"workspace_dir": "."}})
                
            return plan
        except Exception as e:
            logging.error(f"Planning failed: {e}")
            return {"pipeline": []}
