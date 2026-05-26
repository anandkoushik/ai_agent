from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class WorkflowState:
    job_id: str
    current_step: int = 1
    status: str = "PENDING"
    experiment_metrics: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def add_history(self, step_index: int, tool_name: str, result: dict):
        self.history.append({
            "step": step_index,
            "tool_name": tool_name,
            "result": result
        })

    def mark_failed(self):
        self.status = "FAILED"
        
    def mark_running(self):
        self.status = "RUNNING"
        
    def mark_success(self):
        self.status = "SUCCESS"
