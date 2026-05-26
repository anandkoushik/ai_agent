import logging
import re
from agent_system.core.state import WorkflowState
from agent_system.tools.registry import ToolRegistry

# Known error signatures and their auto-remediation strategies
_REMEDIATION_RULES = [
    {
        "pattern": re.compile(r"cuda out of memory|cuda oom", re.IGNORECASE),
        "description": "CUDA OOM detected",
        "adjustments": {"batch_size": 4, "epochs": 3}
    },
    {
        "pattern": re.compile(r"missing label|no label|labelimg", re.IGNORECASE),
        "description": "Missing labels in dataset",
        "adjustments": {"skip_missing": True}
    },
    {
        "pattern": re.compile(r"not enough memory|memoryerror", re.IGNORECASE),
        "description": "System memory error",
        "adjustments": {"batch_size": 2, "workers": 1}
    },
]


def _detect_error(result: dict) -> dict | None:
    """Check stderr for known error signatures and return the matching rule."""
    stderr = result.get("stderr", "")
    for rule in _REMEDIATION_RULES:
        if rule["pattern"].search(stderr):
            return rule
    return None


class ObservationLoop:
    def __init__(self, tool_registry: ToolRegistry, max_retries: int = 2):
        self.registry = tool_registry
        self.max_retries = max_retries

    async def observe_and_heal(
        self, tool_name: str, kwargs: dict, state: WorkflowState, step_idx: int
    ) -> dict:
        """
        Execute a tool and if it fails, attempt auto-remediation up to max_retries.
        Returns the final result dict (strict schema).
        """
        attempt = 0
        current_kwargs = dict(kwargs)

        while attempt <= self.max_retries:
            result = await self.registry.execute(tool_name, **current_kwargs)

            # Success path
            if result.get("status_code") == 0:
                return result

            # Failure path — inspect stderr
            attempt += 1
            rule = _detect_error(result)

            if rule and attempt <= self.max_retries:
                logging.warning(
                    f"[ObservationLoop] Step {step_idx} ({tool_name}) failed. "
                    f"Reason: {rule['description']}. "
                    f"Applying adjustments: {rule['adjustments']} — Retry {attempt}/{self.max_retries}"
                )
                # Merge adjustments into kwargs for next attempt
                current_kwargs.update(rule["adjustments"])
                # Record the retry in state history
                state.add_history(step_idx, f"{tool_name}_retry_{attempt}", {
                    "info": f"Auto-healing: {rule['description']}",
                    "applied_adjustments": rule["adjustments"]
                })
            else:
                # Unrecognised error or retries exhausted
                logging.error(
                    f"[ObservationLoop] Step {step_idx} ({tool_name}) failed permanently "
                    f"after {attempt - 1} retries."
                )
                return result

        return result
