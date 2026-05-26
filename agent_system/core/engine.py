from .state import WorkflowState
from .observation import ObservationLoop
from agent_system.tools.registry import ToolRegistry
import logging

class WorkflowEngine:
    def __init__(self, tool_registry: ToolRegistry, max_retries: int = 2):
        self.registry = tool_registry
        self.observation_loop = ObservationLoop(tool_registry, max_retries=max_retries)

    async def execute_workflow(self, job_id: str, steps: list[dict]) -> WorkflowState:
        """
        Executes a series of tools sequentially from the provided plan.
        Steps format: [{"tool_name": "train_whisper", "kwargs": {"epochs": 10}}]
        Failed steps are handed to ObservationLoop for auto-remediation before aborting.
        """
        state = WorkflowState(job_id=job_id)
        state.mark_running()

        for step_idx, step in enumerate(steps, start=1):
            state.current_step = step_idx
            tool_name = step.get("tool_name")
            kwargs = step.get("kwargs", {})

            # Delegate execution to the observation loop (handles retries automatically)
            result = await self.observation_loop.observe_and_heal(
                tool_name=tool_name,
                kwargs=kwargs,
                state=state,
                step_idx=step_idx
            )

            # Record final outcome
            state.add_history(step_idx, tool_name, result)

            if result.get("status_code") != 0:
                logging.error(f"Workflow {job_id} permanently failed at step {step_idx} ({tool_name}).")
                state.mark_failed()
                return state

        state.mark_success()
        return state

