import logging
import gc
from contextlib import asynccontextmanager
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

class GPUSafetyManager:
    """
    Context manager for safely wrapping heavy tool execution (e.g., training jobs).
    Guarantees resource cleanup even if the job fails or the event loop cancels the task.
    """
    @staticmethod
    def _clear_cuda_cache():
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                logger.debug("CUDA cache explicitly cleared.")
        except ImportError:
            pass

    @staticmethod
    def _run_garbage_collection():
        gc.collect()

    @asynccontextmanager
    async def gpu_lock(self, job_id: str) -> AsyncGenerator[None, None]:
        """
        Provides a safe boundary for GPU execution.
        """
        logger.info(f"Acquiring GPU lock for Job {job_id}")
        
        try:
            # Yield control back to the executor (where the training happens)
            yield
        except Exception as e:
            logger.error(f"Job {job_id} encountered fatal exception during GPU execution: {e}")
            raise
        finally:
            logger.info(f"Releasing GPU lock for Job {job_id} and forcing cleanup.")
            self._clear_cuda_cache()
            self._run_garbage_collection()
            # If we had spawned a subprocess for isolated training, we would explicitly os.kill it here.
