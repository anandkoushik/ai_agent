import logging
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import or_
from agent_system.db.models import Job, Workflow
from agent_system.config import Config
import psutil

logger = logging.getLogger(__name__)

class ResourceMonitor:
    @staticmethod
    def get_available_vram_gb() -> float:
        """
        Mock implementation for VRAM checking.
        In a real production system, this would use pynvml or torch.cuda to check actual free memory.
        """
        # For structural purposes, we assume 8GB free unless overridden
        return 8.0

    @staticmethod
    def get_available_ram_gb() -> float:
        return psutil.virtual_memory().available / (1024 ** 3)

class JobQueue:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_next_job(self) -> Optional[Job]:
        """
        Pulls the next 'queued' job that has its dependencies met.
        For Phase 5, a job is ready if its workflow is running.
        """
        # Find a job that is pending
        result = await self.session.execute(
            select(Job)
            .join(Workflow, Job.workflow_id == Workflow.workflow_id)
            .filter(Job.job_status == "pending")
            .filter(Workflow.status == "running")
            .order_by(Job.created_at.asc())
        )
        job = result.scalars().first()
        return job

    async def check_resources(self, job: Job) -> bool:
        """
        Resource-Aware Scheduler: blocks job dispatch if resources are too low.
        """
        # E.g. fetch required VRAM from job_metadata
        required_vram = job.job_metadata.get("required_vram_gb", 0)
        
        if required_vram > 0:
            free_vram = ResourceMonitor.get_available_vram_gb()
            if free_vram < required_vram:
                logger.warning(f"Job {job.id} blocked: Requires {required_vram}GB VRAM, only {free_vram}GB available.")
                return False
                
        return True

    async def dispatch(self, job: Job) -> bool:
        """
        Marks job as running if resources allow.
        """
        if not await self.check_resources(job):
            return False
            
        job.job_status = "running"
        async with self.session.begin_nested():
            self.session.add(job)
        await self.session.flush()
        return True

    async def complete_job(self, job: Job, success: bool, error_msg: str = None):
        job.job_status = "completed" if success else "failed"
        if error_msg:
            job.error_message = error_msg
            
        async with self.session.begin_nested():
            self.session.add(job)
        await self.session.flush()
