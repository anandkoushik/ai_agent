import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from agent_system.db.models import Workflow, Job
from agent_system.core.job_queue import JobQueue
from agent_system.config import Config
import asyncio

logger = logging.getLogger(__name__)

class WorkflowEngine:
    def __init__(self, session: AsyncSession, registry):
        self.session = session
        self.registry = registry
        self.queue = JobQueue(session)

    async def create_workflow(self, workflow_id: str, dag_definition: Dict[str, Any]) -> Workflow:
        """
        Registers a new DAG workflow.
        dag_definition example:
        {
            "job1": {"tool": "yolo_train", "deps": []},
            "job2": {"tool": "evaluate", "deps": ["job1"]}
        }
        """
        wf = Workflow(
            workflow_id=workflow_id,
            status="running",
            dag_state=dag_definition
        )
        
        async with self.session.begin_nested():
            self.session.add(wf)
            await self.session.flush()
            
            for job_name, job_data in dag_definition.items():
                job = Job(
                    workflow_id=workflow_id,
                    job_type=job_data.get("tool"),
                    job_metadata={"name": job_name, "deps": job_data.get("deps", [])}
                )
                self.session.add(job)
                
        await self.session.flush()
        return wf

    async def recover_active_workflows(self):
        """
        CRASH RECOVERY: Reloads paused or running workflows after a server restart.
        """
        result = await self.session.execute(
            select(Workflow).filter(Workflow.status.in_(["running", "paused"]))
        )
        active_wfs = result.scalars().all()
        logger.info(f"Recovered {len(active_wfs)} active workflows from PostgreSQL.")
        
        for wf in active_wfs:
            # We would spawn background async tasks for each active workflow here
            pass

    async def _execute_job(self, job: Job):
        """
        Executes a single job with retry checkpoints.
        """
        from agent_system.tools.gpu_manager import GPUSafetyManager
        gpu_manager = GPUSafetyManager()
        
        retries = job.job_metadata.get("retries_attempted", 0)
        
        while retries <= Config.MAX_RETRIES:
            try:
                # Wrap execution in GPU context manager
                async with gpu_manager.gpu_lock(job.id):
                    logger.info(f"Executing job {job.job_type} (Attempt {retries+1})")
                    
                    # Mock execution
                    await asyncio.sleep(1)
                
                await self.queue.complete_job(job, success=True)
                return True
                
            except Exception as e:
                retries += 1
                job.job_metadata["retries_attempted"] = retries
                async with self.session.begin_nested():
                    self.session.add(job)
                await self.session.flush()
                
                if retries > Config.MAX_RETRIES:
                    await self.queue.complete_job(job, success=False, error_msg=str(e))
                    return False
                logger.warning(f"Job failed, retrying... ({retries}/{Config.MAX_RETRIES})")

    async def run_workflow_loop(self, workflow: Workflow):
        """
        Resolves dependencies and runs the DAG.
        """
        while True:
            # Refresh workflow state
            result = await self.session.execute(select(Job).filter_by(workflow_id=workflow.workflow_id))
            jobs = result.scalars().all()
            
            pending = [j for j in jobs if j.job_status == "pending"]
            running = [j for j in jobs if j.job_status == "running"]
            failed = [j for j in jobs if j.job_status == "failed"]
            completed = [j for j in jobs if j.job_status == "completed"]
            
            if failed:
                workflow.status = "failed"
                async with self.session.begin_nested():
                    self.session.add(workflow)
                await self.session.flush()
                break
                
            if not pending and not running:
                workflow.status = "completed"
                async with self.session.begin_nested():
                    self.session.add(workflow)
                await self.session.flush()
                break
                
            completed_names = [j.job_metadata.get("name") for j in completed]
            
            # Find jobs whose dependencies are met
            for job in pending:
                deps = job.job_metadata.get("deps", [])
                if all(d in completed_names for d in deps):
                    if await self.queue.dispatch(job):
                        # Spawn background execution
                        asyncio.create_task(self._execute_job(job))
            
            await asyncio.sleep(2)
