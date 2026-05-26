import asyncio
import logging
import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from agent_system.db.session import AsyncSessionLocal

logger = logging.getLogger(__name__)

class DatabaseCleaner:
    def __init__(self, batch_size: int = 50, sleep_interval: float = 1.0):
        """
        batch_size: small number to avoid table locks during deletes
        sleep_interval: time to yield to the event loop between batches
        """
        self.batch_size = batch_size
        self.sleep_interval = sleep_interval

    async def _incremental_delete(self, session: AsyncSession, table: str, condition: str, params: dict):
        """
        Deletes rows incrementally to avoid blocking active inference/training workloads.
        """
        while True:
            # We use a CTE or nested subquery to delete in small chunks
            query = f"""
            DELETE FROM {table}
            WHERE id IN (
                SELECT id FROM {table}
                WHERE {condition}
                LIMIT :batch_size
            )
            RETURNING id;
            """
            params["batch_size"] = self.batch_size
            
            async with session.begin_nested():
                result = await session.execute(text(query), params)
                deleted_rows = result.fetchall()
                
            await session.flush()
            
            if not deleted_rows:
                break
                
            logger.info(f"Incremental cleanup: deleted {len(deleted_rows)} rows from {table}")
            # Yield event loop so high-priority workflow/GPU tasks can execute
            await asyncio.sleep(self.sleep_interval)

    async def clean_expired_telemetry(self, days_old: int = 30):
        async with AsyncSessionLocal() as session:
            try:
                cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=days_old)
                await self._incremental_delete(
                    session, 
                    "telemetry_logs", 
                    "created_at < :cutoff", 
                    {"cutoff": cutoff}
                )
                await session.commit()
            except Exception as e:
                logger.error(f"Telemetry cleanup failed: {e}")
                await session.rollback()

    async def clean_stale_workflows(self, days_old: int = 7):
        """Cleans up workflows that failed or paused a long time ago and were abandoned."""
        async with AsyncSessionLocal() as session:
            try:
                cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=days_old)
                # In PostgreSQL, cascading deletes on workflows will delete associated jobs
                await self._incremental_delete(
                    session,
                    "workflows",
                    "status IN ('failed', 'paused', 'completed') AND updated_at < :cutoff",
                    {"cutoff": cutoff}
                )
                await session.commit()
            except Exception as e:
                logger.error(f"Stale workflow cleanup failed: {e}")
                await session.rollback()

    async def run_background_cleanup_loop(self):
        """
        Long-running background task that schedules periodic low-priority cleanups.
        Runs once every 24 hours.
        """
        while True:
            logger.info("Starting low-priority background database cleanup...")
            await self.clean_expired_telemetry(days_old=30)
            await self.clean_stale_workflows(days_old=7)
            logger.info("Database cleanup finished. Sleeping for 24 hours.")
            await asyncio.sleep(86400) # 24 hours
