import asyncio
import logging
from agent_system.db.session import engine, AsyncSessionLocal
from agent_system.db.models import Base, Workflow, MemoryEntry
from sqlalchemy.future import select

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_database():
    try:
        logger.info("Connecting to PostgreSQL and creating tables...")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Tables created successfully!")
        
        logger.info("Testing transaction and ORM inserts...")
        async with AsyncSessionLocal() as session:
            # Test Workflow insert
            wf = Workflow(workflow_id="test_wf_001", status="queued")
            session.add(wf)
            
            # Test Memory Entry insert
            mem = MemoryEntry(session_id="session_001", role="user", content="Test phase 4", token_count=3)
            session.add(mem)
            
            await session.commit()
            logger.info("Inserts successful!")
            
            # Test retrieve
            result = await session.execute(select(Workflow).filter_by(workflow_id="test_wf_001"))
            fetched_wf = result.scalars().first()
            if fetched_wf:
                logger.info(f"Successfully retrieved workflow: {fetched_wf.workflow_id} with status {fetched_wf.status}")
                
    except Exception as e:
        logger.error(f"Database test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_database())
