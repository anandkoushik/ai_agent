import logging
from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from agent_system.db.models import FailureMemory
from agent_system.config import Config

logger = logging.getLogger(__name__)

class AdaptiveFailureMemory:
    def __init__(self, db_session: AsyncSession, db_path="./agent_system/rag_db"):
        self.session = db_session
        self.chroma_client = None
        self.encoder = None
        
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
            # Re-use the existing ChromaDB instance for storing vector embeddings of errors
            self.chroma_client = chromadb.PersistentClient(path=db_path)
            self.collection = self.chroma_client.get_or_create_collection("failure_memory_vectors")
            self.encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        except ImportError:
            logger.warning("Adaptive Failure Memory: chromadb/sentence_transformers disabled.")

    async def log_failure(self, error_pattern: str, root_cause: str, applied_fix: str, outcome_success: bool, context: Dict[str, Any]) -> FailureMemory:
        """
        Embeds the error pattern into ChromaDB and stores strict relational metadata into PostgreSQL.
        """
        chroma_id = f"fail_{hash(error_pattern)}"
        
        if self.chroma_client and self.encoder:
            embeddings = self.encoder.encode([error_pattern]).tolist()
            self.collection.add(
                ids=[chroma_id],
                embeddings=embeddings,
                documents=[error_pattern],
                metadatas=[{"status": "resolved" if outcome_success else "unresolved"}]
            )
            
        failure_record = FailureMemory(
            chroma_vector_id=chroma_id,
            error_pattern=error_pattern,
            root_cause=root_cause,
            applied_fix=applied_fix,
            outcome_success=outcome_success,
            hardware_context=context.get("hardware"),
            model_type=context.get("model_type"),
            dataset_type=context.get("dataset_type"),
            framework_compatibility=context.get("framework")
        )
        
        async with self.session.begin_nested():
            self.session.add(failure_record)
        await self.session.flush()
        return failure_record

    async def search_similar_failure(self, current_error: str, current_context: Dict[str, Any]) -> Optional[FailureMemory]:
        """
        Queries ChromaDB for semantically similar errors, then retrieves structured Postgres data.
        Validates context compatibility before returning the historical fix.
        """
        if not self.chroma_client or not self.encoder:
            return None
            
        embeddings = self.encoder.encode([current_error]).tolist()
        results = self.collection.query(
            query_embeddings=embeddings,
            n_results=1
        )
        
        if not results['ids'] or not results['ids'][0]:
            return None
            
        best_match_id = results['ids'][0][0]
        
        # Fetch detailed record from PostgreSQL
        pg_result = await self.session.execute(
            select(FailureMemory).filter_by(chroma_vector_id=best_match_id)
        )
        record = pg_result.scalars().first()
        
        if not record:
            return None
            
        # Validation Checks
        if record.model_type and current_context.get("model_type") != record.model_type:
            logger.info("Discarding historical fix: Model type mismatch")
            return None
            
        if record.dataset_type and current_context.get("dataset_type") != record.dataset_type:
            logger.info("Discarding historical fix: Dataset type mismatch")
            return None
            
        return record
