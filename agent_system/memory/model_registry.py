import logging
from typing import Optional, Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from agent_system.db.models import ModelRegistry

logger = logging.getLogger(__name__)

class ModelRegistryTracker:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def register_model(self, model_name: str, version: str, dataset_version_id: str, hyperparameters: Dict[str, Any]) -> ModelRegistry:
        """
        Registers a new model training job into the registry.
        """
        model_entry = ModelRegistry(
            model_name=model_name,
            version=version,
            dataset_version_id=dataset_version_id,
            hyperparameters=hyperparameters,
            status="training"
        )
        
        async with self.session.begin_nested():
            self.session.add(model_entry)
        await self.session.flush()
        return model_entry

    async def update_model_status(self, model_id: str, status: str) -> Optional[ModelRegistry]:
        """
        Updates the status (e.g., 'training' -> 'ready' or 'failed').
        """
        result = await self.session.execute(select(ModelRegistry).filter_by(id=model_id))
        model_entry = result.scalars().first()
        
        if model_entry:
            model_entry.status = status
            async with self.session.begin_nested():
                self.session.add(model_entry)
            await self.session.flush()
            
        return model_entry

    async def get_models_by_dataset(self, dataset_version_id: str) -> List[ModelRegistry]:
        """
        Lineage tracking: Find all models trained on a specific dataset version.
        """
        result = await self.session.execute(select(ModelRegistry).filter_by(dataset_version_id=dataset_version_id))
        return list(result.scalars().all())
