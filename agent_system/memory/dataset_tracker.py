import hashlib
import logging
from typing import Optional, Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from agent_system.db.models import Dataset, DatasetVersion

logger = logging.getLogger(__name__)

class DatasetTracker:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def _hash_file(self, filepath: str) -> str:
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    async def register_dataset(self, filepath: str, original_filename: str, file_type: str) -> DatasetVersion:
        """
        Hashes the dataset to prevent double-ingestion.
        Returns the initial DatasetVersion.
        """
        file_hash = await self._hash_file(filepath)
        
        # Check if dataset exists
        result = await self.session.execute(select(Dataset).filter_by(dataset_hash=file_hash))
        existing_ds = result.scalars().first()
        
        async with self.session.begin_nested():
            if existing_ds:
                logger.info(f"Dataset {original_filename} already registered. Skipping ingestion.")
                # Return the latest version
                v_res = await self.session.execute(select(DatasetVersion).filter_by(dataset_id=existing_ds.id))
                return v_res.scalars().first()
            
            # Create new dataset
            new_ds = Dataset(
                dataset_hash=file_hash,
                original_filename=original_filename,
                file_type=file_type
            )
            self.session.add(new_ds)
            await self.session.flush() # Ensure ID is generated
            
            # Create v1
            new_version = DatasetVersion(
                dataset_id=new_ds.id,
                version_tag="v1.0-raw",
                validation_status="pending"
            )
            self.session.add(new_version)
            
        await self.session.flush()
        return new_version

    async def create_derived_version(self, parent_version_id: str, new_tag: str, operations: List[str], val_split: float = 0.2) -> DatasetVersion:
        """
        Lineage tracking: creates a child version recording the exact preprocessing applied.
        """
        result = await self.session.execute(select(DatasetVersion).filter_by(id=parent_version_id))
        parent = result.scalars().first()
        if not parent:
            raise ValueError(f"Parent version {parent_version_id} not found.")
            
        new_version = DatasetVersion(
            dataset_id=parent.dataset_id,
            version_tag=new_tag,
            parent_version_id=parent_version_id,
            preprocessing_operations=operations,
            validation_split_ratio=val_split,
            validation_status="valid"
        )
        
        async with self.session.begin_nested():
            self.session.add(new_version)
        await self.session.flush()
        return new_version
