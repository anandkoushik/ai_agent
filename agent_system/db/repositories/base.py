from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import TypeVar, Generic, Type, Optional, List

T = TypeVar("T")

class BaseRepository(Generic[T]):
    """
    Generic repository providing transaction-safe CRUD operations.
    """
    def __init__(self, model_cls: Type[T], session: AsyncSession):
        self.model_cls = model_cls
        self.session = session

    async def get_by_id(self, id: str) -> Optional[T]:
        result = await self.session.execute(select(self.model_cls).filter_by(id=id))
        return result.scalars().first()

    async def get_all(self, skip: int = 0, limit: int = 100) -> List[T]:
        result = await self.session.execute(select(self.model_cls).offset(skip).limit(limit))
        return result.scalars().all()

    async def add(self, obj: T) -> T:
        """
        Uses explicit begin() block for non-blocking transaction safety.
        """
        async with self.session.begin_nested():
            self.session.add(obj)
        await self.session.flush()
        return obj

    async def delete(self, obj: T) -> None:
        async with self.session.begin_nested():
            await self.session.delete(obj)
        await self.session.flush()
