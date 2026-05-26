import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
import tiktoken
from agent_system.db.models import MemoryEntry
from agent_system.config import Config
from sqlalchemy.future import select

logger = logging.getLogger(__name__)

class HierarchicalMemoryManager:
    def __init__(self, session: AsyncSession, session_id: str, model_name: str = "gpt-4"):
        self.session = session
        self.session_id = session_id
        try:
            self.encoder = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoder = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    async def add_interaction(self, role: str, content: str, is_critical_context: bool = False) -> MemoryEntry:
        """
        Adds a new raw interaction to PostgreSQL MemoryEntry.
        Critical context (active workflows, preferences) are never summarized.
        """
        entry = MemoryEntry(
            session_id=self.session_id,
            role=role,
            content=content,
            token_count=self._count_tokens(content),
            is_critical_context=is_critical_context
        )
        async with self.session.begin_nested():
            self.session.add(entry)
        await self.session.flush()
        return entry

    async def get_context(self) -> List[Dict[str, str]]:
        """
        Retrieves context, triggering compression if limits are exceeded.
        """
        entries = await self._fetch_active_entries()
        total_tokens = sum(e.token_count for e in entries)
        
        if total_tokens > Config.MAX_TOKEN_SUMMARIZATION_THRESHOLD:
            await self._compress_history(entries, total_tokens)
            # Re-fetch after compression
            entries = await self._fetch_active_entries()

        # Format for LLM
        return [{"role": e.role, "content": e.content} for e in entries]

    async def _fetch_active_entries(self) -> List[MemoryEntry]:
        result = await self.session.execute(
            select(MemoryEntry)
            .filter_by(session_id=self.session_id)
            .order_by(MemoryEntry.created_at.asc())
        )
        return list(result.scalars().all())

    async def _compress_history(self, entries: List[MemoryEntry], current_tokens: int) -> None:
        """
        Hierarchical compression: 
        1. Identifies the oldest non-critical entries.
        2. Aggregates them until token threshold is safe.
        3. Replaces them with a single summarized MemoryEntry.
        """
        tokens_to_free = current_tokens - (Config.MAX_TOKEN_SUMMARIZATION_THRESHOLD * 0.8) # Free up 20%
        freed = 0
        to_compress = []
        
        for entry in entries:
            if freed >= tokens_to_free:
                break
            if not entry.is_critical_context and not entry.is_summarized:
                to_compress.append(entry)
                freed += entry.token_count

        if not to_compress:
            logger.warning("Token limit exceeded but no non-critical context available to compress.")
            return

        # In a real system, we would call an LLM here to generate the summary of `to_compress`.
        # For structural implementation, we mock the generated summary string:
        raw_text = "\n".join([f"{e.role}: {e.content}" for e in to_compress])
        summary_content = f"[Summarized Historical Context]: Previously discussed {len(to_compress)} interactions."
        
        summary_entry = MemoryEntry(
            session_id=self.session_id,
            role="system",
            content=summary_content,
            token_count=self._count_tokens(summary_content),
            is_summarized=True,
            is_critical_context=False
        )

        async with self.session.begin_nested():
            for e in to_compress:
                await self.session.delete(e)
            self.session.add(summary_entry)
        await self.session.flush()
