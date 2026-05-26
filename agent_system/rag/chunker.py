import re
from typing import List, Dict, Any
from .parser import LogicalUnit
from agent_system.config import Config

class LogicalChunker:
    def __init__(self, chunk_size: int = None, overlap: int = None):
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.overlap = overlap or Config.CHUNK_OVERLAP_DEFAULT

    def chunk_units(self, units: List[LogicalUnit]) -> List[Dict[str, Any]]:
        """
        Takes pre-segmented LogicalUnits and chunks them.
        Strict Isolation: NO overlap allowed between different LogicalUnits.
        """
        all_chunks = []
        for unit in units:
            if unit.type == "error":
                continue
            
            # If the unit fits completely within the chunk size, keep it whole.
            if len(unit.content) <= self.chunk_size:
                all_chunks.append(self._create_chunk_dict(unit.content, unit))
                continue
                
            # If unit is larger than chunk size, split it internally
            internal_chunks = self._split_internal(unit.content, has_equation=unit.equation_present)
            for ic in internal_chunks:
                all_chunks.append(self._create_chunk_dict(ic, unit))
                
        # Assign global chunk IDs
        for idx, chunk in enumerate(all_chunks):
            chunk["metadata"]["chunk_id"] = idx
            
        return all_chunks

    def _split_internal(self, text: str, has_equation: bool) -> List[str]:
        """
        Splits a single LogicalUnit internally.
        If it has an equation, equation boundaries take priority for splits.
        Internal chunks DO have overlap, but they are guaranteed to belong to the same logical question.
        """
        separators = []
        if has_equation:
            # Prioritize splitting around equations
            separators.extend(["\n$", "\n\\[", "\ny =", "\nx =", "\n\n"])
        
        separators.extend(["\n\n", "\n", ". ", " "])
        
        return self._recursive_split(text, separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]

        separator = separators[0] if separators else ""
        for sep in separators:
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                break

        if separator:
            splits = text.split(separator)
        else:
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        chunks = []
        current_chunk = ""

        for split in splits:
            if not split.strip():
                continue

            if len(current_chunk) + len(split) + len(separator) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Internal overlap is allowed here since it's the SAME logical unit
                if self.overlap > 0 and len(current_chunk) > self.overlap:
                    current_chunk = current_chunk[-self.overlap:] + separator + split
                else:
                    current_chunk = split
            else:
                if current_chunk:
                    current_chunk += separator + split
                else:
                    current_chunk = split

        if current_chunk:
            chunks.append(current_chunk.strip())

        final_chunks = []
        next_separators = separators[separators.index(separator) + 1:] if separator in separators else []
        
        for chunk in chunks:
            if len(chunk) > self.chunk_size and next_separators:
                final_chunks.extend(self._recursive_split(chunk, next_separators))
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _create_chunk_dict(self, text: str, unit: LogicalUnit) -> Dict[str, Any]:
        # FINAL BOUNDARY TRIMMING: aggressively remove any leaked next-question markers
        match = re.search(r'\n\s*(?:Q\s*)?\d+[.)]\s*', text)
        if match and match.start() > 10:
            text = text[:match.start()].strip()
            
        return {
            "text": text.strip(),
            "metadata": {
                "chunk_type": unit.type,
                "page": unit.page_num,
                "question_number": unit.question_num,
                "section_title": unit.section_title,
                "subquestion": unit.subquestion,
                "equation_present": unit.equation_present,
                "code_present": getattr(unit, "code_present", False)
            }
        }
