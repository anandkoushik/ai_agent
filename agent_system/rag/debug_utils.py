import json

class RAGDebugger:
    @staticmethod
    def visualize_retrieval(query: str, results: list[dict]) -> str:
        """
        Returns a formatted string for debugging retrieved chunks,
        including similarity scores, metadata boundaries, and text overlap.
        """
        output = [f"=== RAG Debugging Report ==="]
        output.append(f"Query: '{query}'")
        output.append(f"Total Chunks Retrieved: {len(results)}\n")
        
        for i, res in enumerate(results):
            dist = res.get('distance', 'N/A')
            meta = res.get('metadata', {})
            text = res.get('text', '')
            
            output.append(f"--- Chunk {i+1} ---")
            output.append(f"Distance Score: {dist}")
            output.append(f"Chunk Type:     {meta.get('chunk_type', 'none')}")
            output.append(f"Question Num:   {meta.get('question_number', 'unknown')}")
            output.append(f"Section:        {meta.get('section_title', 'unknown')}")
            output.append(f"Subquestion:    {meta.get('subquestion', 'none')}")
            output.append(f"Has Equation:   {meta.get('equation_present', False)}")
            output.append(f"Content Length: {len(text)} chars")
            output.append(f"--- Content Start ---")
            output.append(text)
            output.append(f"--- Content End ---\n")
            
        return "\n".join(output)

    @staticmethod
    def validate_chunk_coherence(chunk_metadata: dict) -> bool:
        """
        Validates if a chunk is coherent before passing to LLM.
        For example, if a chunk has no boundary but contains partial equations, 
        or if it spans multiple unrelated questions, we can flag it.
        """
        # Placeholder for more advanced hallucination/leakage checks
        return True
