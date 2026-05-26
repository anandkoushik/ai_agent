import logging
import re
from .parser import DocumentParser
from .chunker import LogicalChunker
from agent_system.config import Config

class RAGSystem:
    def __init__(self, db_path="./rag_db"):
        self.db_path = db_path
        self.client = None
        self.encoder = None
        
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_or_create_collection("agent_knowledge")
            self.encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
            self.chunker = LogicalChunker(
                chunk_size=Config.CHUNK_SIZE, 
                overlap=Config.CHUNK_OVERLAP_DEFAULT
            )
        except ImportError:
            logging.warning("RAG disabled: chromadb or sentence_transformers not installed.")
        except Exception as e:
            logging.error(f"RAG init failed: {e}")

    def ingest(self, doc_id: str, filepath: str) -> dict:
        if not self.client or not self.encoder:
            return {"status": "error", "message": "RAG System not initialized"}

        try:
            # Step 1: Structure Extraction
            units = DocumentParser.parse_to_logical_units(filepath)
            
            # Step 2: Strict Chunking within boundaries
            chunks = self.chunker.chunk_units(units)
            
            if not chunks:
                return {"status": "error", "message": "No text extracted from document"}
                
            for i, chunk_data in enumerate(chunks):
                chunk_text = chunk_data["text"]
                metadata = chunk_data["metadata"]
                metadata["doc_id"] = doc_id
                
                # ChromaDB requires metadata values to be str, int, float or bool
                # Convert None to empty string
                for key, val in metadata.items():
                    if val is None:
                        metadata[key] = ""
                
                chunk_id = f"{doc_id}_chunk_{i}"
                embeddings = self.encoder.encode([chunk_text]).tolist()
                
                self.collection.add(
                    ids=[chunk_id], 
                    embeddings=embeddings, 
                    documents=[chunk_text],
                    metadatas=[metadata]
                )
            
            return {"status": "success", "chunks_added": len(chunks), "doc_id": doc_id}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def retrieve(self, query: str, k: int = None, method: str = "hybrid") -> list[dict]:
        k = k or Config.RAG_TOP_K
        
        if not self.client or not self.encoder:
            return []
            
        retrieved = []
        try:
            # 1. Semantic Retrieval
            if method in ["semantic", "hybrid"]:
                emb = self.encoder.encode([query]).tolist()
                results = self.collection.query(
                    query_embeddings=emb, 
                    n_results=k * 2,  # Fetch more for post-processing ranking
                    include=["documents", "metadatas", "distances"]
                )
                
                # Check if query is highly technical
                is_tech_query = DocumentParser._is_equation(query) or DocumentParser._is_code_or_assembly(query)
                
                if results['documents']:
                    for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
                        
                        # POST-RETRIEVAL TRIMMING
                        # Aggressive catch for leaking next-question markers (e.g., "\n5. ", "\nQ5:")
                        match = re.search(r'\n\s*(?:Q\s*)?\d+[.)]\s*', doc)
                        if match and match.start() > 10:
                            doc = doc[:match.start()].strip()
                            
                        # TECHNICAL BOOSTING
                        if is_tech_query:
                            if meta.get("equation_present") == True or meta.get("code_present") == True:
                                dist = dist * 0.8  # Lower distance = better score
                        
                        retrieved.append({
                            "text": doc,
                            "metadata": meta,
                            "distance": dist,
                            "source": "semantic"
                        })
                        
            # Hybrid hook (BM25) would go here
            
            retrieved.sort(key=lambda x: x.get("distance", float('inf')))
            return retrieved[:k]
            
        except Exception as e:
            logging.error(f"Error retrieving from RAG: {e}")
            return []
