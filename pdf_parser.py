import fitz  # PyMuPDF
from typing import Dict, Any

class PDFParser:
    @staticmethod
    def parse(file_path: str, extract_text: bool = True) -> Dict[str, Any]:
        """
        Parses a PDF file to extract metadata and optionally text.
        """
        try:
            doc = fitz.open(file_path)
            metadata = {
                "page_count": doc.page_count,
                "author": doc.metadata.get("author", ""),
                "title": doc.metadata.get("title", ""),
                "creation_date": doc.metadata.get("creationDate", "")
            }
            
            text_content = ""
            if extract_text:
                # Extract up to first 5 pages to avoid massive text strings
                for i in range(min(5, doc.page_count)):
                    text_content += doc[i].get_text()
            
            doc.close()
            
            return {
                "status": "success", 
                "metadata": metadata, 
                "text_snippet": text_content[:2000] if extract_text else ""
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
