import os
import zipfile
from typing import Dict, Any

class ZipParser:
    @staticmethod
    def extract_and_parse(zip_path: str, extract_to: str, max_size_bytes=5 * 1024 * 1024 * 1024) -> Dict[str, Any]:
        """
        Safely extracts a ZIP file, preventing path traversal and size bombs.
        Returns metadata about the extracted contents.
        """
        total_size = 0
        extracted_files = []
        os.makedirs(extract_to, exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for info in zf.infolist():
                    total_size += info.file_size
                    if total_size > max_size_bytes:
                        raise ValueError("ZIP extraction exceeds maximum allowed size.")
                    if '..' in info.filename or info.filename.startswith('/'):
                        raise ValueError(f"Invalid path in ZIP archive: {info.filename}")
                    
                    zf.extract(info, extract_to)
                    if not info.is_dir():
                        extracted_files.append(info.filename)
                        
            metadata = {
                "total_files": len(extracted_files),
                "total_size_bytes": total_size,
                "extracted_to": extract_to,
                "file_types": list(set([os.path.splitext(f)[1].lower() for f in extracted_files if '.' in f]))
            }
            return {"status": "success", "metadata": metadata}
        except Exception as e:
            return {"status": "error", "error": str(e)}
