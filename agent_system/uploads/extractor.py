import os
import shutil
import tempfile
import zipfile
import logging

logger = logging.getLogger(__name__)

class UploadExtractor:
    """Handles ZIP extraction with path traversal & size bomb protection."""
    
    @staticmethod
    def extract_safely(zip_path: str, max_size_bytes=5 * 1024 * 1024 * 1024) -> str:
        extract_dir = tempfile.mkdtemp(prefix="chat_session_")
        logger.info(f"Extracting {zip_path} into sandboxed session {extract_dir}")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                total_size = 0
                for info in zf.infolist():
                    total_size += info.file_size
                    if total_size > max_size_bytes:
                        raise ValueError("Zip bomb detected. Extraction aborted.")
                    # Prevent path traversal
                    if ".." in info.filename or info.filename.startswith("/"):
                        raise ValueError(f"Path traversal attempt blocked: {info.filename}")
                        
                zf.extractall(extract_dir)
            return extract_dir
        except Exception as e:
            # Resumable cleanup: clean partial extraction state on failure
            logger.error(f"Extraction failed: {e}. Cleaning up {extract_dir}")
            shutil.rmtree(extract_dir, ignore_errors=True)
            raise e
