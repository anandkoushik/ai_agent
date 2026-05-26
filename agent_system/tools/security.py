import os
import zipfile
import mimetypes
import logging
from typing import Any, Dict
from agent_system.config import Config

class SecurityException(Exception):
    pass

class SecurityManager:
    @staticmethod
    def sanitize_path(user_path: str, allowed_base: str) -> str:
        """
        Prevents directory traversal (e.g., '../../etc/passwd').
        Returns the absolute path if safe, otherwise raises SecurityException.
        """
        abs_base = os.path.abspath(allowed_base)
        abs_target = os.path.abspath(os.path.join(abs_base, user_path))
        
        if not os.path.commonpath([abs_base, abs_target]) == abs_base:
            raise SecurityException(f"Path traversal detected: {user_path}")
            
        return abs_target

    @staticmethod
    def validate_extension(filename: str):
        """Checks against a strict whitelist of extensions."""
        ext = os.path.splitext(filename)[1].lower()
        if ext not in Config.ALLOWED_EXTENSIONS:
            raise SecurityException(f"Unsupported file extension: {ext}")

    @staticmethod
    def validate_mime_type(filepath: str):
        """
        Attempts to validate the MIME type using python's built-in mimetypes
        or basic file signature checks.
        """
        mime_type, _ = mimetypes.guess_type(filepath)
        if not mime_type:
            # Fallback for unknown extensions or files without extensions
            logging.warning(f"Could not determine MIME type for {filepath}")
            return
        
        # Disallow executable/script MIME types
        forbidden_mimes = ['application/x-msdownload', 'application/x-sh', 'text/x-python', 'application/javascript']
        if mime_type in forbidden_mimes:
            raise SecurityException(f"Forbidden MIME type detected: {mime_type}")

    @staticmethod
    def check_zip_bomb(filepath: str):
        """
        Inspects a ZIP file for ZIP bombs by checking uncompressed sizes
        and compression ratios before extracting.
        """
        try:
            with zipfile.ZipFile(filepath, 'r') as zf:
                total_uncompressed = 0
                for info in zf.infolist():
                    total_uncompressed += info.file_size
                    
                    if info.file_size > 0 and info.compress_size > 0:
                        ratio = info.file_size / info.compress_size
                        if ratio > Config.MAX_COMPRESSION_RATIO:
                            raise SecurityException(
                                f"Possible ZIP bomb detected! High compression ratio: {ratio:.2f}"
                            )
                
                if total_uncompressed > Config.MAX_UNCOMPRESSED_SIZE:
                    raise SecurityException(
                        f"ZIP extracted size exceeds limit! ({total_uncompressed / 1e9:.2f} GB)"
                    )
        except zipfile.BadZipFile:
            raise SecurityException("Invalid ZIP file format.")

    @staticmethod
    def sanitize_tool_kwargs(tool_name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates that arguments do not contain shell injections.
        """
        for key, value in kwargs.items():
            if isinstance(value, str):
                forbidden_chars = [';', '&&', '|', '`', '$(']
                for char in forbidden_chars:
                    if char in value:
                        raise SecurityException(f"Potential shell injection detected in argument '{key}': {value}")
        return kwargs
