from PIL import Image
from typing import Dict, Any

class ImageParser:
    @staticmethod
    def parse(file_path: str) -> Dict[str, Any]:
        """
        Parses an image file to extract metadata such as dimensions and format.
        """
        try:
            with Image.open(file_path) as img:
                metadata = {
                    "format": img.format,
                    "mode": img.mode,
                    "width": img.width,
                    "height": img.height
                }
            return {"status": "success", "metadata": metadata}
        except Exception as e:
            return {"status": "error", "error": str(e)}
