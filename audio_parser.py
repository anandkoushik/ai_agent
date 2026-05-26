import librosa
from typing import Dict, Any

class AudioParser:
    @staticmethod
    def parse(file_path: str) -> Dict[str, Any]:
        """
        Parses an audio file to extract metadata such as duration and sample rate.
        """
        try:
            # We only load the metadata without loading the full array into memory using sr=None, duration=0 if possible
            # librosa.get_duration is faster
            duration = librosa.get_duration(path=file_path)
            sr = librosa.get_samplerate(file_path)
            
            metadata = {
                "duration_seconds": round(duration, 2),
                "sample_rate": sr
            }
            return {"status": "success", "metadata": metadata}
        except Exception as e:
            return {"status": "error", "error": str(e)}
