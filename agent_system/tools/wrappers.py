import asyncio
import traceback
import io
import sys
import contextlib

def _strict_schema(status_code: int, stdout: str, stderr: str, metrics: dict) -> dict:
    return {
        "status_code": status_code,
        "stdout": stdout,
        "stderr": stderr,
        "metrics": metrics
    }

async def wrap_sync_function(func, *args, **kwargs):
    """
    Wraps a synchronous function in an asyncio thread to avoid blocking the event loop.
    Captures stdout/stderr and enforces the strict schema.
    """
    def _execute():
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        result = None
        
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            try:
                result = func(*args, **kwargs)
                return _strict_schema(0, stdout_buf.getvalue(), stderr_buf.getvalue(), {"result": result})
            except Exception as e:
                err_str = stderr_buf.getvalue() + "\n" + traceback.format_exc()
                return _strict_schema(1, stdout_buf.getvalue(), err_str, {"error": str(e)})

    return await asyncio.to_thread(_execute)

# Specifically wrapped tools from the existing untouched backend
async def train_whisper_tool(**kwargs):
    import training
    return await wrap_sync_function(training.train_whisper, **kwargs)

async def train_llm_tool(**kwargs):
    import training
    return await wrap_sync_function(training.train_llm, **kwargs)

async def make_model_zip_tool(**kwargs):
    import inference
    return await wrap_sync_function(inference.make_model_zip, **kwargs)

async def analyze_dataset_tool(**kwargs):
    # This comes from chatbot.py legacy logic or external class
    # For now, we simulate the wrapper
    def _dummy_analyze(workspace_dir):
        return {"type": "llm", "valid": True}
        
    return await wrap_sync_function(_dummy_analyze, **kwargs)
