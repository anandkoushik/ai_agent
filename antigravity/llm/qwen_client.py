"""
antigravity.llm.qwen_client — Qwen Inference Client

Provides local Qwen inference via HuggingFace Transformers.
Supports both single-shot and streaming generation.

Usage:
    from antigravity.llm.qwen_client import ask_qwen

    answer = ask_qwen("Explain gradient descent in 3 sentences.")
    print(answer)
"""

import os
import logging
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy singletons — model + tokenizer loaded once, reused across calls
# ---------------------------------------------------------------------------
_model = None
_tokenizer = None
_loaded_model_id: Optional[str] = None

# Default Qwen model (small enough for RTX 4090 24 GB)
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def _load_model(model_id: Optional[str] = None, device: Optional[str] = None):
    """Load (or reload) Qwen model + tokenizer as lazy singletons."""
    global _model, _tokenizer, _loaded_model_id

    model_id = model_id or DEFAULT_MODEL_ID

    # Skip reload if already loaded with the same ID
    if _model is not None and _loaded_model_id == model_id:
        return _model, _tokenizer

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32

    logger.info(f"[qwen_client] Loading {model_id} on {device} ({dtype})")

    _tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    _model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device if device == "cuda" else None,
        trust_remote_code=True,
    )

    if device != "cuda":
        _model.to(device)

    _model.eval()
    _loaded_model_id = model_id

    logger.info(f"[qwen_client] {model_id} ready on {device}")
    return _model, _tokenizer


# ============================================================
#  ASK QWEN — Single-shot generation
# ============================================================

def ask_qwen(
    prompt: str,
    system_prompt: str = "You are a helpful AI assistant specializing in AI, ML, and Python.",
    model_id: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Generate a single response from Qwen.

    Args:
        prompt:          The user query.
        system_prompt:   System-level instruction for the model.
        model_id:        HuggingFace model ID (defaults to Qwen2.5-1.5B-Instruct).
        max_new_tokens:  Maximum tokens to generate.
        temperature:     Sampling temperature (lower = more deterministic).
        top_p:           Nucleus sampling threshold.
        do_sample:       Whether to sample or use greedy decoding.
        history:         Optional chat history as list of {"role": ..., "content": ...}.

    Returns:
        The model's response as a string.
    """
    import torch

    model, tokenizer = _load_model(model_id)

    # Build chat messages
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": prompt})

    # Use the chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        # Fallback: simple concatenation
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        text += "\nassistant:"

    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )

    # Decode only the newly generated tokens
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return response


# ============================================================
#  ASK QWEN STREAM — Token-by-token generator
# ============================================================

def ask_qwen_stream(
    prompt: str,
    system_prompt: str = "You are a helpful AI assistant specializing in AI, ML, and Python.",
    model_id: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Generator[str, None, None]:
    """
    Stream tokens from Qwen one at a time.

    Yields:
        Individual token strings as they are generated.
    """
    import torch
    from transformers import TextIteratorStreamer
    from threading import Thread

    model, tokenizer = _load_model(model_id)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = f"system: {system_prompt}\nuser: {prompt}\nassistant:"

    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True,
        "streamer": streamer,
    }

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for token in streamer:
        yield token

    thread.join()


# ============================================================
#  UNLOAD — Free GPU memory
# ============================================================

def unload_model() -> None:
    """Unload the Qwen model and free GPU memory."""
    global _model, _tokenizer, _loaded_model_id
    import gc

    if _model is not None:
        del _model
        _model = None

    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None

    _loaded_model_id = None
    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("[qwen_client] GPU memory released")
    except ImportError:
        pass

    logger.info("[qwen_client] Model unloaded")
