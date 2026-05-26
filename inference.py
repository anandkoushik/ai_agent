from __future__ import annotations
import os
import zipfile
from datetime import datetime

# ----------------------------
# Inference file manifests
# ----------------------------

LLM_INFERENCE_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors",      # preferred
    "pytorch_model.bin",      # fallback
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]

WHISPER_INFERENCE_FILES = [
    "config.json",
    "preprocessor_config.json",
    "model.safetensors",      # preferred
    "pytorch_model.bin",      # fallback
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]


def _log(msg: str) -> None:
    import logging
    logging.getLogger(__name__).info(msg)


def _timestamp() -> str:
    """YYYYMMDD-HHMM"""
    return datetime.now().strftime("%Y%m%d-%H%M")

# ============================================================
# VISION INFERENCE ZIP (ResNet18 / ResNet50 / MobileNet)
# ============================================================

VISION_INFERENCE_FILES = [
    "model.pth",
    "labels.json",
    "model_config.json",
]


def make_vision_inference_zip(
    ws_dir: str,
    model_name: str,          # resnet18 | resnet50 | mobilenet
    workspace_name: str,
) -> str | None:
    """
    Create inference-only ZIP for Vision models.

    Source:
        ws_dir/models/vision/<model_name>

    ZIP name:
        <workspace>_vision_<model_name>_<YYYYMMDD-HHMM>.zip
    """

    vision_dir = os.path.join(ws_dir, "models", "vision", model_name)

    if not os.path.isdir(vision_dir):
        _log(f"[VISION ZIP] Model directory not found: {vision_dir}")
        return None

    zip_root = os.path.join(ws_dir, "models_zip")
    os.makedirs(zip_root, exist_ok=True)

    zip_name = f"{workspace_name}_vision_{model_name}_{_timestamp()}"
    zip_path = os.path.join(zip_root, f"{zip_name}.zip")

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in VISION_INFERENCE_FILES:
                fpath = os.path.join(vision_dir, fname)
                if not os.path.exists(fpath):
                    raise FileNotFoundError(f"Missing vision file: {fname}")
                zf.write(fpath, arcname=fname)

        _log(f"[VISION ZIP] Created {zip_name}.zip")
        return zip_path

    except Exception as e:
        _log(f"[VISION ZIP] Failed: {e}")
        return None
# ============================================================
# GENERIC MODEL ZIP (YOLO / WHISPER full directory)
# ============================================================

def make_model_zip(
    ws_dir: str,
    model_type: str,
    model_name: str,
    workspace_name: str,
) -> str | None:
    """
    Create ZIP of:
        ws_dir/models/<model_type>/<model_name>

    ZIP name:
        <workspace>_<model_type>_<model_name>_<YYYYMMDD-HHMM>.zip
    """

    models_src = os.path.join(ws_dir, "models", model_type, model_name)

    if not os.path.isdir(models_src):
        _log(f"[ZIP] Model directory not found: {models_src}")
        return None

    zip_root = os.path.join(ws_dir, "models_zip")
    os.makedirs(zip_root, exist_ok=True)

    zip_name = f"{workspace_name}_{model_type}_{model_name}_{_timestamp()}"
    zip_path = os.path.join(zip_root, f"{zip_name}.zip")

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(models_src):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, models_src)
                    zf.write(full_path, rel_path)

        _log(f"[ZIP] Created {zip_name}.zip")
        return zip_path

    except Exception as e:
        _log(f"[ZIP] Error: {e}")
        return None


# ============================================================
# LLM INFERENCE ZIP (TinyLlama / GPT-2)
# ============================================================

def make_llm_inference_zip(
    ws_dir: str,
    model_name: str,
    workspace_name: str,
) -> str | None:
    """
    Create inference-only ZIP for LLM.

    Source:
        ws_dir/models/llm/<model_name>

    ZIP name:
        <workspace>_llm_<model_name>_<YYYYMMDD-HHMM>.zip
    """

    llm_dir = os.path.join(ws_dir, "models", "llm", model_name)

    if not os.path.isdir(llm_dir):
        _log(f"[LLM ZIP] Model directory not found: {llm_dir}")
        return None

    zip_root = os.path.join(ws_dir, "models_zip")
    os.makedirs(zip_root, exist_ok=True)

    zip_name = f"{workspace_name}_llm_{model_name}_{_timestamp()}"
    zip_path = os.path.join(zip_root, f"{zip_name}.zip")

    try:
        # ----------------------------------------
        # ✅ VALIDATE REQUIRED FILES
        # ----------------------------------------
        required_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ]

        for fname in required_files:
            if not os.path.exists(os.path.join(llm_dir, fname)):
                raise FileNotFoundError(f"Missing required LLM file: {fname}")

        # ✅ CHECK MODEL WEIGHTS (at least one must exist)
        weight_files = ["model.safetensors", "pytorch_model.bin"]

        if not any(os.path.exists(os.path.join(llm_dir, f)) for f in weight_files):
            raise FileNotFoundError(
                "Missing model weights (model.safetensors or pytorch_model.bin)"
            )

        # ----------------------------------------
        # ✅ CREATE ZIP (ONLY IF VALID)
        # ----------------------------------------
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in LLM_INFERENCE_FILES:
                fpath = os.path.join(llm_dir, fname)

                if os.path.exists(fpath):
                    zf.write(fpath, arcname=fname)

        _log(f"[LLM ZIP] Created {zip_name}.zip")
        return zip_path

    except Exception as e:
        _log(f"[LLM ZIP] Failed: {e}")
        return None


# ============================================================
# WHISPER INFERENCE ZIP (tiny / base / small)
# ============================================================

def make_whisper_inference_zip(
    ws_dir: str,
    model_name: str,
    workspace_name: str,
) -> str | None:
    """
    Create inference-ready Whisper ZIP from latest checkpoint.
    """

    import os
    import zipfile
    import shutil
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    size = model_name.split("_")[-1]

    base_dir = os.path.join(ws_dir, "models", "whisper", size)

    if not os.path.isdir(base_dir):
        _log(f"[Whisper ZIP] Model directory not found: {base_dir}")
        return None

    checkpoints = [
        d for d in os.listdir(base_dir)
        if d.startswith("checkpoint")
    ]

    if not checkpoints:
        _log("[Whisper ZIP] No checkpoints found → using base dir")
        checkpoint_dir = base_dir
    else:
        checkpoints.sort(key=lambda x: int(x.split("-")[-1]), reverse=True)
        checkpoint_dir = os.path.join(base_dir, checkpoints[0])

    final_dir = os.path.join(base_dir, "final_model")

    if os.path.exists(final_dir):
        shutil.rmtree(final_dir)

    os.makedirs(final_dir, exist_ok=True)

    try:
        model = WhisperForConditionalGeneration.from_pretrained(checkpoint_dir)
        processor = WhisperProcessor.from_pretrained(checkpoint_dir)

        model.save_pretrained(final_dir)
        processor.save_pretrained(final_dir)

    except Exception as e:
        _log(f"[Whisper ZIP] Export failed: {e}")
        return None

    required = [
        "config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]

    for r in required:
        if not os.path.exists(os.path.join(final_dir, r)):
            _log(f"[Whisper ZIP] Missing file: {r}")
            return None

    if not any(os.path.exists(os.path.join(final_dir, f)) for f in ["model.safetensors", "pytorch_model.bin"]):
        _log("[Whisper ZIP] Missing model weights")
        return None

    zip_root = os.path.join(ws_dir, "models_zip")
    os.makedirs(zip_root, exist_ok=True)

    zip_name = f"{workspace_name}_whisper_{size}_{_timestamp()}"
    zip_path = os.path.join(zip_root, f"{zip_name}.zip")

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in os.listdir(final_dir):
                full = os.path.join(final_dir, f)
                if os.path.isfile(full):
                    zf.write(full, f)

        _log(f"[Whisper ZIP] Created {zip_name}.zip")
        return zip_path

    except Exception as e:
        _log(f"[Whisper ZIP] Failed: {e}")
        return None


def train_whisper(data_path: str, epochs: int, workspace_dir: str, model_type: str):

    import os
    import gc
    import torch
    import librosa

    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
        Trainer,
        TrainingArguments
    )

    from datasets import load_dataset
    from app.core.config import settings

    # --------------------------------------------------
    # FIND FILES
    # --------------------------------------------------

    def get_all_files(data_path, extensions):

        files_found = []

        for root, _, files in os.walk(data_path):

            for file in files:

                if any(file.lower().endswith(ext) for ext in extensions):

                    files_found.append(
                        os.path.join(root, file)
                    )

        return files_found

    # ✅ SUPPORT WAV + MP3
    audio_files = get_all_files(
        data_path,
        [".wav", ".mp3"]
    )

    csv_files = get_all_files(
        data_path,
        [".csv"]
    )

    if not audio_files:
        raise ValueError("❌ No audio files found")

    if not csv_files:
        raise ValueError("❌ No CSV files found")

    print(f"✅ Found {len(audio_files)} audio files")
    print(f"✅ Found {len(csv_files)} csv files")

    # --------------------------------------------------
    # BUILD AUDIO LOOKUP TABLE
    # --------------------------------------------------

    audio_lookup = {}

    for f in audio_files:

        filename = os.path.basename(f).strip()

        audio_lookup[filename] = f

    print("✅ REGISTERED AUDIO FILES:")
    print(list(audio_lookup.keys())[:10])

    # --------------------------------------------------
    # MODEL SELECT
    # --------------------------------------------------

    size = (
        "tiny" if "tiny" in model_type
        else "base" if "base" in model_type
        else "small"
    )

    model_id = f"openai/whisper-{size}"

    revision = getattr(
        settings,
        f"hf_revision_whisper_{size}"
    )

    output_dir = os.path.join(
        workspace_dir,
        "models",
        "whisper",
        size
    )

    os.makedirs(output_dir, exist_ok=True)

    print(f"🚀 [WHISPER INIT MODEL] → {output_dir}")

    # --------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------

    try:

        model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=True
        )

        processor = WhisperProcessor.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=True
        )

    except OSError:

        model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            revision=revision
        )

        processor = WhisperProcessor.from_pretrained(
            model_id,
            revision=revision
        )

    # --------------------------------------------------
    # LOAD CSV DATASET
    # --------------------------------------------------

    dataset = load_dataset(
        "csv",
        data_files={"train": csv_files}
    )["train"]

    print(f"✅ CSV rows loaded: {len(dataset)}")

    # --------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------

    def preprocess(example):

        try:

            keys = list(example.keys())

            print("\n🔍 ROW KEYS:", keys)

            # ------------------------------------------
            # DETECT AUDIO COLUMN
            # ------------------------------------------

            audio_col = next(

                (
                    k for k in keys

                    if (
                        "audio" in k.lower()
                        or "path" in k.lower()
                        or "file" in k.lower()
                    )
                ),

                None
            )

            if audio_col is None:

                print("❌ AUDIO COLUMN NOT FOUND")

                return {}

            # ------------------------------------------
            # DETECT TEXT COLUMN
            # ------------------------------------------

            if "text" in keys:
                text_col = "text"

            elif "transcription" in keys:
                text_col = "transcription"

            elif "sentence" in keys:
                text_col = "sentence"

            elif "label" in keys:
                text_col = "label"

            else:

                print(f"❌ TEXT COLUMN NOT FOUND IN {keys}")

                return {}

            # ------------------------------------------
            # GET RAW VALUES
            # ------------------------------------------

            raw_audio_path = str(
                example[audio_col]
            ).strip()

            text = str(
                example[text_col]
            ).strip()

            print(f"🎧 RAW AUDIO PATH: {raw_audio_path}")
            print(f"📝 TEXT: {text[:50]}")

            if not raw_audio_path:

                print("❌ EMPTY AUDIO PATH")

                return {}

            if not text:

                print("❌ EMPTY TEXT")

                return {}

            # ------------------------------------------
            # REMOVE LABEL STUDIO PREFIX
            # ------------------------------------------

            audio_filename = os.path.basename(
                raw_audio_path
            ).strip()

            print(f"🎯 AUDIO FILENAME: {audio_filename}")

            # ------------------------------------------
            # MATCH AUDIO
            # ------------------------------------------

            if audio_filename not in audio_lookup:

                print(f"❌ AUDIO NOT FOUND: {audio_filename}")

                print("✅ AVAILABLE AUDIO:")
                print(list(audio_lookup.keys())[:20])

                return {}

            audio_path = audio_lookup[audio_filename]

            print(f"✅ MATCHED AUDIO: {audio_path}")

            # ------------------------------------------
            # LOAD AUDIO
            # ------------------------------------------

            speech, sr = librosa.load(
                audio_path,
                sr=16000
            )

            if speech is None or len(speech) == 0:

                print(f"❌ EMPTY AUDIO FILE: {audio_path}")

                return {}

            print(f"✅ AUDIO LOADED: {len(speech)} samples")

            # ------------------------------------------
            # EXTRACT FEATURES
            # ------------------------------------------

            input_features = processor.feature_extractor(
                speech,
                sampling_rate=16000
            ).input_features[0]

            # ------------------------------------------
            # TOKENIZE TEXT
            # ------------------------------------------

            labels = processor.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=128
            ).input_ids

            return {

                "input_features": input_features,

                "labels": labels
            }

        except Exception as e:

            print("❌ PREPROCESS ERROR:", str(e))

            print("❌ ROW:", example)

            return {}

    # --------------------------------------------------
    # APPLY PREPROCESSING
    # --------------------------------------------------

    dataset = dataset.map(

        preprocess,

        remove_columns=dataset.column_names
    )

    # --------------------------------------------------
    # REMOVE INVALID ROWS
    # --------------------------------------------------

    dataset = dataset.filter(

        lambda x: (
            x.get("input_features") is not None
            and len(x.get("input_features")) > 0
        )
    )

    # --------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------

    print(f"✅ FINAL DATASET SIZE: {len(dataset)}")

    if len(dataset) == 0:

        raise RuntimeError(
            "❌ No valid audio-text pairs found after preprocessing"
        )

    # --------------------------------------------------
    # DATA COLLATOR
    # --------------------------------------------------

    def data_collator(features):

        input_features = torch.tensor(
            [f["input_features"] for f in features]
        )

        labels = torch.tensor(
            [f["labels"] for f in features]
        )

        return {

            "input_features": input_features,

            "labels": labels
        }

    # --------------------------------------------------
    # TRAINING CONFIG
    # --------------------------------------------------

    training_args = TrainingArguments(

        output_dir=output_dir,

        per_device_train_batch_size=2,

        num_train_epochs=epochs,

        logging_steps=10,

        save_strategy="no",

        report_to="none",

        fp16=torch.cuda.is_available(),
    )

    # --------------------------------------------------
    # TRAINER
    # --------------------------------------------------

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=dataset,

        data_collator=data_collator,
    )

    print("🚀 [WHISPER TRAIN START]")

    trainer.train()

    # --------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------

    trainer.save_model(output_dir)

    processor.save_pretrained(output_dir)

    print(f"✅ [WHISPER TRAIN COMPLETE] → {output_dir}")

    # --------------------------------------------------
    # CLEANUP
    # --------------------------------------------------

    del trainer
    del model
    del processor
    del dataset

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        print("✅ GPU memory released")

    return output_dir
# ---------- LLM ----------
def train_llm(data_path, epochs, workspace_dir, model_type="tinyllama") -> str:
    import os
    import torch
    import zipfile      # ✅ ADD THIS
    import tempfile     # ✅ ADD THIS
    import shutil       # ✅ ADD THIS
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

    # ----------------------------------------
    # ✅ 1. AUTO-EXTRACT ZIP DATASETS (ADD THIS BLOCK)
    # ----------------------------------------
    cleanup_dataset_dir = None

    # If data_path is directly a ZIP file
    if os.path.isfile(data_path) and data_path.lower().endswith(".zip"):
        cleanup_dataset_dir = tempfile.mkdtemp(prefix="llm_data_")
        print(f"📦 Extracting dataset ZIP to {cleanup_dataset_dir}...")
        with zipfile.ZipFile(data_path, 'r') as zf:
            zf.extractall(cleanup_dataset_dir)
        data_path = cleanup_dataset_dir # Point to the extracted files

    # If data_path is a folder containing a ZIP file
    elif os.path.isdir(data_path):
        zip_files = [f for f in os.listdir(data_path) if f.lower().endswith(".zip")]
        if zip_files:
            cleanup_dataset_dir = tempfile.mkdtemp(prefix="llm_data_")
            zip_path = os.path.join(data_path, zip_files[0])
            print(f"📦 Extracting dataset ZIP {zip_path} to {cleanup_dataset_dir}...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(cleanup_dataset_dir)
            data_path = cleanup_dataset_dir # Point to the extracted files


    # ------------------------
    # MODEL SELECT
    # ------------------------
    from app.core.config import settings
    use_gpt2 = "gpt" in model_type
    model_name = "gpt2" if use_gpt2 else "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    revision = settings.hf_revision_gpt2 if use_gpt2 else settings.hf_revision_tinyllama

    # ------------------------
    # LOAD DATA (.txt/.csv files, recursive)
    # ------------------------
    def get_all_files(data_path, extensions):
        files_found = []
        for root, _, files in os.walk(data_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    files_found.append(os.path.join(root, file))
        return files_found
    txt_files = get_all_files(data_path, [".txt", ".csv"])

    if not txt_files:
        raise RuntimeError("❌ No TXT or CSV files found in dataset")

    # ----------------------------------------
    # ✅ VALIDATE FILE CONTENT
    # ----------------------------------------
    valid_files = []

    for f in txt_files:
        try:
            if os.path.getsize(f) > 0:
                valid_files.append(f)
        except Exception:
            continue

    if not valid_files:
        raise RuntimeError("❌ All dataset files are empty or invalid")

    txt_files = valid_files

    if any(f.endswith(".csv") for f in txt_files):
        dataset = load_dataset("csv", data_files={"train": txt_files})
    else:
        dataset = load_dataset("text", data_files={"train": txt_files})

    # ------------------------
    # TOKENIZER
    # ------------------------
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, local_files_only=True)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------
    # TOKENIZATION
    # ------------------------
    def tokenize(examples):
        keys = list(examples.keys())

        # ------------------------
        # DETECT INPUT / OUTPUT
        # ------------------------
        input_col = None
        output_col = None

        for k in keys:
            if "instruction" in k.lower() or "question" in k.lower():
                input_col = k
            if "output" in k.lower() or "answer" in k.lower():
                output_col = k

        # fallback (safe)
        if input_col is None:
            input_col = keys[0]
        if output_col is None:
            output_col = keys[-1]

        # ------------------------
        # FORMAT FOR TRAINING
        # ------------------------
        texts = [
            f"### Question:\n{q}\n\n### Answer:\n{a}"
            for q, a in zip(examples[input_col], examples[output_col])
        ]

        tokens = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=512
        )

        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    remove_cols = list(dataset["train"].column_names)
    dataset = dataset.map(tokenize, batched=True, num_proc=4, remove_columns=remove_cols)

    # ------------------------
    # MODEL
    # ------------------------
    # Load in fp32 — let TrainingArguments fp16/bf16 own the precision.
    # Loading in fp16 + fp16=True in TrainingArguments conflicts: AMP needs fp32
    # master weights internally, so the combination causes save_model() to only
    # write config/tokenizer files and silently skip the actual weights.
    use_cuda = torch.cuda.is_available()
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, local_files_only=True)
    except OSError:
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)

    # ------------------------
    # GPU precision detection
    # bf16 preferred on Ampere+ (3090, A100, etc); fp16 fallback for older GPUs
    # ------------------------
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = use_cuda and not use_bf16

    # ------------------------
    # OUTPUT DIR
    # ------------------------
    output_dir = os.path.join(workspace_dir, "models", "llm", model_type)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------
    # TRAINING CONFIG
    # ------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,   # effective batch size = 16
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_num_workers=4,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    )

    # ------------------------
    # TRAIN
    # ------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
    )

    print(f"🚀 [LLM TRAIN START] → {model_type}")
    trainer.train()

    # ------------------------
    # SAVE
    # ------------------------
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"✅ [LLM TRAIN COMPLETE] → {output_dir}")

    # ------------------------
    # RELEASE GPU MEMORY
    # ------------------------
    import gc
    del trainer, model, tokenizer, dataset
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()
        print("✅ [LLM] GPU memory released")

    # ----------------------------------------
    # ✅ 2. CLEANUP EXTRACTED ZIP (ADD THIS BLOCK)
    # ----------------------------------------
    if cleanup_dataset_dir and os.path.exists(cleanup_dataset_dir):
        shutil.rmtree(cleanup_dataset_dir, ignore_errors=True)
        print("🗑️ Cleaned up temporary extracted dataset.")

    return output_dir

# ---------- YOLO helpers ----------
import os
import random
import shutil
import yaml
import time
from pathlib import Path


def patch_yolo_data_yaml(yaml_path: str) -> None:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # 🔥 REMOVE unwanted 'path'
    if "path" in data:
        print(f"⚠️ Removing path: {data['path']}")
        data.pop("path")

    # 🔥 FIX train/val/test paths
    for key in ["train", "val", "test"]:
        if key in data:
            data[key] = os.path.normpath(data[key])

    # 🔥 FIX names (dict → list)
    if "names" in data and isinstance(data["names"], dict):
        data["names"] = [data["names"][k] for k in sorted(data["names"].keys())]

    print("🔥 FINAL NAMES:", data.get("names"))

    with open(yaml_path, "w") as f:
        yaml.dump(data, f)


# 🔥 DATASET VALIDATION
def validate_yolo_dataset(data_path: str):
    has_yaml = False
    has_images = False
    has_labels = False
    has_classes = False

    for root, dirs, files in os.walk(data_path):
        if "data.yaml" in files:
            has_yaml = True
        if "images" in dirs:
            has_images = True
        if "labels" in dirs:
            has_labels = True
        if "classes.txt" in files:
            has_classes = True

    if has_yaml:
        return True, "✅ Valid YOLO dataset (data.yaml found)"

    if has_images and has_labels and has_classes:
        return True, "✅ Valid custom dataset (images + labels + classes.txt)"

    return False, """
❌ Invalid dataset format

Supported formats:

1️⃣ YOLO format:
   train/
   valid/
   test/
   data.yaml

2️⃣ Custom format:
   images/
   labels/
   classes.txt
"""


def restructure_yolo_dataset(data_path: str) -> Path:
    images_dir = Path(data_path) / "images"
    labels_dir = Path(data_path) / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise ValueError("❌ images/ or labels/ folder missing")

    (images_dir / "train").mkdir(parents=True, exist_ok=True)
    (images_dir / "val").mkdir(parents=True, exist_ok=True)
    (labels_dir / "train").mkdir(parents=True, exist_ok=True)
    (labels_dir / "val").mkdir(parents=True, exist_ok=True)

    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    random.shuffle(image_files)

    split = int(len(image_files) * 0.8)

    for i, img in enumerate(image_files):
        lbl = labels_dir / (img.stem + ".txt")
        if not lbl.exists():
            continue

        if i < split:
            shutil.move(str(img), str(images_dir / "train" / img.name))
            shutil.move(str(lbl), str(labels_dir / "train" / lbl.name))
        else:
            shutil.move(str(img), str(images_dir / "val" / img.name))
            shutil.move(str(lbl), str(labels_dir / "val" / lbl.name))

    # 🔥 read classes.txt
    classes_file = Path(data_path) / "classes.txt"
    if not classes_file.exists():
        raise ValueError("❌ classes.txt not found")

    with open(classes_file, "r") as f:
        class_names = [line.strip() for line in f if line.strip()]

    print("🔥 USING CLASS NAMES:", class_names)

    yaml_path = Path(data_path) / "data.yaml"

    with open(yaml_path, "w") as f:
        yaml.dump({
            "train": "images/train",
            "val": "images/val",
            "nc": len(class_names),
            "names": class_names
        }, f)

    return yaml_path


# ---------- YOLO TRAIN ----------
def train_yolo(data_path, epochs, workspace_dir, model_type="yolov8n") -> str:
    import os
    from pathlib import Path as _Path
    from ultralytics import YOLO

    # 🔥 VALIDATE
    is_valid, msg = validate_yolo_dataset(data_path)
    print(msg)

    if not is_valid:
        raise ValueError(msg)

    size = model_type[-1] if model_type[-1] in ["n", "s", "m", "l", "x"] else "n"
    model_name = f"yolov8{size}.pt"

    # Cache setup
    yolo_config_dir = _Path(os.environ.get("YOLO_CONFIG_DIR", _Path.home() / ".config" / "Ultralytics"))
    yolo_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(yolo_config_dir)

    cached = yolo_config_dir / model_name
    model = YOLO(str(cached) if cached.exists() else model_name)

    # 🔥 DETECT DATASET TYPE
    yaml_path = None

    for root, dirs, files in os.walk(data_path):
        if "data.yaml" in files:
            yaml_path = os.path.join(root, "data.yaml")
            break

    if yaml_path:
        print("✅ Using existing YOLO dataset")
        yaml_path = os.path.abspath(yaml_path)

    else:
        print("✅ Using custom dataset")
        yaml_path = restructure_yolo_dataset(data_path)
        yaml_path = os.path.abspath(str(yaml_path))

    patch_yolo_data_yaml(str(yaml_path))

    # 🔥 TRAIN
    runs_dir = os.path.join(workspace_dir, "runs", "detect")

    model.train(
        data=str(yaml_path),
        epochs=epochs,
        project=runs_dir,
        name="train",
        exist_ok=True
    )

    best_pt = os.path.join(runs_dir, "train", "weights", "best.pt")

    if not os.path.exists(best_pt):
        raise RuntimeError("YOLO best.pt not found after training")

    # 🔥 DEBUG
    model = YOLO(best_pt)
    print("🔥 MODEL NAMES AFTER TRAIN:", model.names)

    # 🔥 SAVE
    dest_dir = os.path.join(workspace_dir, "models", "yolo")
    os.makedirs(dest_dir, exist_ok=True)

    dest_path = os.path.join(dest_dir, f"yolo_{int(time.time())}.pt")
    shutil.copy(best_pt, dest_path)

    shutil.rmtree(runs_dir, ignore_errors=True)

    print(f"✅ [YOLO MODEL SAVED] → {dest_path}")

    return dest_path

def train_vision_classifier(data_path: str, epochs: int, workspace_dir: str, model_type: str):
    import os
    import json
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from PIL import Image
    from torchvision import datasets, transforms, models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------------------
    # ✅ SAFE IMAGE LOADER (INLINE, NO PICKLE ISSUE)
    # ----------------------------------------
    def safe_loader(path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return Image.new("RGB", (224, 224))

    # ----------------------------------------
    # ✅ FIX DATASET STRUCTURE
    # ----------------------------------------
    def fix_dataset_structure(path):
        class_dirs = [
            d for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))
        ]

        if class_dirs:
            return path

        print("⚠️ Flat dataset → creating default class")
        new_dir = os.path.join(path, "default_class")
        os.makedirs(new_dir, exist_ok=True)

        for f in os.listdir(path):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                os.rename(os.path.join(path, f), os.path.join(new_dir, f))

        return path

    dataset_path = fix_dataset_structure(data_path)

    # ----------------------------------------
    # ✅ TRANSFORMS
    # ----------------------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # ----------------------------------------
    # ✅ LOAD DATASET (NO MULTIPROCESSING)
    # ----------------------------------------
    dataset = datasets.ImageFolder(
        dataset_path,
        transform=transform,
        loader=safe_loader
    )

    if len(dataset) == 0:
        raise RuntimeError("❌ No valid images found")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0   # 🔥 CRITICAL FIX
    )

    num_classes = len(dataset.classes)
    print("🔥 CLASSES:", dataset.classes)

    if num_classes < 2:
        raise ValueError("❌ Need at least 2 classes for classification")

    # ----------------------------------------
    # ✅ MODEL
    # ----------------------------------------
    if model_type == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_type == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_type == "mobilenet":
        model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model = model.to(device)

    # ----------------------------------------
    # ✅ TRAINING
    # ----------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print(f"🚀 [VISION TRAIN START] → {model_type}")

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")

    # ----------------------------------------
    # ✅ SAVE
    # ----------------------------------------
    output_dir = os.path.join(workspace_dir, "models", "vision", model_type)
    os.makedirs(output_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))

    with open(os.path.join(output_dir, "labels.json"), "w") as f:
        json.dump(dataset.classes, f)

    with open(os.path.join(output_dir, "model_config.json"), "w") as f:
        json.dump({"model_type": model_type}, f)

    print(f"✅ [VISION MODEL SAVED] → {output_dir}")

    # ----------------------------------------
    # CLEANUP
    # ----------------------------------------
    import gc
    del model, optimizer, dataloader, dataset
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_dir