"""
Dashboard chatbot powered by Phi-2
+ Smart training guidance (dataset + format + epochs based on size)
+ AI Agent: File Ingestion & Dataset Understanding (Phase 1)
"""

import os
import zipfile
import json
from typing import Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# =====================================================
# AI AGENT: PHASE 1 - INGESTION & DATASET UNDERSTANDING
# =====================================================
class FileIngestion:
    @staticmethod
    def safe_extract_zip(zip_path: str, extract_to: str, max_size_bytes=5 * 1024 * 1024 * 1024):
        """Safely extract ZIP preventing path traversal and size bombs."""
        from zip_parser import ZipParser
        return ZipParser.extract_and_parse(zip_path, extract_to, max_size_bytes)

    @staticmethod
    def parse_csv(file_path: str) -> Dict[str, Any]:
        """Modular parser for CSV"""
        from csv_parser import CSVParser
        return CSVParser.parse(file_path)

    @staticmethod
    def parse_audio(file_path: str) -> Dict[str, Any]:
        """Modular parser for Audio"""
        from audio_parser import AudioParser
        return AudioParser.parse(file_path)

    @staticmethod
    def parse_image(file_path: str) -> Dict[str, Any]:
        """Modular parser for Image"""
        from image_parser import ImageParser
        return ImageParser.parse(file_path)

    @staticmethod
    def parse_pdf(file_path: str) -> Dict[str, Any]:
        """Modular parser for PDF"""
        from pdf_parser import PDFParser
        return PDFParser.parse(file_path)

    @staticmethod
    def parse_text(file_path: str) -> str:
        """Modular parser for Text/PDF"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read(2000)

class DatasetUnderstanding:
    @staticmethod
    def analyze_workspace(workspace_dir: str) -> Dict[str, Any]:
        """Automatically detect dataset type and validate quality."""
        analysis = {"type": "unknown", "valid": False, "details": {}}
        files = []
        for root, dirs, filenames in os.walk(workspace_dir):
            for f in filenames:
                files.append(os.path.join(root, f))
        
        # Detect YOLO
        if any('images' in p for p in files) and any('labels' in p for p in files):
            analysis["type"] = "yolo"
            img_count = sum(1 for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg')))
            lbl_count = sum(1 for f in files if f.lower().endswith('.txt') and 'labels' in f)
            analysis["valid"] = img_count > 0 and img_count == lbl_count
            analysis["details"] = {"images": img_count, "labels": lbl_count}
            return analysis
            
        # Detect Whisper
        if any(f.lower().endswith('.wav') for f in files) and any(f.lower().endswith('.csv') for f in files):
            analysis["type"] = "whisper"
            wav_count = sum(1 for f in files if f.lower().endswith('.wav'))
            analysis["valid"] = wav_count > 0
            analysis["details"] = {"audio_files": wav_count}
            return analysis
            
        # Detect LLM
        if any(f.lower().endswith('.txt') or f.lower().endswith('.csv') for f in files):
            analysis["type"] = "llm"
            analysis["valid"] = True
            analysis["details"] = {"text_files": len(files)}
            return analysis
            
        return analysis


# =====================================================
# AI AGENT: PHASE 2 - MEMORY, RAG & TOOL REGISTRY
# =====================================================
import traceback

class AgentMemory:
    def __init__(self, db_url="postgresql://postgres@localhost/agent_db"):
        self.short_term = []
        self.db_url = db_url
        self._init_db()

    def _init_db(self):
        try:
            import psycopg2
            self.conn = psycopg2.connect(self.db_url)
            with self.conn.cursor() as cur:
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS long_term_memory (
                        id SERIAL PRIMARY KEY,
                        role VARCHAR(50),
                        content TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
            self.conn.commit()
            print("PostgreSQL Long-Term Memory Initialized.")
        except Exception as e:
            print(f"PostgreSQL Init failed: {e}. Falling back to in-memory.")
            self.conn = None

    def add_interaction(self, role: str, content: str):
        self.short_term.append({"role": role, "content": content})
        if len(self.short_term) > 20:
            self.short_term.pop(0)
            
        if self.conn:
            try:
                with self.conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO long_term_memory (role, content) VALUES (%s, %s)",
                        (role, content)
                    )
                self.conn.commit()
            except Exception as e:
                print(f"Failed to save to long term memory: {e}")

    def get_long_term_context(self, limit=5):
        if not self.conn:
            return []
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT role, content FROM long_term_memory ORDER BY timestamp DESC LIMIT %s", (limit,))
                rows = cur.fetchall()
                return [{"role": row[0], "content": row[1]} for row in reversed(rows)]
        except Exception:
            return []

class RAGSystem:
    def __init__(self, db_path="./rag_db"):
        self.db_path = db_path
        self.client = None
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_or_create_collection("agent_knowledge")
            self.encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        except ImportError:
            print("RAG disabled: chromadb or sentence_transformers not installed.")
        except Exception as e:
            print(f"RAG init failed: {e}")

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50):
        """Recursive basic chunking for RAG."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
        return chunks

    def add_document(self, doc_id: str, text: str):
        if not self.client: return
        chunks = self._chunk_text(text)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            embeddings = self.encoder.encode([chunk]).tolist()
            self.collection.add(ids=[chunk_id], embeddings=embeddings, documents=[chunk])
        print(f"Added {len(chunks)} chunks to RAG for document {doc_id}")

    def search(self, query: str, k=2) -> list[str]:
        if not self.client: return []
        emb = self.encoder.encode([query]).tolist()
        results = self.collection.query(query_embeddings=emb, n_results=k)
        return results['documents'][0] if results['documents'] else []

class ToolRegistry:
    def __init__(self):
        self.tools = {}
        self._auto_register()

    def _auto_register(self):
        # Dynamically map backend functions
        try:
            import training
            self.tools["train_whisper"] = self._safe_wrap(training.train_whisper)
            self.tools["train_llm"] = self._safe_wrap(training.train_llm)
        except ImportError:
            pass
            
        try:
            import inference
            self.tools["make_model_zip"] = self._safe_wrap(inference.make_model_zip)
        except ImportError:
            pass
            
        self.tools["analyze_dataset"] = self._safe_wrap(DatasetUnderstanding.analyze_workspace)

    def _safe_wrap(self, func):
        def wrapper(*args, **kwargs):
            try:
                return {"status": "success", "result": func(*args, **kwargs)}
            except Exception as e:
                return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}
        return wrapper

    def execute(self, tool_name: str, **kwargs):
        if tool_name not in self.tools or self.tools[tool_name] is None:
            return {"status": "error", "error": f"Tool '{tool_name}' not found in registry."}
    def get_registered_tools(self):
        return list(self.tools.keys())

class WorkflowEngine:
    def __init__(self, tool_registry: ToolRegistry):
        self.registry = tool_registry

    def execute_workflow(self, workflow_steps: list) -> dict:
        """
        Executes a series of tools sequentially.
        workflow_steps format:
        [
            {"tool": "analyze_dataset", "kwargs": {"workspace_dir": "./data"}},
            {"tool": "train_whisper", "kwargs": {"epochs": 10}}
        ]
        """
        results = []
        for step_idx, step in enumerate(workflow_steps):
            tool_name = step.get("tool")
            kwargs = step.get("kwargs", {})
            print(f"Executing workflow step {step_idx + 1}: {tool_name}")
            
            result = self.registry.execute(tool_name, **kwargs)
            results.append({"step": step_idx + 1, "tool": tool_name, "result": result})
            
            if result.get("status") == "error":
                print(f"Workflow aborted at step {step_idx + 1} due to error.")
                return {"status": "failed", "completed_steps": results}
                
        return {"status": "success", "completed_steps": results}

# =====================================================
# INTENT DETECTION
# =====================================================
def is_dashboard_question(text: str) -> bool:
    keywords = [
        "dashboard", "train", "training", "inference", "upload",
        "dataset", "label", "labeling", "label studio", "auto label",
        "job status", "epoch", "epochs", "model select",
        "where", "how", "steps", "section"
    ]
    return any(k in text.lower() for k in keywords)


def is_training_question(text: str) -> bool:
    keywords = [
        "dataset", "format", "download", "data", "training data", "epochs"
    ]
    return any(k in text.lower() for k in keywords)


# =====================================================
# PROMPTS
# =====================================================
GENERAL_PROMPT = """
Answer the question clearly and simply.
Do NOT mention dashboards, UI, steps, or software.
Keep the answer short and understandable.
"""

DASHBOARD_CONTEXT = """
Lyra AI Dashboard sections:
- Train Your Model
- Run Inference
- Dataset Utilities
- Labeling (Auto Label YOLO, Label Studio)
- Job Status

YOLO:
- Upload ZIP with images/ and labels/
- Epochs usually 20–50 (can increase based on dataset)

WHISPER:
- Upload audio in Run Inference
- Select tiny / base / small model
"""

PDF_CONTEXT = """
PDF covers:
- YOLO dataset, labeling, training, inference
- Whisper usage
"""

DASHBOARD_PROMPT = f"""
You are an Lyra AI Dashboard assistant.

Rules:
- Answer ONLY in terms of dashboard usage
- Mention exact sections and fields
- Give practical recommendations
- Do NOT explain ML algorithms

Dashboard reference:
{DASHBOARD_CONTEXT}

PDF reference:
{PDF_CONTEXT}
"""


# =====================================================
# TRAINING GUIDANCE (FINAL 🔥)
# =====================================================
def get_training_guidance(question: str) -> str:
    q = question.lower()

    # ---------------- YOLO ----------------
    if "yolo" in q:
        return """
You can create your own dataset or download from Roboflow or Hugging Face.

Go to:
- Roboflow (recommended)
- Hugging Face (datasets / spaces)

Export settings:
- Format: YOLOv8 OBB
- Structure:
    images/
    labels/

Epochs (based on dataset size):

< 50 MB → 80–100
50 MB – 500 MB → 50–80
> 500 MB → 30–50

🔗 [Roboflow](https://roboflow.com/)
🔗 [Hugging Face Datasets](https://huggingface.co/datasets)
"""

    # ---------------- WHISPER ----------------
    elif "whisper" in q or "audio" in q:
        return """
You can create your own dataset or download from Kaggle.

Go to Kaggle and search audio dataset.

Required format:
- .wav files
- .csv transcription

Epochs (based on dataset size):

< 100 MB → 50–100
100 MB – 1 GB → 20–70
> 1 GB → 10–25

🔗 [Kaggle Datasets](https://www.kaggle.com/datasets)
"""

    # ---------------- VISION ----------------
    elif "resnet" in q or "mobilenet" in q or "vision" in q:
        return """
You can create your own dataset or download from Kaggle.

Go to Kaggle and search image dataset.

Required format:
- train/
    - class1/
    - class2/

Epochs (based on dataset size):

< 100 MB → 50–100
100 MB – 1 GB → 20–70
> 1 GB → 10–25

🔗 [Kaggle Datasets](https://www.kaggle.com/datasets)
"""

    # ---------------- LLM ----------------
    elif "llm" in q or "text" in q:
        return """
You can create your own dataset or download from Kaggle.

Go to Kaggle and search text dataset.

Required format:
- .txt or .csv

Epochs (based on dataset size):

< 100 MB → 50–100
100 MB – 1 GB → 20–70
> 1 GB → 10–25

🔗 [Kaggle Datasets](https://www.kaggle.com/datasets)
"""

    return """
Please specify model type:

- YOLO
- Whisper
- Vision (ResNet / MobileNet)
- LLM
"""


# =====================================================
# MAIN FUNCTION
# =====================================================
def ask_dashboard_bot(question: str) -> str:
    import gc
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from app.core.config import settings

    # ✅ STEP 1: HANDLE TRAINING QUESTIONS FIRST
    if is_training_question(question):
        return get_training_guidance(question)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "microsoft/phi-2"
    revision = settings.hf_revision_phi2

    # Load model
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            local_files_only=True,
        ).to(device)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)

    model.eval()

    try:
        if is_dashboard_question(question):
            prompt = f"""
{DASHBOARD_PROMPT}

User question:
{question}

Answer (dashboard only):
"""
        else:
            prompt = f"""
{GENERAL_PROMPT}

User question:
{question}

Answer:
"""

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)

        if "Answer (dashboard only):" in decoded:
            reply = decoded.split("Answer (dashboard only):")[-1].strip()
        elif "Answer:" in decoded:
            reply = decoded.split("Answer:")[-1].strip()
        else:
            reply = decoded.strip()

        return reply

    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =====================================================
# TESTING BLOCK (SERVER-SIDE TESTING)
# =====================================================
if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI, UploadFile, File
    from fastapi.middleware.cors import CORSMiddleware
    import shutil
    import tempfile

    app = FastAPI(title="AI Agent Testing Server")

    # --- CORS: Allow all origins for local dev/testing ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Old v1 test endpoints removed to clean up Swagger UI

    from fastapi import Form
    @app.post("/test/phase2_rag")
    async def test_phase2_rag(file: UploadFile = File(...), query: str = Form(...)):
        """Test endpoint for Modular Phase 2: RAG System"""
        from agent_system.rag.engine import RAGSystem
        from agent_system.rag.debug_utils import RAGDebugger
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
            
        rag = RAGSystem(db_path="./agent_system/rag_db")
        rag.ingest(doc_id=file.filename, filepath=temp_path)
        results = rag.retrieve(query=query, k=2)
        
        # Validate chunks and generate visualization for debugging
        for res in results:
            res["is_coherent"] = RAGDebugger.validate_chunk_coherence(res.get("metadata", {}))
            
        debug_visualization = RAGDebugger.visualize_retrieval(query, results)
        
        try:
            os.remove(temp_path)
        except:
            pass
            
        return {
            "status": "success", 
            "query": query, 
            "debug_visualization": debug_visualization,
            "retrieved_chunks": results
        }

    @app.post("/test/phase3_tools")
    async def test_phase3_tools(tool_name: str, kwargs_json: str = "{}"):
        """Test endpoint for Phase 3: Async Tool Registry"""
        import json
        from agent_system.tools.registry import ToolRegistry
        
        registry = ToolRegistry()
        try:
            kwargs = json.loads(kwargs_json)
        except Exception:
            kwargs = {}
            
        result = await registry.execute(tool_name, **kwargs)
        return {"status": "success", "tool_execution_result": result}

    @app.post("/test/phase4_workflow")
    async def test_phase4_workflow(job_id: str, steps_json: str):
        """Test endpoint for Phase 4: Workflow Engine & State"""
        import json
        from agent_system.core.engine import WorkflowEngine
        from agent_system.tools.registry import ToolRegistry
        import dataclasses
        
        registry = ToolRegistry()
        engine = WorkflowEngine(registry)
        
        try:
            steps = json.loads(steps_json)
        except Exception:
            return {"status": "error", "message": "Invalid JSON steps"}
            
        state = await engine.execute_workflow(job_id=job_id, steps=steps)
        return {"status": "success", "final_state": dataclasses.asdict(state)}

    @app.post("/test/phase5_planner")
    async def test_phase5_planner(user_query: str):
        """Test endpoint for Phase 5: Planning Engine"""
        from agent_system.agents.planner import PlanningEngine
        from agent_system.tools.registry import ToolRegistry
        
        registry = ToolRegistry()
        planner = PlanningEngine(available_tools=registry.get_registered_tools())
        
        plan = planner.generate_plan(user_query)
        
        return {"status": "success", "user_query": user_query, "generated_plan": plan}

    @app.post("/test/phase6_dataset_analysis")
    async def test_phase6_dataset_analysis(file: UploadFile = File(...)):
        """Test endpoint for Phase 6: Deep Dataset Analysis & Smart Auto-Selection (Upload File)"""
        import tempfile
        import shutil
        import os
        from agent_system.tools.dataset_analyzer import DatasetAnalyzer
        from agent_system.core.hyperparameter_engine import HyperparameterEngine
        
        # 1. Safely extract uploaded zip
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
            shutil.copyfileobj(file.file, tmp_zip)
            zip_path = tmp_zip.name
            
        extract_dir = tempfile.mkdtemp()
        
        try:
            # We assume it's a zip file for dataset upload
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                
            # 2. Auto-Detect Model Type using Phase 1 Logic
            detection = DatasetUnderstanding.analyze_workspace(extract_dir)
            model_type = detection.get("type", "unknown")
            
            if model_type == "unknown":
                return {"status": "error", "message": "Could not automatically detect dataset format. Ensure it's a valid YOLO, Whisper, Vision, or LLM dataset format."}
            
            # 3. Analyze Dataset Size and Quality
            analyzer = DatasetAnalyzer(dataset_path=extract_dir, dataset_type=model_type)
            report = analyzer.analyze()
            
            # 4. Generate Smart Training Plan
            plan = HyperparameterEngine.generate_training_plan(model_type=model_type, analysis_report=report)
            
            # 5. Map Training and Inference functions
            train_func_map = {
                "yolo": "train_yolo()",
                "whisper": "train_whisper()",
                "vision": "train_vision()",
                "llm": "train_llm()"
            }
            infer_func_map = {
                "yolo": "run_yolo_inference()",
                "whisper": "run_whisper_inference()",
                "vision": "run_vision_inference()",
                "llm": "run_llm_inference()"
            }
            
            plan["training_function"] = f"training.py -> {train_func_map.get(model_type, 'unknown_train()')}"
            plan["inference_function"] = f"inference.py -> {infer_func_map.get(model_type, 'unknown_inference()')}"
            
            # 6. Format Output Sentence
            epochs = plan.get('estimated_epochs')
            imbalance = report.get('imbalance_ratio', 1.0)
            
            sentence = f"We successfully detected a {model_type.upper()} dataset structure! "
            sentence += f"Based on the dataset size ({report.get('total_size_mb'):.2f} MB), we have intelligently selected {epochs} training epochs. "
            if imbalance > 5.0:
                sentence += f"Additionally, since we noticed a severe class imbalance ({imbalance}x), we automatically reduced the learning rate to {plan.get('learning_rate')} and applied {plan.get('imbalance_handling')} to protect minority classes. "
                
            sentence += f"The workflow has automatically selected '{plan['training_function']}' to begin training. Once completed, user inputs will be routed through '{plan['inference_function']}' to generate predictions!"

            
            return {
                "status": "success",
                "detected_model": model_type,
                "summary": sentence,
                "analysis_report": report,
                "recommended_training_plan": plan
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
        finally:
            # Cleanup
            if os.path.exists(zip_path):
                os.remove(zip_path)
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir, ignore_errors=True)

    @app.post("/test/phase7_selfheal")
    async def test_phase6_selfheal(tool_name: str, kwargs_json: str = "{}", simulate_error: bool = False):
        """
        Test endpoint for Phase 6: Observation & Self-Healing.
        Set simulate_error=true to inject a fake CUDA OOM error and observe auto-remediation.
        """
        import json
        import dataclasses
        from agent_system.tools.registry import ToolRegistry
        from agent_system.core.observation import ObservationLoop
        from agent_system.core.state import WorkflowState

        registry = ToolRegistry()

        # Optionally patch the tool to simulate failure for testing
        if simulate_error:
            from agent_system.tools.wrappers import _strict_schema
            async def _fake_fail(**kw):
                return _strict_schema(1, "", "CUDA out of memory: tried to allocate 2.00 GiB", {})
            registry.tools[tool_name] = _fake_fail

        observation_loop = ObservationLoop(registry, max_retries=2)
        state = WorkflowState(job_id="phase6-test")
        state.mark_running()

        try:
            kwargs = json.loads(kwargs_json)
        except Exception:
            kwargs = {}

        result = await observation_loop.observe_and_heal(
            tool_name=tool_name, kwargs=kwargs, state=state, step_idx=1
        )
        state.add_history(1, tool_name, result)

        return {
            "status": "success",
            "final_result": result,
            "state_history": dataclasses.asdict(state)["history"]
        }

    # =====================================================
    # PHASE 7: UNIFIED CONVERSATIONAL ORCHESTRATOR
    # =====================================================
    from fastapi import WebSocket, WebSocketDisconnect
    
    # Simple Event Bus for WebSockets
    class ConnectionManager:
        def __init__(self):
            self.active_connections: list[WebSocket] = []

        async def connect(self, websocket: WebSocket):
            await websocket.accept()
            self.active_connections.append(websocket)

        def disconnect(self, websocket: WebSocket):
            self.active_connections.remove(websocket)

        async def broadcast(self, message: str):
            for connection in self.active_connections:
                await connection.send_text(message)

    manager = ConnectionManager()

    @app.websocket("/ws/chat")
    async def websocket_endpoint(websocket: WebSocket):
        await manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                await manager.broadcast(f"Echo: {data}")
        except WebSocketDisconnect:
            manager.disconnect(websocket)

    from typing import Optional
    @app.post("/chat/workflow")
    async def unified_chat_workflow(
        prompt: str = Form(...),
        session_id: str = Form("default-session"),
        dataset: Optional[UploadFile] = File(None),
        inference_target: Optional[UploadFile] = File(None)
    ):
        """Phase 7.0 Unified Entrypoint"""
        import tempfile
        import shutil
        import os
        from agent_system.agents.router import HybridRouter
        from agent_system.agents.conversation_agent import ConversationAgent
        from agent_system.agents.chat_orchestrator import ChatOrchestrator
        from agent_system.memory.session_manager import SessionManager
        
        print(f"Received unified workflow request from session {session_id}")
        print(f"DEBUG: prompt={prompt}, dataset={dataset}, inference_target={inference_target}")
        if dataset:
            print(f"DEBUG: dataset.filename={dataset.filename}")
        
        # TTL Cleanup
        await SessionManager.cleanup_expired()
        session_state = await SessionManager.get_session(session_id)
        
        # 1. Hybrid Intent Routing & Session Context
        files_present = []
        if dataset and dataset.filename:
            files_present.append(dataset.filename)
        if inference_target and inference_target.filename:
            files_present.append(inference_target.filename)
        
        # Check session memory for existing artifacts
        active_dataset_path = session_state.get("active_dataset_path")
        if active_dataset_path and not dataset:
            print(f"Using active dataset from session: {active_dataset_path}")
            files_present.append("session_dataset.zip")
            
        intent_data = HybridRouter.classify_intent(prompt, files_present)
        primary_intent = intent_data["primary_intent"]
        cleaned_prompt = intent_data["cleaned_prompt"]
        
        # 2. Pure Conversation / General Knowledge Handling
        if primary_intent in ["conversation", "concept_explanation", "hyperparameter_guidance", "troubleshooting", "multi_intent"]:
            response = await ConversationAgent.generate_response(cleaned_prompt, intent_data, session_id)
            return {"intent": primary_intent, "response": response}
            
        # 3. Workflow / Hybrid Handling
        if not dataset and not active_dataset_path:
            return {"intent": primary_intent, "error": "A dataset ZIP is required to trigger a workflow."}
            
        workspace_dir = None
        try:
            # Create persistent workspace for session instead of temp
            os.makedirs(f"workspaces/{session_id}", exist_ok=True)
            workspace_dir = f"workspaces/{session_id}/dataset"
            
            # If new dataset uploaded, extract and update memory
            if dataset:
                # Clear previous dataset to avoid modality pollution
                if os.path.exists(workspace_dir):
                    shutil.rmtree(workspace_dir)
                os.makedirs(workspace_dir, exist_ok=True)
                
                zip_path = os.path.join(workspace_dir, "upload.zip")
                with open(zip_path, "wb") as buffer:
                    shutil.copyfileobj(dataset.file, buffer)
                await manager.broadcast(f"Extracting new dataset for session {session_id}...")
                shutil.unpack_archive(zip_path, workspace_dir)
                await SessionManager.update_session(session_id, "active_dataset_path", workspace_dir)
            else:
                # Use existing active dataset
                workspace_dir = active_dataset_path
                
            # Modality routing & selection
            await manager.broadcast("Analyzing dataset modality...")
            
            # 3b. Trigger Orchestrator
            await manager.broadcast("Triggering Chat Orchestrator DAG...")
            result = await ChatOrchestrator.process_workflow_intent(
                workspace_dir=workspace_dir, 
                target_inference_file=inference_target.filename if inference_target else None,
                broadcast_callback=manager.broadcast
            )
            
            # Wait for execution and optionally hybrid response
            if primary_intent == "hybrid":
                # Combine ML insight with Workflow execution
                insight = await ConversationAgent.generate_response(cleaned_prompt, intent_data)
                result["hybrid_insight"] = insight
                
            return {"intent": primary_intent, "workflow_response": result}
            
        except Exception as e:
            await manager.broadcast(f"Workflow error: {str(e)}")
            return {"intent": primary_intent, "error": str(e)}
        finally:
            # Cleanup
            if 'zip_path' in locals() and os.path.exists(zip_path):
                os.remove(zip_path)

if __name__ == "__main__":
    print("Starting Test Server for AI Agent Integration on port 8000...")
    print("Test all phases at: http://localhost:8000/docs")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
