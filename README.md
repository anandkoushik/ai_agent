# Antigravity AI Orchestrator

Antigravity AI is a highly advanced, orchestrator-driven machine learning assistant. It features a FastAPI backend with PostgreSQL-backed stateful memory, a DAG-based workflow engine, and an elegant React-based frontend.

## Architecture Highlights
- **FastAPI Backend:** Handles WebSockets, asynchronous workflow DAG execution, and dataset preprocessing.
- **Smart Intent Routing:** Uses `litellm` (via OpenAI, Gemini, etc.) to classify user intents (conversation vs. workflow) intelligently without triggering hardcoded paths.
- **React 19 Frontend:** A premium, dark-mode glassmorphic UI built with Vite, TailwindCSS v3.4, framer-motion, and lucide-react.
- **Resource Management:** GPU VRAM-aware model selection and task queuing.

## Setup Instructions

### 1. Backend Setup
Make sure you have PostgreSQL running locally if using the default URL.
```bash
python3 -m venv venv_mac
source venv_mac/bin/activate
pip install -r requirements.txt
pip install litellm  # Added for Smart Intent Routing
```

### 2. Run Backend
Ensure your `OPENAI_API_KEY` or `GEMINI_API_KEY` is exported for the smart intent router.
```bash
export OPENAI_API_KEY="your-key"
python chatbot.py
```
The backend runs on `http://localhost:8000`.

### 3. Frontend Setup & Run
Open a new terminal.
```bash
cd frontend
npm install
npm run dev
```
The frontend runs on `http://localhost:5173`.
