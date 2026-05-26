# Antigravity AI Frontend

This is the frontend for the Antigravity AI Orchestrator. It is built with:
- **React 19**
- **Vite**
- **TailwindCSS 3.4**
- **Framer Motion** (for micro-animations)
- **Lucide React** (for icons)

## Design System
The UI utilizes a **premium dark-mode glassmorphic design**. 
- Deep purple and neon blue glowing effects via `.glow-text`.
- `.glass-panel` backdrop blurs for message bubbles and headers.
- Interactive file attachments with seamless hover states.

## Running Locally

To install dependencies:
```bash
npm install
```

To run the development server:
```bash
npm run dev
```

The frontend will attempt to connect to the backend WebSocket at `ws://localhost:8000/ws/chat`. Make sure the backend server (`chatbot.py`) is running simultaneously.
