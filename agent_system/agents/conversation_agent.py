import logging
import os
from agent_system.config import Config
from agent_system.rag.general_knowledge import GeneralKnowledgeRAG
from agent_system.memory.session_manager import SessionManager
try:
    import litellm
except ImportError:
    litellm = None

logger = logging.getLogger(__name__)

class ConversationAgent:
    """Expert ML guidance agent integrating Config metrics, Hybrid RAG, and LLM conversations."""
    
    @staticmethod
    def _format_epoch_recommendations() -> str:
        """Dynamically pulls configured epoch ranges from config.py to prevent hallucinations."""
        return f"""
**Practical Hyperparameter Guidance:**
Epoch selection depends mainly on dataset size, quality, and model type.

General recommendations from the configured training heuristics:

YOLO:
- <{Config.YOLO_EPOCHS['small_mb']}MB datasets → {Config.YOLO_EPOCHS['small_range'][0]}-{Config.YOLO_EPOCHS['small_range'][1]} epochs
- {Config.YOLO_EPOCHS['small_mb']}MB-{Config.YOLO_EPOCHS['medium_mb']}MB → {Config.YOLO_EPOCHS['medium_range'][0]}-{Config.YOLO_EPOCHS['medium_range'][1]} epochs
- >{Config.YOLO_EPOCHS['medium_mb']}MB → {Config.YOLO_EPOCHS['large_range'][0]}-{Config.YOLO_EPOCHS['large_range'][1]} epochs

Whisper / Vision / LLM:
- <{Config.WHISPER_EPOCHS['small_mb']}MB → {Config.WHISPER_EPOCHS['small_range'][0]}-{Config.WHISPER_EPOCHS['small_range'][1]} epochs
- >1GB → {Config.WHISPER_EPOCHS['large_range'][0]}-{Config.WHISPER_EPOCHS['large_range'][1]} epochs
"""

    @staticmethod
    async def generate_response(prompt: str, intent_data: dict, session_id: str = "default-session") -> str:
        """
        Retrieval-First Answering based on Semantic Intent, synthesized via LLM for natural conversation.
        """
        logger.info(f"[ConversationAgent] Generating response for intent: {intent_data['primary_intent']}")
        prompt_lower = prompt.lower()
        
        intents = intent_data.get("intents", [])
        primary_intent = intent_data.get("primary_intent", "conversation")
        context_parts = []
        
        # 1. Selective RAG (Only for explicit documentation/specific queries)
        if primary_intent == "rag_document_query" or "rag_document_query" in intents:
            rag_data = GeneralKnowledgeRAG.query_concept(prompt)
            if rag_data["synthesized"]:
                concept_text = []
                for snippet in rag_data["snippets"]:
                    concept_text.append(f"{snippet['text']}\n(Source: {snippet['source']} | Confidence: {snippet['score']*100:.0f}%)")
                context_parts.append("EXTERNAL RAG KNOWLEDGE:\n" + "\n\n".join(concept_text))
            else:
                context_parts.append("EXTERNAL RAG KNOWLEDGE: No external documentation found for this query.")
                
        # 2. Practical Hyperparameter Guidance (Config Append)
        if "hyperparameter_guidance" in intents:
            if "epoch" in prompt_lower:
                context_parts.append(ConversationAgent._format_epoch_recommendations())
            else:
                context_parts.append("SYSTEM RULE: For optimal performance, I automatically adjust parameters like batch size based on available VRAM.")
                
        # 3. Troubleshooting
        if "troubleshooting" in intents:
            context_parts.append("SYSTEM RULE: Ensure dataset ZIP is correctly formatted and GPU has enough VRAM.")
            
        # Get Session Memory
        session = await SessionManager.get_session(session_id)
        chat_history = session.get("chat_history", [])[-10:] # Keep last 10 messages (5 turns)
        
        # Use LLM to synthesize if available
        if litellm:
            try:
                model = Config.LLM_MODEL
                system_prompt = (
                    "You are Antigravity AI, a highly skilled Machine Learning Engineering Assistant. "
                    "CRITICAL RULE: You MUST ONLY answer questions related to AI, Machine Learning, Deep Learning, and Python programming. "
                    "If the user asks about ANY other topic (e.g., history, geography, general chit-chat, other languages), you MUST politely refuse and guide them back to AI/ML or Python topics. "
                    "Do NOT be overly robotic. Avoid repetitive phrases like 'As an AI'. "
                    "Be concise for simple chat, but highly technical when discussing ML concepts. "
                    "Your primary expertise is training ML models (you support YOLO, Whisper, and LLMs), automatically selecting hyperparameters, and running highly efficient inference. "
                    "Always emphasize that you are highly compatible and built specifically for robust ML model training and running inference."
                )
                
                if context_parts:
                    system_prompt += "\n\nUse the following technical context to ground your answer. NEVER hallucinate parameters that contradict the SYSTEM RULES.\n\n"
                    system_prompt += "\n\n".join(context_parts)
                else:
                    system_prompt += "\n\nRespond naturally. The user is just chatting or asking a general question."

                messages = [{"role": "system", "content": system_prompt}]
                
                # Append History
                for msg in chat_history:
                    messages.append(msg)
                    
                messages.append({"role": "user", "content": prompt})

                response = litellm.completion(
                    model=model,
                    messages=messages,
                    temperature=0.6,
                    max_tokens=1024
                )
                
                ai_response = response.choices[0].message.content
                
                # Update Session Memory
                chat_history.append({"role": "user", "content": prompt})
                chat_history.append({"role": "assistant", "content": ai_response})
                await SessionManager.update_session(session_id, "chat_history", chat_history)
                
                return ai_response
                
            except litellm.exceptions.AuthenticationError:
                logger.error("LLM Authentication failed. API key missing or invalid.")
                return (
                    "**Oops! I'm missing my API key.** 😅\n\n"
                    "To enable my AI-generated responses (via Groq), please create a `.env` file in the root directory and add your API key like this:\n\n"
                    "`GROQ_API_KEY=your_key_here`\n\n"
                    "Don't worry though—your Machine Learning workflows and training pipelines will still work perfectly without it!"
                )
            except Exception as e:
                logger.error(f"LLM Response synthesis failed: {str(e)}")
                return f"I encountered a temporary issue generating a response ({str(e)}). However, your ML pipelines are fully operational!"
        
        return "LiteLLM is not installed. Please install it or configure your environment to enable AI chat."
