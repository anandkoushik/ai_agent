import os

content = '''            else:
                # Use existing active dataset
                workspace_dir = active_dataset_path
                
            # Modality routing & selection
            await manager.broadcast("Analyzing dataset modality...")
            
            # 3b. Trigger Orchestrator
            await manager.broadcast("Triggering Chat Orchestrator DAG...")
            result = await ChatOrchestrator.execute_workflow(
                workflow_id=str(uuid.uuid4()), 
                workspace_dir=workspace_dir, 
                target_inference_file=inference_target.filename if inference_target else None
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
'''

with open('chatbot.py', 'a', encoding='utf-8') as f:
    f.write('\n' + content)
