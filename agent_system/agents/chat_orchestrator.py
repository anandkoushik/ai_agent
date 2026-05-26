import logging
from typing import Dict, Any, List
import uuid

# Import the specific pipelines to wire up the orchestrator
from agent_system.uploads.manifest_builder import ManifestBuilder
from agent_system.core.model_selector import ModelSelector
from agent_system.adapters.training_adapter import TrainingAdapter
from agent_system.adapters.inference_adapter import InferenceAdapter

logger = logging.getLogger(__name__)

class ChatOrchestrator:
    """Dynamically generates DAGs from chat prompts and delegates strictly to Adapters."""
    
    @staticmethod
    async def process_workflow_intent(workspace_dir: str, target_inference_file: str = None, broadcast_callback=None) -> Dict[str, Any]:
        """
        Executes the auto-generation DAG:
        1. Analyze Dataset
        2. Select Model & Epochs
        3. Train (Via Adapter)
        4. Infer (Via Adapter)
        """
        workflow_id = str(uuid.uuid4())
        logger.info(f"[Orchestrator] Starting workflow {workflow_id}")
        
        # 1. Unified Dataset Manifest
        manifest = ManifestBuilder.build_manifest(workspace_dir)
        if manifest['dataset_type'] == 'unknown' or manifest['num_files'] == 0:
            return {"status": "error", "message": "No files found in the dataset. Please upload a valid zip containing files to train."}
            
        # 2. Smart Model Selection
        selected_model = ModelSelector.select_model(manifest, vram_gb=8.0)
        manifest['recommended_model'] = selected_model
        
        # 3. Dynamic Hyperparameter Auto-Selection
        from agent_system.tools.dataset_analyzer import DatasetAnalyzer
        from agent_system.core.hyperparameter_engine import HyperparameterEngine
        
        analyzer = DatasetAnalyzer(dataset_path=workspace_dir, dataset_type=manifest['dataset_type'])
        analysis_report = analyzer.analyze()
        
        hyperparameters = HyperparameterEngine.generate_training_plan(
            model_type=manifest['dataset_type'], 
            analysis_report=analysis_report
        )
        
        if broadcast_callback:
            msg = (
                f"Awesome! I've successfully analyzed your {manifest['dataset_type'].upper()} dataset. "
                f"Based on the dataset size ({analysis_report.get('total_size_mb', 0):.2f} MB), "
                f"I have automatically chosen the optimal training parameters:\n\n"
                f"• Model: {manifest['recommended_model']}\n"
                f"• Epochs: {hyperparameters['estimated_epochs']}\n"
                f"• Batch Size: {hyperparameters['batch_size']}\n"
                f"• Learning Rate: {hyperparameters['learning_rate']}\n\n"
                f"Commencing training now!"
            )
            await broadcast_callback(msg)
            
        # 4. Training execution via Adapter (Strict ML logic isolation)
        logger.info("[Orchestrator] Handing off to TrainingAdapter")
        train_result = await TrainingAdapter.run_training(
            workflow_id=workflow_id,
            model_type=manifest['dataset_type'],
            dataset_path=workspace_dir,
            hyperparameters=hyperparameters
        )
        
        if train_result.get("status") != "success":
            return {"status": "error", "message": f"Training failed: {train_result.get('message')}"}
            
        # 5. Inference execution (if requested)
        infer_result = {}
        if target_inference_file:
            logger.info("[Orchestrator] Handing off to InferenceAdapter")
            infer_result = await InferenceAdapter.run_inference(
                workflow_id=workflow_id,
                model_type=manifest['dataset_type'],
                target_file=target_inference_file,
                model_path="trained_model.pt"
            )
            
        # Workflow Completed (No hallucinated successes allowed here)
        return {
            "status": "success",
            "workflow_id": workflow_id,
            "manifest": manifest,
            "hyperparameters": hyperparameters,
            "training_result": train_result,
            "inference_result": infer_result
        }
