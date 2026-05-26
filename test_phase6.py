import logging
from agent_system.tools.dataset_analyzer import DatasetAnalyzer
from agent_system.core.hyperparameter_engine import HyperparameterEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_smart_auto_selection():
    logger.info("--- Testing Phase 6: Dataset Analysis & Smart Auto-Selection ---")
    
    # 1. Mock Dataset Analysis
    analyzer = DatasetAnalyzer(dataset_path="./mock_dataset", dataset_type="yolo")
    
    try:
        report = analyzer.analyze()
        logger.info(f"Analysis Report Generated: {report}")
        
        # 2. Smart Auto-Selection
        logger.info("Generating YOLO Training Plan based on strict Config boundaries...")
        plan = HyperparameterEngine.generate_training_plan(model_type="yolo", analysis_report=report)
        
        logger.info(f"Final Smart Training Plan: {plan}")
        
    except Exception as e:
        logger.error(f"Workflow failed-fast due to: {e}")

if __name__ == "__main__":
    test_smart_auto_selection()
