import json
import numpy as np
import os
import mlflow
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init():
    """Initialize the scoring environment and load the model."""
    global model
    
    try:
        # Get the path to the deployed model directory
        model_root = Path(os.getenv('AZUREML_MODEL_DIR', ''))
        model_path = next(model_root.iterdir())
        
        logger.info(f"Loading model from {model_path}")
        
        # Load the model using MLflow
        model = mlflow.sklearn.load_model(model_path)
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def run(raw_data):
    """Score data using the loaded model.
    
    Args:
        raw_data: JSON string containing the input data
        
    Returns:
        JSON string containing predictions
    """
    try:
        # Parse input data
        data = np.array(json.loads(raw_data)['data'])
        logger.info(f"Received data shape: {data.shape}")
        
        # Get predictions
        predictions = model.predict(data)
        logger.info(f"Generated {len(predictions)} predictions")
        
        # Map predictions to class names
        classnames = ['not-diabetic', 'diabetic']
        predicted_classes = [classnames[prediction] for prediction in predictions]
        
        # Return predictions as JSON
        return json.dumps(predicted_classes)
        
    except Exception as e:
        error_message = f"Error during prediction: {str(e)}"
        logger.error(error_message)
        return json.dumps({"error": error_message})