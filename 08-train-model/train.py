"""
Azure Machine Learning training script for Diabetes prediction use case.
"""

import sys
import os
import argparse
import logging
from typing import Tuple
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import mlflow

from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from azure.ai.ml import MLClient
from azure.identity import ManagedIdentityCredential

# Configure logging
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'training_{datetime.now():%Y%m%d_%H%M%S}.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppress Azure SDK HTTP logging as it pollutes and makes it difficult to see use case code output
logging.getLogger("azure").setLevel(logging.ERROR)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.ERROR)

@dataclass
class TrainingConfig:
    """Data class for training configuration parameters. These parameters will include use-case specific paraneters so should be adapted"""
    pipeline: str
    data_asset_name: str
    model_name: str
    reg_rate: float
    solver: str
    model_output: str
    evaluation_output: str

class MLPipelineError(Exception):
    """Base exception for all pipeline-related errors."""
    pass

class DataError(MLPipelineError):
    """Exception for data-related errors."""
    pass

class ModelError(MLPipelineError):
    """Exception for model-related errors."""
    pass

def get_ml_client_from_env() -> MLClient:
    """
    Initialize and return Azure ML client using environment variables.
    Suitable for pipeline execution scenarios using managed identity.
    
    Returns:
        MLClient: Authenticated Azure ML client
    
    Raises:
        EnvironmentError: If required environment variables are missing
        ConnectionError: If unable to establish connection with Azure ML
    """
    required_env_vars = {
        "AZUREML_ARM_WORKSPACE_NAME": os.environ.get("AZUREML_ARM_WORKSPACE_NAME"),
        "AZUREML_ARM_SUBSCRIPTION": os.environ.get("AZUREML_ARM_SUBSCRIPTION"),
        "AZUREML_ARM_RESOURCEGROUP": os.environ.get("AZUREML_ARM_RESOURCEGROUP"),
        "MANAGED_IDENTITY_CLIENT_ID": os.environ.get("MANAGED_IDENTITY_CLIENT_ID")
    }

    # Validate all required environment variables are present
    missing_vars = [k for k, v in required_env_vars.items() if not v or v.strip() == ""]
    if missing_vars:
        raise EnvironmentError(f"Missing or blank required environment variables: {', '.join(missing_vars)}")

    try:
        credential = ManagedIdentityCredential(
            client_id=required_env_vars["MANAGED_IDENTITY_CLIENT_ID"]
        )
        ml_client = MLClient(
            credential=credential,
            subscription_id=required_env_vars["AZUREML_ARM_SUBSCRIPTION"],
            resource_group_name=required_env_vars["AZUREML_ARM_RESOURCEGROUP"],
            workspace_name=required_env_vars["AZUREML_ARM_WORKSPACE_NAME"]
        )
        logger.info("Successfully initialized ML Client using environment variables")
        return ml_client
    except Exception as e:
        logger.error(f"Failed to initialize ML Client: {str(e)}")
        raise ConnectionError(f"Unable to establish connection with Azure ML: {str(e)}")

def setup_mlflow(ml_client: MLClient) -> None:
    """
    Configure MLflow tracking with Azure ML workspace.

    Args:
        ml_client: Authenticated Azure ML client instance

    Raises:
        MLPipelineError: If MLflow setup fails
    """
    try:
        workspace = ml_client.workspaces.get(ml_client.workspace_name)
        logger.info(f"Workspace: {workspace.name}")
        mlflow.set_tracking_uri(workspace.mlflow_tracking_uri)
        mlflow.autolog()
        logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    except Exception as e:
        raise MLPipelineError(f"MLflow setup failed: {str(e)}")

def get_data_asset_version(data_asset_name: str, ml_client: MLClient) -> int:
    """
    Get the latest version of an AML data asset.

    Args:
        data_asset_name: Name of the data asset
        ml_client: Azure ML client instance

    Returns:
        Latest version number
    """
    try:
        versions = ml_client.data.list(name=data_asset_name)
        return max((version.version for version in versions), default=0)
    except Exception as e:
        logger.warning(f"Version retrieval failed for {data_asset_name}: {e}")
        return 0

def load_dataset(data_asset_name: str, ml_client: MLClient) -> pd.DataFrame:
    """
    Load and validate dataset from Azure ML data asset.
    IMPORTANT! Implements use case specific quality checks

    Args:
        data_asset_name: Name of the data asset
        ml_client: Azure ML client instance

    Returns:
        Loaded and validated DataFrame

    Raises:
        DataError: If data loading or validation fails
    """
    try:
        version = get_data_asset_version(data_asset_name, ml_client)
        data_asset = ml_client.data.get(name=data_asset_name, version=version)
        
        df = pd.read_csv(data_asset.path)
        
        if df.empty:
            raise DataError("Empty dataset loaded")
        
        null_counts = df.isnull().sum()
        if null_counts.any():
            logger.warning(f"Null values detected:\n{null_counts[null_counts > 0]}")
        
        return df
    except Exception as e:
        raise DataError(f"Data loading failed: {str(e)}")

def prepare_training_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare features and labels for model training.
    IMPORTANT! Implements use case specific data prep

    Args:
        df: Input DataFrame

    Returns:
        Tuple of features and labels arrays
    """
    feature_columns = ['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure', 
                      'TricepsThickness', 'SerumInsulin', 'BMI', 
                      'DiabetesPedigree', 'Age']
    label_column = 'Diabetic'
    
    return df[feature_columns].values, df[label_column].values

def train_model(
    config: TrainingConfig,
    X: np.ndarray,
    y: np.ndarray
) -> BaseEstimator:
    """
    Train and evaluate the machine learning model.
    IMPORTANT! Implements use case specific training code with train-test split and evaluation

    Args:
        config: Training configuration
        X: Features
        y: Labels

    Returns:
        Trained model instance

    Raises:
        ModelError: If model training fails
    """
    try:
        test_size = 0.2
        random_state = 42
        # Split data into train and test sets with stratification to maintain class balance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        mlflow.log_param("custom_test_size", test_size)
        mlflow.log_param("custom_random_state", random_state)
        logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
        
        model = LogisticRegression(
            C=1/config.reg_rate,
            solver=config.solver
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions on test set
        y_pred = model.predict(X_test)
        
        # Calculate performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Log metrics
        # Note with Scikit-learn's autologged metrics via MLflow's autolog, typically 5-fold cross-validation is used.
        # This means the model is trained 5 times, each time using a different part of the data for validation.
        # This is not the same as the test set used here, which is a single split of the data.
        # In practice with non-toy dataset, you would use cross-validation for the training set and then evaluate on a separate test set.

        mlflow.log_metric("custom_test_set_accuracy", accuracy)
        mlflow.log_metric("custom_test_set_precision", precision)
        mlflow.log_metric("custom_test_set_recall", recall)
        mlflow.log_metric("custom_test_set_f1_score", f1)
        
        logger.info(f"Model performance: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        
        # Save model to the specified output path
        os.makedirs(config.model_output, exist_ok=True)
        mlflow.sklearn.save_model(model, config.model_output)
        
        # Also save to the original location for backward compatibility
        mlflow.sklearn.save_model(model, "model")
        logger.info(f"Model saved to {config.model_output} and model directories")
        
        # Create evaluation output directory
        os.makedirs(config.evaluation_output, exist_ok=True)
        
        return model
    except Exception as e:
        raise ModelError(f"Model training and evaluation failed: {str(e)}")

def parse_arguments() -> TrainingConfig:
    """Parse and validate command line arguments, including use-case specific hyperparameters for example."""
    parser = argparse.ArgumentParser(description="ML Training Pipeline")
    parser.add_argument("--pipeline", type=str, help="Pipeline name")
    parser.add_argument("--data_asset_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--reg_rate", type=float, required=True)
    parser.add_argument("--solver", type=str, required=True)
    parser.add_argument("--model_output", type=str, default="outputs/model_output", help="Path of output model")
    parser.add_argument("--evaluation_output", type=str, default="outputs/evaluation", help="Path of eval results")
    
    args = parser.parse_args()
    return TrainingConfig(**vars(args))

def run_training_pipeline(config: TrainingConfig) -> None:
    """
    Execute the main training pipeline.

    Args:
        config: Training configuration

    Raises:
        MLPipelineError: If pipeline execution fails
    """
    try:
        # Initialize AML client
        ml_client = get_ml_client_from_env()
        setup_mlflow(ml_client)
        
        # Load and prepare data
        df = load_dataset(config.data_asset_name, ml_client)
        X, y = prepare_training_data(df)
        
        # Train and evaluate model
        model = train_model(config, X, y)
        
        # Log total samples
        mlflow.log_metric("total_samples", len(X))
        logger.info("Training pipeline completed successfully")
        
    except Exception as e:
        raise MLPipelineError(f"Pipeline failed: {str(e)}")

if __name__ == "__main__":
    try:
        config = parse_arguments()
        
        # Log all arguments passed to the script
        logger.info("Training arguments:")
        logger.info(f"  pipeline: {config.pipeline}")
        logger.info(f"  data_asset_name: {config.data_asset_name}")
        logger.info(f"  model_name: {config.model_name}")
        logger.info(f"  reg_rate: {config.reg_rate}")
        logger.info(f"  solver: {config.solver}")
        logger.info(f"  model_output: {config.model_output}")
        logger.info(f"  evaluation_output: {config.evaluation_output}")
        
        with mlflow.start_run() as run:
            logger.info(f"MLflow run ID: {run.info.run_id}")
            mlflow.log_params(vars(config))
            run_training_pipeline(config)
            
    except Exception as e:
        logger.critical(f"Critical model training error: {str(e)}")
        sys.exit(1)
