"""
Model evaluation script for AML models

This script provides a framework for evaluating machine learning models by:
1. Loading test data from an AML data asset
2. Computing performance metrics
3. Decision rules for deciding whether to promote (in this use case, comparing with existing models)
4. Logging results using MLflow
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
import mlflow
import matplotlib.pyplot as plt

from typing import Tuple, Dict, Any
from dataclasses import dataclass
from mlflow.tracking import MlflowClient

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split

from azure.ai.ml import MLClient
from azure.identity import ManagedIdentityCredential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('evaluate.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppress Azure SDK HTTP logging as it pollutes and makes it difficult to see use case code output
logging.getLogger("azure").setLevel(logging.ERROR)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.ERROR)

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

@dataclass
class EvaluationConfig:
    """Configuration class for model evaluation parameters."""
    model_name: str
    test_data_asset_name: str
    job_name: str
    experiment_name: str
    model_path: str
    evaluation_output: str

class ModelEvaluator:
    """
    A class for evaluating the model and deciding model promotion.
    """
    
    def __init__(self, model: Any, test_data_asset_name: str, ml_client: MLClient):
        """
        Initialize the model evaluator.
        
        Args:
            model: Trained machine learning model
            test_data_asset_name: Name of the test dataset in AML datastore
            ml_client: Authenticated AML client handle
        """
        self.model = model
        self.test_data_asset_name = test_data_asset_name
        self.ml_client = ml_client

        # Use-case specific
        self._feature_columns = [
            'Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure',
            'TricepsThickness', 'SerumInsulin', 'BMI', 'DiabetesPedigree', 'Age'
        ]
        self._target_column = 'Diabetic'

    def _get_latest_data_asset_version(self, data_asset_name: str) -> int:
        """Fetch the latest version of a data asset from Azure ML."""
        try:
            versions = self.ml_client.data.list(name=data_asset_name)
            latest_version = max(version.version for version in versions)
            logger.info(f"Latest version for {data_asset_name}: {latest_version}")
            return latest_version
        except Exception as e:
            logger.warning(f"Version fetch failed for {data_asset_name}: {e}")
            return 0

    def load_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and prepare test data from Azure ML datastore.
        
        Returns:
            Tuple containing feature matrix and target vector
        """
        try:
            latest_version = self._get_latest_data_asset_version(self.test_data_asset_name)
            test_data = self.ml_client.data.get(
                name=self.test_data_asset_name,
                version=latest_version
            )
            
            df = pd.read_csv(test_data.path)
            X = df[self._feature_columns].values
            y = df[self._target_column].values
            
            # Use the same train-test split parameters as in train.py
            test_size = 0.2
            random_state = 42

            # Split data into train and test sets with stratification to maintain class balance
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            logger.info(f"Test data shape: {X_test.shape}")
            
            return X_test, y_test
            
        except Exception as e:
            logger.error(f"Test data loading failed: {e}")
            raise

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       y_prob: np.ndarray) -> Dict[str, float]:
        """
        Use-case specific: compute and log classification metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_prob[:, 1])
        }
        
        self._plot_roc_curve(y_true, y_prob[:, 1])
        return metrics

    def _plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """Use-case specific. Generate and save ROC curve plot."""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(6, 4))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.savefig("roc_curve.png")
        mlflow.log_artifact("roc_curve.png")
        plt.close()

    def check_model_promotion(self, model_name: str, metrics: Dict[str, float],
                            X_test: np.ndarray, y_test: np.ndarray,
                            y_pred: np.ndarray) -> bool:
        """
        Use-case specific. Determine if the current model should be promoted based on performance metric.
        
        Args:
            model_name: Name of the current model
            metrics: Current model's performance metrics
            X_test: Test features
            y_test: True labels
            y_pred: Current model's predictions
            
        Returns:
            Boolean indicating whether the model should be promoted
        """
        try:
            logger.info("Evaluating model promotion criteria")
            scores = {}
            predictions = {}
            
            # Compare with registered, historical champion models
            client = MlflowClient()
            
            for model_version in client.search_model_versions(f"name='{model_name}'"):
                version = model_version.version
                registered_model = mlflow.sklearn.load_model(
                    model_uri=f"models:/{model_name}/{version}"
                )
                y_hat = registered_model.predict(X_test)
                acc = accuracy_score(y_test, y_hat)
                scores[f"{model_name}:{version}"] = acc
                logger.info(f"Model {model_name} version {version} accuracy: {acc:.4f}")

            # Determine if we should promote this current model, based on historic model scores
            current_acc = metrics['accuracy']
            deploy_flag = not scores or current_acc > max(scores.values())
            
            self._log_promotion_results(scores, current_acc, y_pred, deploy_flag)
            
            return deploy_flag

        except Exception as e:
            logger.error(f"Promotion check failed: {e}")
            raise

    def _log_promotion_results(self, scores: Dict[str, float], current_acc: float,
                             y_pred: np.ndarray, deploy_flag: bool) -> None:
        """Log promotion decision and related visualizations."""
        scores["current_model"] = current_acc
        
        # Create and save use-case specific performance comparison plot
        pd.DataFrame(scores, index=["accuracy"]).plot(kind='bar', figsize=(15, 10))
        plt.savefig("perf_comparison.png")
        mlflow.log_artifact("perf_comparison.png")
        plt.close()
        
        mlflow.log_metric("deploy_flag", int(deploy_flag))
        
        # Log run information
        current_run = mlflow.active_run()
        if current_run:
            run_id = current_run.info.run_id
            logger.info(f"Setting MLflow run ID {run_id} as job tag")
            mlflow.set_tag("job", run_id)

    def evaluate(self) -> Dict[str, float]:
        """
        Use-case specific. Execute the complete evaluation pipeline.
        
        Returns:
            Dictionary of computed metrics
        """
        try:
            X_test, y_test = self.load_test_data()
            y_pred = self.model.predict(X_test)
            y_prob = self.model.predict_proba(X_test)
            
            metrics = self.compute_metrics(y_test, y_pred, y_prob)
            
            # Log metrics
            for metric_name, value in metrics.items():
                logger.info(f"{metric_name}: {value:.4f}")
                mlflow.log_metric(metric_name, value)
                
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

def parse_args() -> EvaluationConfig:
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(description="Model evaluation script")
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--test_data_asset_name', type=str, required=True)
    parser.add_argument('--job_name', type=str, required=True)
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--evaluation_output', type=str, required=True, help="Path for evaluation results")
    
    args = parser.parse_args()
    return EvaluationConfig(**vars(args))

def main():
    """Main execution function."""
    try:
        config = parse_args()
        
        # Log configuration
        logger.info(f"Evaluation parameters:")
        for arg_name, arg_value in vars(config).items():
            logger.info(f"  {arg_name}: {arg_value}")
        
        mlflow.autolog()
        
        # Initialize Azure ML client
        ml_client = get_ml_client_from_env()
        
        # Load model
        model = mlflow.sklearn.load_model(config.model_path)
        logger.info(f"Model: {model}")
        
        # Create evaluator instance
        evaluator = ModelEvaluator(
            model=model,
            test_data_asset_name=config.test_data_asset_name,
            ml_client=ml_client
        )
        
        # Run evaluation
        metrics = evaluator.evaluate()
        
        # Check for model promotion
        X_test, y_test = evaluator.load_test_data()
        y_pred = evaluator.model.predict(X_test)
        deploy_flag = evaluator.check_model_promotion(
            config.model_name, metrics, X_test, y_test, y_pred
        )
        
        logger.info(f"Evaluation completed. Deploy flag: {deploy_flag}")
        
        # Write deploy flag to evaluation output directory
        os.makedirs(config.evaluation_output, exist_ok=True)
        with open(os.path.join(config.evaluation_output, "deploy_flag.txt"), 'w') as outfile:
            outfile.write(f"{int(deploy_flag)}")
        
        # Copy performance comparison plot to evaluation output directory
        import shutil
        if os.path.exists("perf_comparison.png"):
            shutil.copy("perf_comparison.png", os.path.join(config.evaluation_output, "perf_comparison.png"))
            
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
