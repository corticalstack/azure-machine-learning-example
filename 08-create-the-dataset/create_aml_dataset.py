#!/usr/bin/env python
"""
Script to create a new diabetes dataset in the Azure ML workspace.
This script uploads the local diabetes.csv file as a data asset in the workspace.

Usage:
    python create_aml_dataset.py [--data_path PATH] [--dataset_name NAME] [--dataset_description DESC]

Example:
    python create_aml_dataset.py --dataset_name diabetes-diagnostics
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress Azure SDK HTTP logging as it pollutes and makes it difficult to see use case code output
logging.getLogger("azure").setLevel(logging.ERROR)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.ERROR)

def get_ml_client() -> MLClient:
    """
    Initialize and return Azure ML client using config file.
    
    Returns:
        MLClient: Authenticated Azure ML client
    
    Raises:
        Exception: If client initialization fails
    """
    try:
        credential = DefaultAzureCredential()
        ml_client = MLClient.from_config(credential)
        logger.info(f"Successfully connected to Azure ML workspace: {ml_client.workspace_name}")
        return ml_client
    except Exception as e:
        logger.error(f"Failed to initialize ML Client: {str(e)}")
        raise

def create_dataset(
    ml_client: MLClient,
    data_path: str,
    dataset_name: str,
    dataset_description: str = None,
    tags: dict = None
) -> Data:
    """
    Create a new dataset in the Azure ML workspace from a local CSV file.
    
    Args:
        ml_client: Authenticated Azure ML client
        data_path: Path to the local .csv file
        dataset_name: Name for the dataset in the workspace
        dataset_description: Optional description for the dataset
        tags: Optional tags for the dataset
        
    Returns:
        Data: The created data asset
    """
    try:
        # Validate the data file exists
        data_file = Path(data_path)
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        logger.info(f"Creating data asset from file: {data_path}")
        
        # Set default description if not provided
        if not dataset_description:
            dataset_description = (
                "Diabetes dataset containing diagnostic measurements. "
                "The objective is to predict based on diagnostic measurements "
                "whether a patient has diabetes."
            )
        
        # Set default tags if not provided
        if not tags:
            tags = {
                "source": "local_file",
                "format": "csv",
                "created_by": "create_aml_dataset.py"
            }
        
        # Note by not specifying version it will auto-increment
        data_asset = Data(
            name=dataset_name,
            description=dataset_description,
            path=data_path,
            type=AssetTypes.URI_FILE,
            tags=tags
        )

        # Create the data asset in the workspace
        registered_data_asset = ml_client.data.create_or_update(data_asset)
        
        logger.info(f"Successfully created data asset '{dataset_name}' (version: {registered_data_asset.version})")
        logger.info(f"Data asset ID: {registered_data_asset.id}")
        
        return registered_data_asset
    
    except Exception as e:
        logger.error(f"Error creating data asset: {str(e)}")
        raise

def main():
    """Main function to parse arguments and create the dataset."""
    parser = argparse.ArgumentParser(description="Create a diabetes dataset in Azure ML workspace")
    
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="../00-assets/data/diabetes.csv",
        help="Path to the diabetes CSV file (default: diabetes.csv)"
    )
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default="diabetes-diagnostics",
        help="Name for the dataset in the workspace (default: diabetes-diagnostics)"
    )
    parser.add_argument(
        "--dataset_description", 
        type=str, 
        help="Description for the dataset (optional)"
    )
    
    args = parser.parse_args()
    
    try:
        # Validate the data file exists
        data_file = Path(args.data_path)
        if not data_file.exists():
            logger.error(f"Data file not found: {args.data_path}")
            sys.exit(1)
        
        # Get the absolute path to the data file
        data_path = str(data_file.resolve())
        
        logger.info(f"Using data file: {data_path}")
        logger.info(f"Dataset name: {args.dataset_name}")
        
        # Connect to the Azure ML workspace
        logger.info("Connecting to Azure ML workspace...")
        ml_client = get_ml_client()
        
        # Create the dataset
        data_asset = create_dataset(
            ml_client=ml_client,
            data_path=data_path,
            dataset_name=args.dataset_name,
            dataset_description=args.dataset_description,
        )
        
        logger.info("Dataset creation completed successfully")
        logger.info(f"Dataset '{args.dataset_name}' (version: {data_asset.version}) is now available in the workspace")
        
    except Exception as e:
        logger.error(f"Dataset creation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()