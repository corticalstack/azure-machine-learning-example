#!/usr/bin/env python3
"""
Azure ML Online Deployment Creation Script. It handles the creation, configuration, and traffic allocation.
"""

import argparse
import logging
import sys

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (CodeConfiguration, ManagedOnlineDeployment,
                                 ManagedOnlineEndpoint)
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError
from azure.identity import DefaultAzureCredential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Suppress Azure SDK HTTP logging as it pollutes and makes it difficult to see use case code output
logging.getLogger("azure").setLevel(logging.ERROR)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.ERROR)

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the deployment script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Create online deployment for Azure ML")
    
    # Required arguments
    parser.add_argument("--deployment_name", type=str, required=True, 
                        help="Name of online deployment")
    parser.add_argument("--endpoint_name", type=str, required=True, 
                        help="Name of the online endpoint")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to model or AML model reference")
    parser.add_argument("--environment", type=str, required=True, 
                        help="Model environment name")
    parser.add_argument("--score_path", type=str, required=True, 
                        help="Path to scoring code directory")
    parser.add_argument("--score_script", type=str, required=True, 
                        help="Scoring script filename")
    
    # Optional arguments with defaults
    parser.add_argument("--instance_type", type=str, default="Standard_DS3_v2", 
                        help="VM instance type (default: Standard_DS3_v2)")
    parser.add_argument("--instance_count", type=int, default=1, 
                        help="Number of instances (default: 1)")
    parser.add_argument("--traffic_allocation", type=str, default="100", 
                        help="Deployment traffic allocation percentage (default: 100)")

    return parser.parse_args()


def get_ml_client(credential: DefaultAzureCredential) -> MLClient:
    """
    Initialize and return the Azure ML client.
    
    Args:
        credential (DefaultAzureCredential): Azure credential object
        
    Returns:
        MLClient: Initialized Azure ML client
        
    Raises:
        FileNotFoundError: If the config file is not found
        Exception: For any other errors during client initialization
    """
    try:
        return MLClient.from_config(credential)
    except Exception as ex:
        logger.error(f"Failed to initialize ML client: {str(ex)}")
        raise


def get_latest_environment_version(ml_client: MLClient, environment_name: str) -> int:
    """
    Get the latest version of the specified environment.
    
    Args:
        ml_client (MLClient): Azure ML client
        environment_name (str): Name of the environment
        
    Returns:
        int: Latest environment version
        
    Raises:
        ResourceNotFoundError: If the environment is not found
        ValueError: If no versions of the environment exist
    """
    try:
        logger.info(f"Retrieving latest version for environment: {environment_name}")
        environment_versions = [
            int(e.version) for e in ml_client.environments.list(name=environment_name)
        ]
        
        if not environment_versions:
            raise ValueError(f"No versions found for environment: {environment_name}")
            
        latest_version = max(environment_versions)
        logger.info(f"Latest environment version: {latest_version}")
        return latest_version
    except ResourceNotFoundError:
        logger.error(f"Environment not found: {environment_name}")
        raise
    except Exception as ex:
        logger.error(f"Error retrieving environment versions: {str(ex)}")
        raise


def create_deployment(ml_client: MLClient, args: argparse.Namespace, 
                     environment_version: int) -> None:
    """
    Create or update an online deployment.
    
    Args:
        ml_client (MLClient): Azure ML client
        args (argparse.Namespace): Command line arguments
        environment_version (int): Latest environment version
        
    Raises:
        Exception: If deployment creation fails
    """
    try:
        # Construct the environment reference string
        environment_reference = f"{args.environment}:{environment_version}"
        logger.info(f"Creating deployment '{args.deployment_name}' with environment {environment_reference}")
        
        # Create deployment configuration
        online_deployment = ManagedOnlineDeployment(
            name=args.deployment_name,
            endpoint_name=args.endpoint_name,
            model=args.model_path,
            environment=environment_reference,
            code_configuration=CodeConfiguration(
                code=args.score_path, 
                scoring_script=args.score_script
            ),
            instance_type=args.instance_type,
            instance_count=args.instance_count,
            egress_public_network_access="enabled" 
        )
        
        # Start deployment creation/update
        logger.info("Starting deployment creation/update operation")
        deployment_job = ml_client.online_deployments.begin_create_or_update(
            deployment=online_deployment
        )
        
        # Wait for deployment to complete
        logger.info("Waiting for deployment operation to complete...")
        deployment_job.wait()
        logger.info(f"Deployment '{args.deployment_name}' created/updated successfully")
        
    except HttpResponseError as ex:
        logger.error(f"Azure API error during deployment creation: {str(ex)}")
        raise
    except Exception as ex:
        logger.error(f"Failed to create deployment: {str(ex)}")
        raise


def update_endpoint_traffic(ml_client: MLClient, args: argparse.Namespace) -> None:
    """
    Update the endpoint to allocate traffic to the deployment.
    
    Args:
        ml_client (MLClient): Azure ML client
        args (argparse.Namespace): Command line arguments
        
    Raises:
        Exception: If endpoint update fails
    """
    try:
        logger.info(f"Updating traffic allocation for endpoint '{args.endpoint_name}'")
        
        # Get the endpoint configuration
        online_endpoint = ManagedOnlineEndpoint(
            name=args.endpoint_name
        )
        
        # Set traffic allocation
        traffic_percentage = args.traffic_allocation
        logger.info(f"Setting traffic allocation: {args.deployment_name} -> {traffic_percentage}%")
        online_endpoint.traffic = {args.deployment_name: traffic_percentage}
        
        # Update the endpoint
        logger.info("Starting endpoint update operation")
        endpoint_update_job = ml_client.begin_create_or_update(online_endpoint)
        
        # Wait for update to complete
        logger.info("Waiting for endpoint update operation to complete...")
        endpoint_update_job.wait()
        logger.info(f"Endpoint '{args.endpoint_name}' updated successfully")
        
    except HttpResponseError as ex:
        logger.error(f"Azure API error during endpoint update: {str(ex)}")
        raise
    except Exception as ex:
        logger.error(f"Failed to update endpoint: {str(ex)}")
        raise


def log_arguments(args: argparse.Namespace) -> None:
    """
    Log all the arguments passed to the script.
    
    Args:
        args (argparse.Namespace): Command line arguments
    """
    logger.info("Script arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")


def main() -> None:
    """
    Main function to orchestrate the deployment creation process.
    """
    try:
        # Parse command line arguments
        args = parse_args()
        log_arguments(args)
        
        # Initialize Azure credentials
        logger.info("Initializing Azure credentials")
        credential = DefaultAzureCredential()
        
        # Get ML client
        ml_client = get_ml_client(credential)
        
        # Get latest environment version
        latest_env_version = get_latest_environment_version(ml_client, args.environment)
        
        # Create or update deployment
        create_deployment(ml_client, args, latest_env_version)
        
        # Update endpoint traffic allocation
        update_endpoint_traffic(ml_client, args)
        
        logger.info("Deployment process completed successfully")
        
    except Exception as ex:
        logger.error(f"Deployment process failed: {str(ex)}")
        sys.exit(1)


if __name__ == "__main__":
    main()