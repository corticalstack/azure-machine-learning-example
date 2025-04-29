"""
Creates an Azure Machine Learning Managed Online Endpoint (MOE) for real-time inferencing of diabetes classification
"""

import argparse
import logging
import sys
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint
from azure.core.exceptions import ResourceNotFoundError, ServiceRequestError
from azure.identity import DefaultAzureCredential, AzureCliCredential, CredentialUnavailableError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Suppress Azure SDK HTTP logging as it pollutes and makes it difficult to see use case code output
logging.getLogger("azure").setLevel(logging.ERROR)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.ERROR)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Create Azure ML managed online endpoint")
    
    parser.add_argument(
        "--endpoint_name", 
        type=str, 
        required=True,
        help="Name of the online endpoint to create or update"
    )
    
    parser.add_argument(
        "--description", 
        type=str, 
        default="",
        help="Description of the online endpoint"
    )
    
    parser.add_argument(
        "--auth_mode", 
        type=str, 
        choices=["key", "aml_token"],
        default="aml_token",
        help="Endpoint authentication mode: 'key' or 'aml_token'"
    )
    
    parser.add_argument(
        "--public_network", 
        type=str,
        choices=["enabled", "disabled"],
        default="disabled",
        help="Public network access: 'enabled' or 'disabled'"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def get_ml_client() -> MLClient:
    """
    Initialize and return an Azure ML client.

    Args:
        config_path (str): Path to the Azure ML configuration file

    Returns:
        MLClient: Initialized Azure ML client

    Raises:
        FileNotFoundError: If the configuration file is not found
        CredentialUnavailableError: If Azure credentials cannot be obtained
        Exception: For other errors during client initialization
    """

    try:
        credential = DefaultAzureCredential()
        logger.debug("Using DefaultAzureCredential for authentication")
    except CredentialUnavailableError:
        # Fall back to AzureCliCredential if DefaultAzureCredential fails
        logger.warning("DefaultAzureCredential failed, falling back to AzureCliCredential")
        try:
            credential = AzureCliCredential()
        except Exception as ex:
            logger.error(f"Failed to obtain Azure credentials: {ex}")
            raise
    
    # Initialize ML client
    try:
        ml_client = MLClient.from_config(credential)
        logger.info(f"Successfully initialized ML client for workspace: {ml_client.workspace_name}")
        return ml_client
    except Exception as ex:
        logger.error(f"Failed to initialize ML client: {ex}")
        raise


def create_or_update_endpoint(
    ml_client: MLClient, 
    endpoint_name: str, 
    description: str, 
    auth_mode: str,
    public_network: str
) -> ManagedOnlineEndpoint:
    """
    Create or update an Azure ML managed online endpoint.

    Args:
        ml_client (MLClient): Azure ML client
        endpoint_name (str): Name of the endpoint
        description (str): Description of the endpoint
        auth_mode (str): Authentication mode ('key' or 'aml_token')
        public_network (str): Public network access ('enabled' or 'disabled')

    Returns:
        ManagedOnlineEndpoint: Created or updated endpoint

    Raises:
        Exception: If endpoint creation or update fails
    """
    logger.info(f"Creating/updating online endpoint: {endpoint_name}")
    
    # Configure the endpoint
    endpoint_config = {
        "name": endpoint_name,
        "auth_mode": auth_mode,
        "public_network_access": public_network
    }
    
    # Add description if provided
    if description:
        endpoint_config["description"] = description
    
    # Create endpoint entity
    online_endpoint = ManagedOnlineEndpoint(**endpoint_config)
    
    try:
        # Begin the create or update operation
        logger.info("Submitting endpoint creation/update job")
        endpoint_job = ml_client.online_endpoints.begin_create_or_update(online_endpoint)
        
        # Wait for the operation to complete
        logger.info("Waiting for endpoint creation/update to complete...")
        result = endpoint_job.result()
        
        # Log success and endpoint details
        logger.info(f"Endpoint '{endpoint_name}' successfully created/updated")
        logger.info(f"Endpoint scoring URI: {result.scoring_uri}")
        
        return result
    except Exception as ex:
        logger.error(f"Failed to create/update endpoint: {ex}")
        raise


def main() -> None:
    """
    Main function to create or update an Azure ML managed online endpoint.
    
    Parses arguments, initializes the ML client, and creates/updates the endpoint.
    """
    args = parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    logger.info("Script arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    try:
        # Initialize ML client
        ml_client = get_ml_client()
        
        # Create or update the endpoint
        endpoint = create_or_update_endpoint(
            ml_client=ml_client,
            endpoint_name=args.endpoint_name,
            description=args.description,
            auth_mode=args.auth_mode,
            public_network=args.public_network
        )
        
        # Output success message
        logger.info(f"Endpoint operation completed successfully")
        
        # Return success exit code
        sys.exit(0)
        
    except Exception as ex:
        logger.error(f"Unexpected error: {ex}")
        sys.exit(1)


if __name__ == "__main__":
    main()