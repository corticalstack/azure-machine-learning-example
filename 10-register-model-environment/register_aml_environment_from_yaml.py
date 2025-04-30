import sys
import os
import argparse
import logging
# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
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
                     
def parse_args():
    """Parse command line arguments for environment registration."""
    parser = argparse.ArgumentParser(description='Register an Azure ML Environment from a conda YAML file')
    parser.add_argument('--name', type=str, default='diabetes-classify',
                      help='Name of the environment to register')
    parser.add_argument('--description', type=str, 
                      default='Environment for ML model training of diabetes classification',
                      help='Description of the environment')
    parser.add_argument('--base_image', type=str,
                      default='mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest',
                      help='Base Docker image to use')
    parser.add_argument('--conda_file', type=str, default='conda.yaml',
                      help='Path to conda environment YAML file')
    return parser.parse_args()

def register_environment(args):
    """
    Register an Azure ML Environment from a conda YAML file.
    
    Args:
        args: Command line arguments containing environment configuration
        
    Returns:
        The registered environment
        
    Raises:
        FileNotFoundError: If the conda file is not found
        Exception: If environment registration fails
    """
    try:
        # Connect to the workspace
        logger.info("Connecting to Azure ML workspace...")
        ml_client = get_ml_client()
        logger.info(f"Connected to workspace: {ml_client.workspace_name}")
        
        # Resolve conda file path
        if not os.path.isabs(args.conda_file):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            conda_file_path = os.path.join(script_dir, args.conda_file)
        else:
            conda_file_path = args.conda_file

        if not os.path.exists(conda_file_path):
            logger.error(f"Conda file not found at: {conda_file_path}")
            raise FileNotFoundError(f"Conda file not found at: {conda_file_path}")

        logger.info(f"Using conda file: {conda_file_path}")
        logger.info(f"Creating environment '{args.name}' with base image: {args.base_image}")

        # Create environment
        env = Environment(
            name=args.name,
            description=args.description,
            image=args.base_image,
            conda_file=conda_file_path
        )

        # Register the environment
        logger.info(f"Registering environment '{args.name}'...")
        registered_env = ml_client.environments.create_or_update(env)
        logger.info(f"Environment '{args.name}' registered successfully with version: {registered_env.version}")
        return registered_env
        
    except Exception as e:
        logger.error(f"Failed to register environment: {str(e)}")
        raise

def main():
    """Main function to parse arguments and register the environment."""
    try:
        logger.info("Starting environment registration process")
        args = parse_args()
        logger.info(f"Environment name: {args.name}")
        register_environment(args)
        logger.info("Environment registration completed successfully")
    except Exception as e:
        logger.error(f"Environment registration failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
