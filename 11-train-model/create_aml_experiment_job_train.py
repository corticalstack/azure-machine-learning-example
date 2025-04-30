"""
Azure ML Training Job Execution Script

This script sets up and executes a machine learning training job on Azure ML.
It handles data asset management, job configuration, and execution monitoring.
"""
import argparse
import os
from typing import Any, Dict
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, command

def get_latest_data_asset_version(client: MLClient, data_asset_name: str) -> int:
    """
    Retrieve the latest version number of a data asset.
    
    Args:
        client: Azure ML client instance
        data_asset_name: Name of the data asset
    
    Returns:
        Latest version number of the data asset
    """
    versions = client.data.list(name=data_asset_name)
    return max(version.version for version in versions)

def setup_ml_client(credential: DefaultAzureCredential) -> MLClient:
    """
    Initialize and verify ML client connection.
    
    Args:
        credential: Azure credential object
    
    Returns:
        Configured ML client instance
    """
    try:
        return MLClient.from_config(credential, path='config.json')
    except Exception as e:
        raise Exception(f"Failed to initialize ML client: {str(e)}")

def create_job_inputs(ml_client: MLClient, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Create job input configuration dictionary.
    
    Args:
        ml_client: Azure ML client instance
        args: Parsed command line arguments
    
    Returns:
        Dictionary containing job input configurations
    """
    
    model_output = "outputs/model_output/"
    evaluation_output = args.evaluation_output
    
    return {
        "pipeline": args.pipeline,
        "data_asset_name": args.data_asset_name,
        "model_name": args.model_name,
        "reg_rate": args.reg_rate,
        "solver": args.solver,
        "model_output": model_output,
        "evaluation_output": evaluation_output
    }
                      
def main(args: argparse.Namespace) -> None:
    """
    Main execution function for setting up and running the ML job.
    
    Args:
        args: Parsed command line arguments
    """
    print("##[section]Defining experiment job...")

    credential = DefaultAzureCredential()
    ml_client = setup_ml_client(credential)

    # Verify ML client handle via compute cluster availability
    try:
        ml_client.compute.get(args.compute_name)
    except AzureError:
        raise Exception(f"Compute cluster '{args.compute_name}' not found")

    job_inputs = create_job_inputs(ml_client, args)

    # Configure experiment job
    experiment_job = command(
        code=".",
        command="python train.py \
                --model_name ${{inputs.model_name}} \
                --data_asset_name ${{inputs.data_asset_name}} \
                --reg_rate ${{inputs.reg_rate}} \
                --solver ${{inputs.solver}} \
                --pipeline ${{inputs.pipeline}} \
                --model_output ${{inputs.model_output}} \
                --evaluation_output ${{inputs.evaluation_output}}",
        environment=f"{args.environment_name}@latest",
        environment_variables={
            "MANAGED_IDENTITY_CLIENT_ID": os.environ.get("MANAGED_IDENTITY_CLIENT_ID")
        },
        compute=args.compute_name,
        experiment_name=args.experiment_name,
        inputs=job_inputs,
        tags={
            "model_name": args.model_name,
            "data_asset_name": args.data_asset_name,
            "reg_rate": args.reg_rate,
            "solver": args.solver
        }
    )

    print("##[section]Creating/updating job...")
    try:
        returned_job = ml_client.jobs.create_or_update(experiment_job)
        ml_client.jobs.stream(returned_job.name)
    except Exception as e:
        raise Exception(f"Job execution failed: {str(e)}")

def parse_args() -> argparse.Namespace:
    """
    Parse and validate command line arguments.
    
    Returns:
        Namespace containing parsed arguments
    """
    parser = argparse.ArgumentParser("Execute experiment job")
    parser.add_argument("--experiment_name", type=str, required=True, help="Experiment Name")
    parser.add_argument("--environment_name", type=str, required=True, help="Registered Environment Name")
    parser.add_argument("--model_name", type=str, required=True, help="Model Name")
    parser.add_argument("--compute_name", type=str, required=True, help="Compute Cluster Name")
    parser.add_argument("--data_asset_name", type=str, required=True, help="Data asset name")
    parser.add_argument("--pipeline", type=str, default="training", help="Pipeline name")
    parser.add_argument("--reg_rate", type=float, required=True, help="Regularization rate")
    parser.add_argument("--solver", type=str, required=True, help="Solver algorithm")
    parser.add_argument("--evaluation_output", type=str, default="outputs/evaluation/", help="Path for evaluation results")

    return parser.parse_args()

if __name__ == "__main__":
    print("##[section]Parsing arguments...")
    args = parse_args()

    # Log configuration parameters
    config_params = {
        "Experiment name": args.experiment_name,
        "Environment name": args.environment_name,
        "Model name": args.model_name,
        "Data asset name": args.data_asset_name,
        "Compute name": args.compute_name,
        "Pipeline": args.pipeline,
        "Regularization rate": args.reg_rate,
        "Solver": args.solver
    }
    
    for param, value in config_params.items():
        print(f"{param}: {value}")

    main(args)
