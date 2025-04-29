import argparse
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
import logging
LOGGER = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test an Azure ML online endpoint")
    parser.add_argument("--endpoint_name", type=str, required=True, 
                        help="Name of the online endpoint")
    parser.add_argument("--deployment_name", type=str, required=True, 
                        help="Name of the deployment")
    parser.add_argument("--request_file", type=str, required=True, 
                        help="Path to the request file")
    return parser.parse_args()


def test_online_endpoint(endpoint_name, deployment_name, request_file):
    """Test an Azure ML online endpoint with the provided parameters."""
    credential = DefaultAzureCredential()

    try:
        ml_client = MLClient.from_config(credential)
    except Exception as ex:
        print(f"Exception: {ex}")

    # Invoke and test endpoint
    response = ml_client.online_endpoints.invoke(
        endpoint_name=endpoint_name,
        deployment_name=deployment_name,
        request_file=request_file
    )

    print(f'Diabetic prediction: {response}')
    LOGGER.debug(f'Diabetic prediction: {response}')
    assert "not-diabetic" in response


if __name__ == "__main__":
    args = parse_args()
    test_online_endpoint(
        endpoint_name=args.endpoint_name,
        deployment_name=args.deployment_name,
        request_file=args.request_file,
    )
