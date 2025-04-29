import argparse
from pathlib import Path
import mlflow
import os
import json


def main(args):
    print("##[section]Verifying deploy flag...")
    with open((Path(args.evaluation_output) / "deploy_flag.txt"), 'rb') as infile:
        deploy_flag = int(infile.read())

    mlflow.log_metric("deploy flag", int(deploy_flag))

    # deploy_flag=1
    if deploy_flag == 1:
        print("##[section]Registering model {}...".format(args.model_name))

        # Load model
        model = mlflow.sklearn.load_model(args.model_path)

        # Log model using mlflow
        mlflow.sklearn.log_model(model, args.model_name)

        # Register logged model using mlflow
        run_id = mlflow.active_run().info.run_id
        model_uri = f'runs:/{run_id}/{args.model_name}'
        mlflow_model = mlflow.register_model(model_uri, args.model_name)
        model_version = mlflow_model.version

        # Write model info
        dict = {"id": "{0}:{1}".format(args.model_name, model_version)}
        output_path = os.path.join(args.model_info_output_path, "model_info.json")
        with open(output_path, "w") as of:
            json.dump(dict, fp=of)

    else:
        print("Deploy flag is 0, model will not be registered!")


def parse_args():
    print("##[section]Parsing arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline', type=str, help='Pipeline')
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered')
    parser.add_argument('--model_path', type=str, help='Model directory')
    parser.add_argument('--evaluation_output', type=str, help='Path of eval results')
    parser.add_argument(
        "--model_info_output_path", type=str, help="Path to write model info JSON"
    )

    return parser.parse_args()


if __name__ == "__main__":
    mlflow.start_run()

    args = parse_args()

    lines = [
        f"Pipeline: {args.pipeline}",
        f"Model name: {args.model_name}",
        f"Model path: {args.model_path}",
        f"Evaluation output path: {args.evaluation_output}",
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()
    