import argparse
import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, Output, command
from azure.ai.ml.dsl import pipeline


def main(args):
    print("##[section]Defining pipeline job...")
    credential = DefaultAzureCredential()
    try:
        ml_client = MLClient.from_config(credential, path='config.json')

    except Exception as ex:
        print("Exception: {}".format(ex))

    try:
        print(ml_client.compute.get(args.compute_name))
    except:
        print("No compute found")

    train_model = command(
        name="train_model",
        display_name="train-model",
        code=".",
        command="python train.py \
                --pipeline ${{inputs.pipeline}} \
                --model_name ${{inputs.model_name}} \
                --reg_rate ${{inputs.reg_rate}} \
                --solver ${{inputs.solver}} \
                --data_asset_name ${{inputs.data_asset_name}} \
                --model_output ${{outputs.model_output}} \
                --evaluation_output ${{outputs.evaluation_output}}",
        environment=args.environment_name + "@latest",
        environment_variables={
            "MANAGED_IDENTITY_CLIENT_ID": os.environ.get("MANAGED_IDENTITY_CLIENT_ID", "")
        },
        inputs={"pipeline": Input(type="string"),
                "model_name": Input(type="string"),
                "reg_rate": Input(type="string"),
                "solver": Input(type="string"),
                "data_asset_name":  Input(type="string")},
        outputs={"model_output": Output(type="uri_folder"),
                 "evaluation_output": Output(type="uri_folder")}
    )

    evaluate_model = command(
        name="evaluate_model",
        display_name="evaluate-model",
        code=".",
        command="python evaluate.py \
                --model_name ${{inputs.model_name}} \
                --test_data_asset_name ${{inputs.test_data_asset_name}} \
                --job_name ${{inputs.job_name}} \
                --experiment_name ${{inputs.experiment_name}} \
                --model_path ${{inputs.model_path}} \
                --evaluation_output ${{outputs.evaluation_output}}",
        environment=args.environment_name + "@latest",
        environment_variables={
            "MANAGED_IDENTITY_CLIENT_ID": os.environ.get("MANAGED_IDENTITY_CLIENT_ID", "")
        },
        inputs={
            "model_name": Input(type="string"),
            "test_data_asset_name": Input(type="string"),
            "job_name": Input(type="string"),
            "experiment_name": Input(type="string"),
            "model_path": Input(type="uri_folder")
        },
        outputs={
            "evaluation_output": Output(type="uri_folder")
        }
    )

    register_model = command(
        name="register_model",
        display_name="register-model",
        code=".",
        command="python register.py \
                --pipeline ${{inputs.pipeline}} \
                --model_name ${{inputs.model_name}} \
                --model_path ${{inputs.model_path}} \
                --evaluation_output ${{inputs.evaluation_output}} \
                --model_info_output_path ${{outputs.model_info_output_path}}",
        environment=args.environment_name + "@latest",
        inputs={
            "pipeline": Input(type="string"),
            "model_name": Input(type="string"),
            "model_path": Input(type="uri_folder"),
            "evaluation_output": Input(type="uri_folder")},
        outputs={
            "model_info_output_path": Output(type="uri_folder")}
    )

    # 2. Construct pipeline
    @pipeline()
    def training_pipeline():
        print("##[section]Contructing pipeline...")
        # latest_version = ml_client.data._get_latest_version(name=args.table_name).version
        # #latest_version = 1  # hard-coded to 1 as latest version stopped working for this data asset oddly
        # data_asset = ml_client.data.get(name=args.data_asset_name, version=latest_version)

        train = train_model(
            data_asset_name=args.data_asset_name,
            pipeline=args.pipeline,
            model_name=args.model_name,
            reg_rate=args.reg_rate,
            solver=args.solver,
        )

        evaluate = evaluate_model(
            model_name=args.model_name,
            test_data_asset_name=args.data_asset_name,
            job_name=f"{args.pipeline}-job",
            experiment_name=args.experiment_name,
            model_path=train.outputs.model_output
        )

        # Register model - the register_simple.py script will check the deploy flag
        register = register_model(
            pipeline=args.pipeline,
            model_name=args.model_name,
            model_path=train.outputs.model_output,
            evaluation_output=evaluate.outputs.evaluation_output
        )

        return {
            "pipeline_job_trained_model": train.outputs.model_output,
            "pipeline_job_score_report": evaluate.outputs.evaluation_output,
        }

    pipeline_job = training_pipeline()

    # Set pipeline level compute
    pipeline_job.settings.default_compute = args.compute_name

    # Set pipeline level datastore
    pipeline_job.settings.default_datastore = "workspaceblobstore"

    print("##[section]Creating/updating pipeline job...")
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name=args.experiment_name
    )

    pipeline_job
    ml_client.jobs.stream(pipeline_job.name)


def parse_args():
    parser = argparse.ArgumentParser("Deploy Training Pipeline")
    parser.add_argument("--pipeline", type=str, help="Pipeline")
    parser.add_argument("--experiment_name", type=str, help="Experiment Name")
    parser.add_argument("--environment_name", type=str, help="Registered Environment Name")
    parser.add_argument("--model_name", type=str, help="", default="diabetes")
    parser.add_argument("--reg_rate", help="Regularization rate", type=float, default=0.01)
    parser.add_argument("--solver", help="Solver algorithm", type=str, default="lbfgs")
    parser.add_argument("--data_asset_name", type=str, help="Data asset name", default="diabetes-diagnostics")
    parser.add_argument("--compute_name", type=str, help="Compute Cluster Name")
    parser.add_argument("--evaluation_output", type=str, help="Path of eval results")

    return parser.parse_args()


if __name__ == "__main__":
    print("##[section]Parsing arguments...")
    args = parse_args()

    lines = [
        f"Pipeline: {args.pipeline}",
        f"Experiment name: {args.experiment_name}",
        f"Environment name: {args.environment_name}",
        f"Model name: {args.model_name}",
        f"Regularization rate: {args.reg_rate}",
        f"Solver: {args.solver}",
        f"`Data asset name: {args.data_asset_name}",
        f"Compute name: {args.compute_name}",
        f"Evaluation output: {args.evaluation_output}"
    ]

    for line in lines:
        print(line)

    main(args)
