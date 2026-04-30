
## Introduction

So what is Azure Machine Learning? 

Azure Machine Learning is a Platform-as-a-Service that enables data scientists and ML engineers to accelerate the end-to-end machine learning lifecycle. 

It includes tools for:
- Building and training models
- Deploying and monitoring models
- Scaling ML workloads efficiently
- Ensuring responsible AI principles


This repository demonstrates how to build and deploy machine learning models for diabetes prediction using both low-code and pro-code approaches.

The repository uses the Pima Indians Diabetes Dataset, a standard dataset in machine learning chosen for its:
- **Simplicity**: Small and straightforward, allowing focus on Azure ML concepts
- **Well-documented**: Widely used with well-understood characteristics
- **Real-world relevance**: Represents a genuine healthcare use case
- **Practical size**: Enables fast training iterations with low compute requirements

### Prerequisites

To follow along with this tutorial, you'll need:
1. An active Azure subscription
2. Sufficient permissions to create resources in your subscription
3. Python 3.10 or later
4. Azure Machine Learning SDK v2 for Python

**Important:** This repository uses Azure Machine Learning SDK v2, which offers an improved and more consistent API design compared to the older v1.

## 01 - Creating an Azure Machine Learning Workspace

The Azure Machine Learning workspace serves as the foundational resource for managing the entire machine learning lifecycle.

### Why Create an AML Workspace?

The workspace provides:
- A centralized place to work with all machine learning artifacts
- Integration with other Azure services like storage and compute
- Version tracking for assets like data, environments, and models
- Comprehensive job and pipeline tracking with metrics and logs
- Monitoring of the machine learning lifecycle
- Collaboration between team members

### Steps to Create a Workspace

1. **Navigate to the Azure Portal** and sign in with your Azure account
2. **Search for and select Azure Machine Learning**
3. **Click Create** and fill out the mandatory configuration:
   - Resource group
   - Workspace name
   - Region
4. **Configure network settings**:
   - For demonstration purposes, select "All networks" for inbound access
   - For production environments, it's recommended to disable public access
5. **Review and create** the workspace
   - Deployment will take a few minutes to complete
   - Once deployed, you can access your new workspace

## 02 - Azure Machine Learning Model Catalog

The model catalog in Azure Machine Learning studio is a hub for discovering and using a wide range of models to build Generative AI applications.

### Model Collections

The catalog organizes models into three primary collections:

1. **Models Curated by Azure AI**:
   - Popular third-party open weight and proprietary models
   - Optimized to work seamlessly on the Azure AI platform
   - Subject to the model provider's license terms

2. **Azure OpenAI Models**:
   - Flagship Azure OpenAI models available exclusively on Azure
   - Accessible through the Azure OpenAI collection
   - Supported by Microsoft
   - Subject to Azure OpenAI Service product terms and SLA

3. **Open Models from the Hugging Face Hub**:
   - Hundreds of models accessible via the HuggingFace collection
   - Available for real-time inference with online endpoints
   - Created, maintained, and supported by HuggingFace

### Key Capabilities

- **Discover**: Review model cards, try sample inference, and browse code samples
- **Compare**: Evaluate benchmarks across models
- **Fine-tune**: Customize models using your own training data
- **Deploy**: Deploy pretrained or fine-tuned models seamlessly for inference

### Deployment Options

The Model Catalog offers two deployment options:

1. **Managed Compute**:
   - Model weights deployed to dedicated Virtual Machines
   - REST API available through managed online endpoints
   - Billing based on Virtual Machine core hours

2. **Serverless API**:
   - Access to models through an API connected to a central GPU pool
   - Referred to as "Models as a Service"
   - Billing based on inputs and outputs to the APIs (typically in tokens)

## 03 - Connections in Azure Machine Learning

Connections in Azure Machine Learning can be configured to be shared across the entire workspace or limited to the creator.

### Key Features

- Secrets associated with connections are securely persisted in the corresponding Azure Key Vault
- Adheres to robust security and compliance standards
- Can include connections to services like Azure OpenAI, Azure AI Search, and Azure Content Safety
- Primarily used for consumption by promptflow

## 04 - Azure Promptflow

Azure Promptflow is a comprehensive development tool that streamlines the lifecycle of applications powered by Large Language Models (LLMs).

### Key Features

- **Visual Flow Creation**: Build executable flows connecting LLMs, prompts, and Python tools through an intuitive visualized graph interface
- **Collaborative Development**: Version control and share promptflows
- **Prompt Engineering & Evaluation**: Create prompt variants and evaluate their performance
- **Enterprise-Ready Deployment**: Deploy promptflows as real-time endpoints for consumption

### Connections in Promptflow

Connections securely manage credentials and secrets for APIs and data sources:
- Include prebuilt connections for Azure OpenAI and AI Search
- Encapsulate essential information like endpoints and authentication details
- Securely store secrets in Azure Key Vault

### Flow Types

Azure Promptflow offers three specialized flow types:
1. **Standard Flow**: For general application development
2. **Chat Flow**: Tailored for conversational applications with enhanced support for chat history management
3. **Evaluation Flow**: Designed to evaluate the performance of previous flow runs and output relevant metrics

### Tools

Tools are the fundamental building blocks of a flow:
- **LLM Tool**: Write custom prompts and leverage LLMs
- **Python Tool**: Create custom Python functions for data processing, evaluation, and API calls
- **Prompt Tool**: Prepare prompts as strings

### Example Standard Promptflow

A standard flow typically includes:
- An input node with parameters
- An LLM node that defines the service connection details and parameters
- A Python node for processing LLM output
- An output node that receives the final result

## 05 - Automated Machine Learning for Diabetes Classification

Automated Machine Learning (AutoML) automates the time and skills-intensive process of selecting and training models for a given machine learning task.

### Creating an Automated ML Job

1. **Navigate to Automated ML** in the Azure Machine Learning studio
2. **Create a new job** with an experiment name and description
3. **Set the task type** as Classification
4. **Upload the diabetes data file** as an Azure Machine Learning data asset
5. **Configure the dataset**:
   - Set type as Tabular
   - Provide name and description
   - Select data source (local files)
   - Choose destination storage (workspaceblobstore)
   - Upload the diabetes.csv file
   - Configure file format (Delimited with comma delimiter)
   - Review schema and include all columns
6. **Configure the model**:
   - Select "Diabetic" as the target column
   - Set primary metric (AUC_weighted)
   - Enable model explanation
   - Select compute type (Serverless)
7. **Submit the training job**

### Monitoring and Evaluating the AutoML Job

1. **Track job progress** on the Job Overview screen
2. **Explore tested algorithms** as they complete
3. **Review model performance** through metrics like AUC
4. **Analyze diagnostic tools** such as confusion matrices
5. **Examine the Responsible AI dashboard** for insights into model behavior and fairness

## 06 - Setting Up Compute Target Resources

Compute resources are essential for training and deploying machine learning models in Azure ML.

### Creating a User-Assigned Managed Identity

A managed identity allows compute resources to securely access other Azure services without storing credentials.

1. **Search for Managed Identities** in the Azure portal
2. **Create a new managed identity** in the same resource group as your AML workspace
3. **Assign appropriate roles** to the managed identity:
   - AzureML Compute Operator: Allows managing workspace compute resources
   - AzureML Data Scientist: Provides permissions for machine learning operations

### Creating a Compute Cluster

A compute cluster is automatically scalable, reusable compute for training models.

1. **Navigate to Compute** in Azure Machine Learning studio
2. **Create a new compute cluster**:
   - Provide a unique name
   - Select virtual machine type (CPU)
   - Choose appropriate VM size
   - Set minimum nodes to 0 to avoid charges when not in use
   - Configure idle seconds before scale down
   - Select your user-assigned managed identity

### Creating a Compute Instance

A compute instance is a fully managed cloud-based workstation for data scientists.

1. **Create a new compute instance**:
   - Provide a unique name
   - Select virtual machine type and size
   - Configure scheduling and auto-shutdown policies to optimize costs

### Accessing Your Compute Instance

Two options are available:

1. **Browser-Based Development**:
   - Jupyter Notebooks
   - JupyterLab
   - Terminal

2. **Local VS Code Connection**:
   - Connect your local VS Code to the remote compute instance

### Setting Up Your Development Environment

1. **Check available Conda environments**:
   ```bash
   conda env list
   ```

2. **Activate the Azure ML SDK v2 environment**:
   ```bash
   conda activate azureml_py310_sdkv2
   ```

## 07 - Git Integration with Azure Machine Learning

Azure Machine Learning fully supports Git repositories for tracking machine learning development work.

### Benefits of Git Integration

- Track code changes
- Collaborate with team members
- Maintain version control for ML experiments
- Implement CI/CD pipelines for ML workflows
- Reproduce experiments with specific code versions

### Setting Up Git Integration

1. **Generate an SSH key** for secure Git operations:
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   ```

2. **Add the public key to your Git account**:
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```

3. **Clone a Git repository**:
   ```bash
   git clone git@ssh.dev.azure.com:v3/<your org>/<your project>/<your repo>
   cd your-repo
   ```

## 08 - Working with Data in Azure Machine Learning

Azure ML provides a framework for working with data throughout the machine learning lifecycle.

### Data Concepts

#### Data Stores

A datastore serves as a reference to an existing Azure storage account, offering:
- A common API for different storage types
- Easy discovery and sharing in team collaborations
- Secure connection information management

Each AML workspace has default datastores:
- **workspaceblobstore**: Stores uploads, job code snapshots, and pipeline data cache
- **workspaceworkingdirectory**: Stores data for notebooks and compute instances
- **workspacefilestore**: Alternative container for data upload
- **workspaceartifactstore**: Storage for assets like metrics, models, and components

#### Data Assets

Data assets are references to data sources that can be used in machine learning workflows, providing:
- Tracking and versioning of data
- Reproducibility of workflows
- Data sharing among team members
- Consistent data references across experiments

### Understanding the Diabetes Dataset

The dataset contains diagnostic measurements for predicting diabetes, with features including:
- Pregnancies
- PlasmaGlucose
- DiastolicBloodPressure
- TricepsThickness
- SerumInsulin
- BMI
- DiabetesPedigree
- Age
- Diabetic (target variable: 1 = has diabetes, 0 = no diabetes)

### Registering the Dataset in Azure ML

The `create_aml_dataset.py` script automates the process of registering the dataset:
1. Connects to your Azure ML workspace
2. Validates the data file exists
3. Creates a data asset in your AML workspace

### Versioning Data Assets

Each time you register a dataset, a new version is created, which is useful for:
- Tracking changes to data over time
- Ensuring reproducibility of experiments
- Comparing model performance across different data versions

## 09 - Exploratory Data Analysis (EDA)

Exploratory Data Analysis is a critical preliminary step in the data science workflow.

### Purpose of EDA

EDA involves analyzing and visualizing datasets to:
- Extract important features
- Detect outliers, anomalies, missing values
- Analyze feature distributions
- Identify correlations between variables
- Test underlying assumptions

### Diabetes Dataset EDA

The `eda.ipynb` notebook demonstrates EDA of the diabetes dataset, including:

1. **Basic Dataset Insights**:
   - Data preview
   - Statistical summary
   - Check for duplicate records

2. **Class Distribution**:
   - Visualization of diabetic vs. non-diabetic patients
   - Identification of potential class imbalance

3. **Feature Distributions**:
   - Histograms with kernel density estimation
   - Density plots showing distribution differences between diabetic and non-diabetic patients

### Running the EDA Notebook

The notebook can be run in:
1. **AML Studio**: Using the selected compute target and Python kernel
2. **VS Code**: Connected to the Azure ML compute instance with the appropriate Python kernel

## 10 - Azure Machine Learning Environments

An Azure Machine Learning environment defines a versioned execution environment for machine learning jobs and deployments.

### Purpose of AML Environments

Environments ensure:
- Reproducibility
- Consistency
- Portability across different compute targets and team members

### Environment Definition

The `conda.yaml` file defines the Python dependencies for the diabetes classification use case, including:
- Python 3.10 as the base interpreter
- Data processing libraries (pandas, numpy)
- Machine learning libraries (scikit-learn)
- Visualization tools (matplotlib)
- Azure ML integration packages
- MLflow for experiment tracking

### Registering an Environment

The `register_aml_environment_from_yaml.py` script registers the environment with your Azure ML workspace, making it available for training jobs and deployments.

## 11 - Training Models with Azure Machine Learning

Azure Machine Learning offers two approaches for training models: the Experiment Job approach and the Pipeline approach.

### AML Jobs

An Azure ML job is the fundamental unit of work, representing:
- A single execution of a specific machine learning task
- One discrete operation like training a model
- Execution in a specific compute context
- Resource requirements, logs, metrics, and outputs

### AML Pipelines

An Azure ML pipeline is a workflow connecting multiple jobs:
- Orchestrates a sequence of machine learning tasks
- Enables workflow automation
- Allows parallel execution of independent steps
- Provides reusability across different scenarios
- Supports incremental runs
- Enables tracking of data lineage

### When to Use Each

- **Use Jobs**: For single, isolated tasks like experimenting with a model training script
- **Use Pipelines**: For reproducible, automated workflows with multiple steps

### Benefits of AML Pipelines

1. **Standardized MLOps & Team Collaboration**: Enables different teams to work independently while integrating their work
2. **Training Efficiency & Cost Reduction**: Reuses previous outputs and optimizes resource usage

### Experiment Job Approach

The `create_aml_experiment_job_train.py` script creates a standalone job for diabetes model training, which:
- Runs in a specified experiment
- Uses a defined environment
- Trains a model with specified parameters
- Tracks metrics and artifacts

### Pipeline Approach

The `create_aml_pipeline_train.py` script creates a pipeline with three connected nodes:
1. **Train Model Node**: Trains the model with specified parameters
2. **Evaluate Model Node**: Evaluates model performance and compares with historical versions
3. **Register Model Node**: Registers the model if it meets performance criteria

## 12 - Deploying Models with Azure Machine Learning

Azure ML provides online endpoints and deployments for real-time model inferencing.

### Online Endpoints and Deployments

#### Model Online Managed Endpoints

An online managed endpoint serves as the HTTPS interface for model inference, featuring:
- Authentication (key-based or token-based)
- Traffic management across deployments
- Monitoring of availability and performance
- Network isolation options

#### Model Deployments

A deployment hosts the model and executes the inference code, with:
- Model files, scoring script, and environment
- Compute resource specifications
- Independent lifecycle from the endpoint

### Creating an Online Endpoint

The `create_online_endpoint.py` script creates a managed online endpoint with parameters for:
- Endpoint name
- Description
- Authentication mode
- Network access

### Creating a Deployment

The `create_online_deployment.py` script creates a deployment with parameters for:
- Deployment name
- Endpoint name
- Model reference
- Environment
- Scoring script
- Compute resources
- Traffic allocation

### Inference Process

When a request is sent to the endpoint:
1. The endpoint authenticates the request
2. Traffic is routed to the appropriate deployment
3. The deployment executes the scoring script
4. The script processes input data and returns a prediction

### Monitoring and Security

Azure ML provides capabilities for:
- Tracking metrics like request counts and latency
- Accessing logs for debugging
- Implementing autoscaling
- Securing endpoints with private access and network isolation

## 13 - Testing the Deployed Model

After deploying the diabetes classification model, it can be tested using sample data.

### Test Files

- `test_online_endpoint.py`: Script to invoke the deployed endpoint
- `diabetes-classify-request.json`: Sample patient diagnostics data

### Running the Test

The test script sends a request to the endpoint with sample data and outputs the prediction result:
```
Diabetic prediction: ["not-diabetic"] or ["diabetic"]
```

### Alternative Testing with cURL

The endpoint can also be tested directly using cURL:
```bash
curl -X POST \
https://diabetes-classify.francecentral.inference.ml.azure.com/score \
-H 'Content-Type: application/json' \
-H 'Accept: application/json' \
-H 'Authorization: Bearer <your endpoint key>' \
-d @diabetes-classify-request.json
```

## 14 - Azure Machine Learning Components

Components are self-contained pieces of code that perform specific steps in a machine learning pipeline.

### Component Structure

A component consists of:
1. **Metadata**: Name, version, type, and other identifying information
2. **Interface**: Input/output specifications
3. **Command, Code & Environment**: Execution instructions and requirements

### Benefits of Components

- **Well-defined interface**: Clear input and output definitions
- **Share and reuse**: Easy sharing across pipelines and teams
- **Version control**: Versioned for compatibility and reproducibility
- **Unit testable**: Self-contained for independent testing

## 15 - MLOps Considerations

MLOps (Machine Learning Operations) focuses on standardizing and automating the machine learning lifecycle.

### Standardized Pipeline Templates

Standardizing MLOps pipeline steps offers advantages:
1. **Consistency**: Uniform implementation across projects
2. **Efficiency**: Reduced development time with ready-to-use templates
3. **Maintainability**: Centralized pipeline logic
4. **Governance**: Compliance with organizational standards

### Automated Change Detection

Incorporating repository change detection into MLOps pipelines allows:
- Selective execution of pipeline stages based on file changes
- Skipping unnecessary steps (e.g., skipping model training if only deployment parameters change)

## Conclusion

This walkthrough has covered the entire machine learning lifecycle using Azure Machine Learning, from workspace creation to model deployment and MLOps considerations. By following these steps, you can build, train, deploy, and manage machine learning models for various use cases, including the diabetes prediction example demonstrated in this repository.

Azure Machine Learning provides a comprehensive platform for data scientists and ML engineers to accelerate their work, collaborate effectively, and implement best practices for machine learning operations.
