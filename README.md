# 🤖 Azure Machine Learning Diabetes Demo

This repository demonstrates features of Azure Machine Learning (AML), including how to build and deploy low and pro code machine learning models for diabetes prediction. It provides a comprehensive, step-by-step guide through the entire machine learning lifecycle, from workspace creation to model training, evaluation, deployment, and developing ML use cases with MLOps considerations in mind.

## 🔍 Introduction to Azure Machine Learning

Azure Machine Learning is a cloud-based service that enables data scientists and ML engineers to accelerate the end-to-end machine learning lifecycle. It provides a comprehensive set of tools for:

- Building and training models using a robust set of tools
- Deploying and monitoring models in production
- Managing the ML lifecycle with MLOps practices
- Scaling ML workloads efficiently
- Ensuring responsible AI principles

## 📊 About the Diabetes Dataset

This repository uses the Pima Indians Diabetes Dataset, a standard toy dataset in machine learning. The dataset was chosen for several reasons:

- **Simplicity**: The dataset is small and straightforward, allowing us to focus on Azure Machine Learning concepts rather than complex data contexts.
- **Well-documented**: As a widely used dataset, its characteristics are well understood.
- **Real-world relevance**: Despite being a toy dataset, it represents a genuine healthcare use case.
- **Practical size**: The small size enables fast training iterations and low compute requirements, making it ideal for learning purposes.

## 🔧 Prerequisites

To run the examples in this repository, you'll need:

1. An active Azure subscription
2. Sufficient permissions to create resources in your subscription
3. Python 3.10 or later
4. Azure Machine Learning SDK v2 for Python

**Important:** This repo uses Azure Machine Learning SDK v2, not the older v1. While many examples and documentation for v1 still exist on the internet, v2 is recommended for new projects, offering an improved and more consistent API design.

## 📚 Examples

🧮 [01-create-aml-workspace](./01-create-aml-workspace/README.md): Step-by-step instructions for creating an AML workspace in the Azure portal.

🧮 [02-model-catalog](./02-model-catalog/README.md): Intro to the model catalog in AML.

🧮 [03-connections](./03-connections/README.md): Brief intro to connections in AML.

🧮 [04-promptflow](./04-promptflow/README.md): Brief intro to promptflow.

🧮 [05-automated-ml](./05-automated-ml/README.md): A walkthrough of creating an automated machine learning job for diabetes classification using AML.

🧮 [06-create-aml-compute](./06-create-aml-compute/README.md): Guide for setting up compute resources in AML.

🧮 [07-git-integration](./07-git-integration/README.md): Instructions for integrating Git repositories with AML.

🧮 [08-create-the-dataset](./08-create-the-dataset/README.md): Guide to working with data in AML, including registering the diabetes dataset.

🧮 [09-exploratory-data-analysis](./09-exploratory-data-analysis/README.md): Notebook demonstrating exploratory data analysis on the diabetes dataset.

🧮 [10-register-model-environment](./10-register-model-environment/README.md): Instructions for registering model environments in Azure Machine Learning.

🧮 [11-train-model](./11-train-model/README.md): Guide to training a diabetes prediction model using Azure Machine Learning, with two different approaches (job and pipeline).

🧮 [12-deploy-model](./12-deploy-model/README.md): Instructions for deploying the trained model as an online endpoint.

🧮 [13-inference](./13-inference/README.md): Example inference request to the deployed model to make a diabetes prediction from patient diagnostics.

🧮 [14-components](./14-components/README.md): Intro to components in AML.

🧮 [15-mlops-considerations](./15-mlops-considerations/README.md): Some MLOps considerations for machine learning projects.

## 🛠️ Getting Started

1. Clone this repository to your local machine or Azure Machine Learning compute instance
2. Follow the examples in order, starting with [01-create-aml-workspace](./01-create-aml-workspace/README.md)
3. Each example includes detailed instructions and explanations

## ❓ FAQ

<details>
<summary><strong>What is Azure Machine Learning?</strong></summary>
Azure Machine Learning is a cloud service for accelerating and managing the machine learning (ML) project lifecycle. ML professionals, data scientists, and engineers can use it in their day-to-day workflows to train and deploy models and manage machine learning operations (MLOps).
</details>

## 📖 Resources

- [Azure Machine Learning Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)
