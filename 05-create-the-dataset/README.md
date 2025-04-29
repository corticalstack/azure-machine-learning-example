# Working with Data in Azure Machine Learning

## Introduction to Data in Azure Machine Learning

Azure Machine Learning (AML) provides a framework for working with data throughout the machine learning lifecycle. It offers features to store, access, prepare, and monitor data.

## Understanding Data Concepts

### Data Stores

A datastore in Azure Machine Learning serves as a reference to an existing Azure storage account. Datastores offer these key benefits:

- A common, easy-to-use API that interacts with different storage types (e.g., blobs and files)
- Easy discovery and sharing of datastores in team collaborations
- For credential-based access, an Azure Machine Learning datastore secures connection information, eliminating the need to expose sensitive information in scripts

Each Azure Machine Learning workspace has default datastores that are automatically created:
- workspaceblobstore: Stores data uploads, job code snapshots, and pipeline data cache
- workspaceworkingdirectory: Stores data for notebooks, compute instances, and prompt flow
- workspacefilestore: Alternative container for data upload
- workspaceartifactstore: Storage for assets such as metrics, models, and components

### Data Assets

Data assets in Azure Machine Learning are references to data sources that can be used in machine learning workflows. They provide a way to:

- Track and version data used in experiments
- Ensure reproducibility of machine learning workflows
- Share data among team members and machine learning use cases
- Reference data consistently across different experiments and pipelines

## Dataset Monitoring

The model data collection feature in Azure Machine Learning allows you to collect data from models deployed in production, for example model predictions from input data. This helps you to monitor data drifts, analysing it with tools like Power BI. 

## Data Connections

Azure Machine Learning connections serve as key vault proxies that securely store connection credentials in the AML workspace key vault. These connections allow you to connect the AML lifecycle to data sources outside of Azure, such as:
  - Snowflake DB
  - Amazon S3
  - Azure SQL DB

Data connections simplify the process of working with external data sources while maintaining security best practices for credential management.

## Prerequisites for working with our sample dataset

- An Azure Machine Learning workspace (created in an [earlier step](../01-create-aml-workspace/README.md))
- Compute resources (set up in in an [earlier step](../02-create-aml-compute/README.md))

## Understanding the Diabetes Dataset

[This dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

### Dataset Features

The datasets consists of several medical predictor variables and one target variable, *Diabetic*. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on:

- **Pregnancies**: Number of times pregnant
- **PlasmaGlucose**: Plasma glucose concentration after 2 hours in an oral glucose tolerance test (mg/dL)
- **DiastolicBloodPressure**: Diastolic blood pressure (mm Hg)
- **TricepsThickness**: Triceps skin fold thickness (mm)
- **SerumInsulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)²)
- **DiabetesPedigree**: Diabetes pedigree function (a measure of diabetes history in relatives)
- **Age**: Age in years
- **Diabetic**: Target variable (1 = has diabetes, 0 = no diabetes)

## Setting Up Your Development Environment

Before working with the data, ensure you're using the correct environment:

1. Connect to your compute instance using VS Code or Jupyter
2. If using a terminal, activate the Azure ML SDK v2 environment:
   ```bash
   conda activate azureml_py310_sdkv2
   ```
3. Ensure you're in the directory containing this README and the data files

## Registering the Dataset in Azure ML

To make the diabetes dataset available for training in Azure ML, you need to register it as a data asset:

### Using the Provided Script

The `create_aml_dataset.py` script in this directory automates the process of registering the dataset:

```bash
python create_aml_dataset.py
```

By default, this will:
1. Use the `diabetes.csv` file in the `00-assets/data` directory
2. Create a new data asset version named `diabetes-diagnostics` in your Azure ML workspace

### Customizing Dataset Registration

You can customize the dataset registration with command-line options:

```bash
python create_aml_dataset.py --data_path PATH --dataset_name NAME --dataset_description DESC
```

### How the Script Works

The script performs the following operations:
1. Connects to your Azure ML workspace using the configuration file `config.json` in the root of your compute instance
2. Validates the data file exists
3. Creates a data asset in your AML workspace with the specified name and description
4. Registers the data asset, making it available for training jobs

## Versioning Data Assets

Each time you register a dataset, a new version is created. This is useful for:
- Tracking changes to data over time
- Ensuring reproducibility of your machine learning experiments
- Comparing model performance across different data versions

## Next Steps

After registering our diabetes dataset, typical next steps would include:
- Performing Exploratory Data Analysis (EDA) to gain insights into the data's characteristics and relationships
- Preparing the data for modeling, including feature selection and engineering
- Experimenting with different machine learning algorithms to assess their predictive performance

For more information on working with data in Azure ML, refer to the [Azure Machine Learning data documentation](https://docs.microsoft.com/en-us/azure/machine-learning/concept-data?view=azureml-api-2).
