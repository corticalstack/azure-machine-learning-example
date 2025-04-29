## Exploratory Data Analysis (EDA)

### Introduction to EDA

Exploratory Data Analysis (EDA) is a critical preliminary step in the data science workflow that involves analyzing and visualizing datasets to understand and summarize their main characteristics. Typically, we employ techniques to:

- Extract important features
- Detect outliers, anomalies, missing values
- Analyze feature distributions
- Identify correlations between variables
- Test underlying assumptions

### Diabetes Dataset EDA Notebook Contents

The `eda.ipynb` notebook is a simple example to demonstrate what EDA of the diabetes dataset might look like, including:

1. **Basic Dataset Insights**
   - A data preview
   - A statistical summary
   - A check for duplicate records

2. **Class Distribution**

   *This visualization shows the distribution of diabetic vs. non-diabetic patients in the dataset. It helps identify potential class imbalance, which is critical for understanding model performance metrics and potentially implementing techniques like class weighting or resampling strategies.*
   
   ![Class Distribution](..//00-assets/images/class_distribution.png)
   


3. **Feature Distributions**

   *These histograms with kernel density estimation show the distribution of each feature in the dataset. They help identify the shape of distributions (normal, skewed, bimodal), potential outliers, and the range of values, which can inform feature scaling and transformation decisions.*
   
   ![Feature Distribution](../00-assets/images/feature_distribution.png)
   

   
   *These density plots show how feature distributions differ between diabetic and non-diabetic patients. Areas where the distributions diverge significantly indicate features that may have strong predictive power for diabetes, helping guide feature selection and importance analysis.*
   
   ![Density Plot](../00-assets/images/density_plot.png)
   

4. **Running the EDA notebook in AML studio**

   *Note in the screenshot below the selected compute target and python kernel*

   ![EDA notebook AML studio](..//00-assets/images/eda_notebook_aml_studio.png)


4. **Running the EDA notebook in VS Code**

   *Observe we are connected to the Azure ML compute instance, as indicated in the bottom left corner. Additionally, the correct Python kernel has been selected, ensuring that we're using the appropriate environment for our analysis.*

   ![EDA notebook VSCode](..//00-assets/images/eda_notebook_vscode.png)
