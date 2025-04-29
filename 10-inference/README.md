# Testing Azure ML Diabetes Classification Model

Here we test the the trained diabetes classification model which has been deployed to an AML online dendpoint deployment. 
## Overview

## Files in this Directory

- `test_online_endpoint.py`: Python script to invoke the deployed endpoint with test data
- `diabetes-classify-request.json`: Sample diabetes patient diagnostics data (below)

```json
{
  "data": [
    [2, 180, 74, 24, 21, 23.9091702, 1.488172308, 22]
  ]
}
```

## Running the Test

To test the deployed model, run the following command:

```bash
python test_online_endpoint.py --endpoint_name diabetes-classify --deployment_name diabetes-classify-blue-dp --request_file diabetes-classify-request.json
```

## Expected Output

Upon successful execution, the script will output the prediction result:

```
Diabetic prediction: ["not-diabetic"]
```

The prediction will be one of two possible values:
- `"not-diabetic"`: The model predicts the patient is not diabetic
- `"diabetic"`: The model predicts the patient is diabetic
