# MLFlow Tutorial
MLflow is an open source platform for the end-to-end machine learning lifecycles. We can identify four major components:
* MLflow Tracking: An API to log parameters, code, and results in machine learning experiments and compare them using an interactive UI.
* MLflow Projects: A code packaging format for reproducible runs using Docker, so you can share your ML code with others.
* MLflow Models: A model packaging format and tools that let you easily deploy the same model (from any ML library) to batch and real-time scoring on other platforms.
* MLflow Model Registry: A centralized model store, set of APIs, and UI, to collaboratively manage the full lifecycle of MLflow Models.

## Create a virtualenv
Be sure you have created a virtualenv with the necessary dependencies.
In case you are using `pyenv`, you can run the following commands to create a working Python 3.10.0 environment:

    $ pyenv virtualenv 3.10.0 mlflow
    $ pyenv activate mlflow
    $ pip install -r requirements.txt

## Running a Simple ML Pipeline with the Tracking API
First of all, you need to lunch the the tracking UI, which will show runs logged in `./mlruns` at `http://127.0.0.1:5000`.
To do so, please open the terminal and run the following command:


    $ mlflow ui

You can now run the `train.py` file as follows:

    $ python train.py 1000

where, for example, `1000` is the argument that specifies the number of iterations needed at training time.

## Running a ML Experiment with a specific Tracking ID
Sometimes it is good practice to assign a unique ID to track a specific ML Experiment, 
so that every new run will be associated the same ID (but different tag). To do do, before launching the experiment,
we can run the following:


    $ mlflow experiments create --experiment-name <my_experiment>
    $ MLFLOW_EXPERIMENT_ID=<mlflow_experiment_id> python train.py 1000

where `mlflow_experiment_id` is the ID associated to the newly created experiment.

## Running a Project from a URI
The `mlflow run` command lets you run a project packaged with a `MLproject` file from a local path or a Git URI.
For example:

    $ mlflow run .  -P max_iter=1000 --experiment-id <mlflow_experiment_id> --env-manager=local

##  Saving and Serving Models
Let us take the `translotor.py` example. To serve a registered model, please run the following:

    $ MLFLOW_EXPERIMENT_ID=<mlflow_experiment_id> python translator.py
    Model saved in run <run-id>

    $ mlflow models serve -m runs:/<run-id>/model -p 5001 --env-manager=local

    $  curl http://127.0.0.1:5001/invocations -H 'Content-Type:application/json' -d '{"dataframe_split": {"index": [0], "columns": ["text"], "data": [["My name is Andrea. Ed il tuo?"]]}}'


## Serving the model via API
```python
import mlflow
logged_model = 'runs:/<run_id>/model' # you can easily find this throught the mlflow UI

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
df = pd.DataFrame({'text': 'I like the food'}, index=[0])
loaded_model.predict(df)
```

## Extra Tips
In case you need ro rename the registered model via the Python client, you can run the following command:
```python
from mlflow.tracking.client import MlflowClient
client = MlflowClient()
latest_version_info = client.get_latest_versions(name='a_model_name')
client.rename_registered_model(
    name='a_model_name', new_name='new_name'
)
````

To move the model to Stage, 
```python
from mlflow.tracking.client import MlflowClient
client = MlflowClient()
latest_version_info = client.get_latest_versions(name='a_model_name')

# To move the model to Stage
client.transition_model_version_stage(
    name='a_model_name', version=latest_version_info[0].version, stage="Staging"
)
```