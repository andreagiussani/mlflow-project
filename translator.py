import json

from transformers import pipeline
import mlflow
from mlflow.models import ModelSignature

from constants import (
    TASK_NAME,
    TEXT_COLNAME,
)


class Translator(mlflow.pyfunc.PythonModel):

    def __init__(self, model: str = "Helsinki-NLP/opus-mt-en-fr"):
        self.model = model

    def hf_pipeline(self, row):
        translator = pipeline(task=TASK_NAME, model=self.model)
        return translator(row[TEXT_COLNAME])

    def predict(self, context, model_input):
        model_input[[TEXT_COLNAME]] = model_input.apply(self.hf_pipeline, axis=1)
        return model_input


model_input = json.dumps([{'name': TEXT_COLNAME, 'type': 'string'}])
model_output = json.dumps([{'name': TEXT_COLNAME, 'type': 'string'}])
signature = ModelSignature.from_dict({'inputs': model_input, 'outputs': model_output})


mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Start tracking
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model('model',
                            python_model=Translator(),
                            registered_model_name=None,
                            signature=signature,
                            input_example=None,
                            await_registration_for=0
                            )
