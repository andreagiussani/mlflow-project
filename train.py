import sys

import pandas as pd
import matplotlib.pyplot as plt

import mlflow.sklearn
from mlflow.tracking.client import MlflowClient

from sklearn.metrics import classification_report, precision_recall_fscore_support as score, ConfusionMatrixDisplay, \
    confusion_matrix
from sklearn.linear_model import LogisticRegression

from constants import URL_DIABETES_DATA
from utils import get_data

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Args
max_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

# Load Diabetes datasets
X_train, X_test, y_train, y_test = get_data(filepath=URL_DIABETES_DATA)

# Model
with mlflow.start_run() as run:

    mlflow.set_tag("CostAllocation", "a_specific_tag")
    mlflow.set_tag("team", "a_team")
    lr = LogisticRegression(max_iter=max_iter)
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    precision, recall, f1score, support = score(
        y_test, y_pred, average='macro'
    )

    # Log mlflow attributes for mlflow UI
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1score)

    # Log Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=lr.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lr.classes_)
    disp.plot()
    plt.savefig('report.png')
    mlflow.log_artifact('report.png')

    # Log Model
    mlflow.sklearn.log_model(
        lr, "model",
        registered_model_name='diabetes-model'
    )
    print("Model saved in run %s" % run.info.run_uuid)


# Register model name in the model registry
registered_model_name = 'diabetes-model'
client = MlflowClient()

latest_version_info = client.get_latest_versions(name=registered_model_name)
print(latest_version_info[0].version)

desc = f"Precision> {precision}"
client.update_model_version(registered_model_name, version=latest_version_info[0].version, description=desc)

client.set_model_version_tag(
  name=registered_model_name,
  version=latest_version_info[0].version,
  key="CostAllocation",
  value="a_specific_tag"
)

