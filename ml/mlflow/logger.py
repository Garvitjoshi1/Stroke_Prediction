import mlflow
import mlflow.sklearn


class MLFlowLogger:

    def log_params(self, params):

        mlflow.log_params(params)

    def log_metrics(self, metrics):

        mlflow.log_metrics(metrics)

    def log_artifact(self, path):

        mlflow.log_artifact(path)

    def log_model(
        self,
        model,
        name="model",
    ):

        mlflow.sklearn.log_model(
            model,
            artifact_path=name,
        )