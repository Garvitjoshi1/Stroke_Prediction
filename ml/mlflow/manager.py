from pathlib import Path

from ml.mlflow.tracker import MLFlowTracker
from ml.mlflow.logger import MLFlowLogger


class MLFlowManager:

    def __init__(self):

        self.tracker = MLFlowTracker()

        self.logger = MLFlowLogger()

    def log_training(
        self,
        trainer,
        metrics,
    ):

        with self.tracker.start_run("Baseline"):

            self.logger.log_params(

                trainer.model.get_params()

            )

            self.logger.log_metrics(metrics)

            figures = Path(
                "artifacts/figures"
            )

            if figures.exists():

                for file in figures.glob("*"):

                    self.logger.log_artifact(str(file))

            explain = Path(
                "artifacts/explainability"
            )

            if explain.exists():

                for file in explain.glob("*"):

                    self.logger.log_artifact(str(file))

            self.logger.log_model(
                trainer.model,
                "stroke_model",
            )