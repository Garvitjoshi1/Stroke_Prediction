import logging

from ml.models.metrics import MetricsCalculator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


class ModelEvaluator:

    def __init__(self):

        self.metrics = None

        self.confusion_matrix = None

        self.report = None

    def evaluate(
        self,
        trainer,
        X_test,
        y_test
    ):

        logger.info("Evaluating model...")

        predictions = trainer.predict(X_test)

        probabilities = trainer.predict_proba(X_test)[:, 1]

        self.metrics = MetricsCalculator.classification_metrics(
            y_test,
            predictions,
            probabilities
        )

        self.confusion_matrix = (
            MetricsCalculator.confusion_matrix_df(
                y_test,
                predictions
            )
        )

        self.report = (
            MetricsCalculator.classification_report_dict(
                y_test,
                predictions
            )
        )

        logger.info("Evaluation completed.")

        return {

            "metrics": self.metrics,

            "confusion_matrix": self.confusion_matrix,

            "classification_report": self.report

        }