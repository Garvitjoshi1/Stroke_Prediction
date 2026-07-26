import logging

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


class MetricsCalculator:

    @staticmethod
    def classification_metrics(
        y_true,
        y_pred,
        y_prob=None
    ):

        logger.info("Calculating classification metrics...")

        metrics = {

            "accuracy": accuracy_score(
                y_true,
                y_pred
            ),

            "precision": precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

            "recall": recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

            "f1_score": f1_score(
                y_true,
                y_pred,
                zero_division=0
            )

        }

        if y_prob is not None:

            metrics["roc_auc"] = roc_auc_score(
                y_true,
                y_prob
            )

            metrics["pr_auc"] = average_precision_score(
                y_true,
                y_prob
            )

        logger.info("Metrics calculated successfully.")

        return metrics

    @staticmethod
    def confusion_matrix_df(
        y_true,
        y_pred
    ):

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        return pd.DataFrame(
            cm,
            index=[
                "Actual Negative",
                "Actual Positive"
            ],
            columns=[
                "Predicted Negative",
                "Predicted Positive"
            ]
        )

    @staticmethod
    def classification_report_dict(
        y_true,
        y_pred
    ):

        return classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0
        )

    @staticmethod
    def classification_report_text(
        y_true,
        y_pred
    ):

        return classification_report(
            y_true,
            y_pred,
            zero_division=0
        )