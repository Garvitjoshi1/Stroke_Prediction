import logging
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


class ThresholdOptimizer:

    def __init__(
        self,
        start=0.05,
        stop=0.95,
        step=0.05
    ):

        self.thresholds = np.arange(
            start,
            stop + step,
            step
        )

    def evaluate(
        self,
        model,
        X_test,
        y_test
    ):

        logger.info("Searching optimal threshold...")

        probabilities = model.predict_proba(X_test)[:, 1]

        rows = []

        for threshold in self.thresholds:

            predictions = (
                probabilities >= threshold
            ).astype(int)

            precision = precision_score(
                y_test,
                predictions,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                predictions,
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                predictions,
                zero_division=0
            )

            tn, fp, fn, tp = confusion_matrix(
                y_test,
                predictions
            ).ravel()

            specificity = tn / (tn + fp)

            balanced_accuracy = (
                recall + specificity
            ) / 2

            youdens_j = (
                recall + specificity - 1
            )

            rows.append({

                "threshold": threshold,

                "precision": precision,

                "recall": recall,

                "f1_score": f1,

                "specificity": specificity,

                "balanced_accuracy": balanced_accuracy,

                "youdens_j": youdens_j

            })

        results = pd.DataFrame(rows)

        logger.info("Threshold search completed.")

        return results

    def best_threshold(
        self,
        results,
        metric="recall"
    ):

        idx = results[metric].idxmax()

        best = results.loc[idx]

        logger.info(
            f"Best threshold based on {metric}: "
            f"{best['threshold']:.2f}"
        )

        return best