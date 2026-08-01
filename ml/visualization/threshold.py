from __future__ import annotations

import logging

import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

from ml.visualization.base import BasePlotter


logger = logging.getLogger(__name__)


class ThresholdPlotter(BasePlotter):

    def __init__(self):

        super().__init__()

    def compute_metrics(
        self,
        model,
        X_test,
        y_test,
        step=0.05,
    ):

        logger.info("Computing Threshold Metrics...")

        probabilities = model.predict_proba(X_test)[:, 1]

        thresholds = np.arange(
            step,
            1.0,
            step,
        )

        precision_list = []
        recall_list = []
        f1_list = []

        for threshold in thresholds:

            predictions = (
                probabilities >= threshold
            ).astype(int)

            precision_list.append(
                precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            )

            recall_list.append(
                recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            )

            f1_list.append(
                f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            )

        return (
            thresholds,
            precision_list,
            recall_list,
            f1_list,
        )

    def plot(
        self,
        model,
        X_test,
        y_test,
    ):

        (
            thresholds,
            precision,
            recall,
            f1,
        ) = self.compute_metrics(
            model,
            X_test,
            y_test,
        )

        best_index = np.argmax(f1)

        best_threshold = thresholds[best_index]

        best_f1 = f1[best_index]

        logger.info(
            f"Best Threshold : {best_threshold:.2f}"
        )

        fig, ax = self.create_figure(
            figsize=(9, 6)
        )

        ax.plot(
            thresholds,
            precision,
            linewidth=2,
            label="Precision",
        )

        ax.plot(
            thresholds,
            recall,
            linewidth=2,
            label="Recall",
        )

        ax.plot(
            thresholds,
            f1,
            linewidth=2.5,
            label="F1 Score",
        )

        ax.scatter(
            best_threshold,
            best_f1,
            s=120,
            marker="o",
            label=f"Best = {best_threshold:.2f}",
        )

        ax.axvline(
            best_threshold,
            linestyle="--",
            alpha=0.6,
        )

        ax.legend()

        self.style_axis(
            ax,
            title="Threshold Analysis",
            xlabel="Decision Threshold",
            ylabel="Metric Score",
        )

        return fig

    def plot_and_save(
        self,
        model,
        X_test,
        y_test,
        filename="threshold_analysis.png",
    ):

        fig = self.plot(
            model,
            X_test,
            y_test,
        )

        return self.finish(
            fig,
            filename,
        )