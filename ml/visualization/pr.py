from __future__ import annotations

import logging

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
)

from ml.visualization.base import BasePlotter


logger = logging.getLogger(__name__)


class PRCurvePlotter(BasePlotter):
    
    def __init__(self):

        super().__init__()

    def plot(
        self,
        model,
        X_test,
        y_test,
    ):

        logger.info("Generating Precision-Recall Curve...")

        probabilities = model.predict_proba(X_test)[:, 1]

        precision, recall, _ = precision_recall_curve(
            y_test,
            probabilities
        )

        ap = average_precision_score(
            y_test,
            probabilities
        )

        fig, ax = self.create_figure(
            figsize=(7, 6)
        )

        ax.plot(
            recall,
            precision,
            linewidth=2.5,
            label=f"AP = {ap:.4f}"
        )

        ax.legend(
            loc="lower left"
        )

        self.style_axis(
            ax,
            title="Precision-Recall Curve",
            xlabel="Recall",
            ylabel="Precision"
        )

        return fig

    def plot_and_save(
        self,
        model,
        X_test,
        y_test,
        filename="precision_recall_curve.png",
    ):

        fig = self.plot(
            model,
            X_test,
            y_test
        )

        return self.finish(
            fig,
            filename
        )