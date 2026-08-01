from __future__ import annotations

import logging

from sklearn.metrics import (
    roc_curve,
    auc,
)

from ml.visualization.base import BasePlotter


logger = logging.getLogger(__name__)


class ROCPlotter(BasePlotter):
    
    def __init__(self):

        super().__init__()

    def plot(
        self,
        model,
        X_test,
        y_test,
    ):

        logger.info("Generating ROC Curve...")

        probabilities = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(
            y_test,
            probabilities,
        )

        roc_auc = auc(
            fpr,
            tpr,
        )

        fig, ax = self.create_figure(
            figsize=(7, 6)
        )

        ax.plot(
            fpr,
            tpr,
            linewidth=2.5,
            label=f"AUC = {roc_auc:.4f}",
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            linewidth=1.5,
        )

        ax.legend(
            loc="lower right"
        )

        self.style_axis(
            ax,
            title="Receiver Operating Characteristic (ROC)",
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
        )

        return fig

    def plot_and_save(
        self,
        model,
        X_test,
        y_test,
        filename="roc_curve.png",
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