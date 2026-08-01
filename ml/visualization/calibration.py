from __future__ import annotations

import logging

import numpy as np

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from ml.visualization.base import BasePlotter


logger = logging.getLogger(__name__)


class CalibrationPlotter(BasePlotter):

    def __init__(self):

        super().__init__()

    def plot(
        self,
        model,
        X_test,
        y_test,
        bins: int = 10,
    ):

        logger.info("Generating Calibration Curve...")

        probabilities = model.predict_proba(X_test)[:, 1]

        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test,
            probabilities,
            n_bins=bins,
            strategy="uniform",
        )

        brier = brier_score_loss(
            y_test,
            probabilities,
        )

        fig, axes = self.create_figure(
            figsize=(8, 10)
        )

        plt = fig.subplots(
            2,
            1,
            height_ratios=[3, 1],
        )

        ax1 = plt[0]
        ax2 = plt[1]

        ax1.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            linewidth=2,
            label="Perfect Calibration",
        )

        ax1.plot(
            mean_predicted_value,
            fraction_of_positives,
            marker="o",
            linewidth=2,
            label=f"Model (Brier={brier:.4f})",
        )

        ax1.legend()

        self.style_axis(
            ax1,
            title="Calibration Curve",
            xlabel="Mean Predicted Probability",
            ylabel="Observed Frequency",
        )

        ax2.hist(
            probabilities,
            bins=bins,
        )

        self.style_axis(
            ax2,
            title="Prediction Distribution",
            xlabel="Predicted Probability",
            ylabel="Count",
            grid=False,
        )

        fig.tight_layout()

        return fig

    def plot_and_save(
        self,
        model,
        X_test,
        y_test,
        filename="calibration_curve.png",
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