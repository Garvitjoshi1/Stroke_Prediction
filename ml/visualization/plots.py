from __future__ import annotations

import logging

from ml.visualization.roc import ROCPlotter
from ml.visualization.pr import PRCurvePlotter
from ml.visualization.confusion import ConfusionMatrixPlotter
from ml.visualization.threshold import ThresholdPlotter
from ml.visualization.calibration import CalibrationPlotter
from ml.visualization.importance import FeatureImportancePlotter


logger = logging.getLogger(__name__)


class PlotManager:
    
    def __init__(self):

        self.roc = ROCPlotter()

        self.pr = PRCurvePlotter()

        self.confusion = ConfusionMatrixPlotter()

        self.threshold = ThresholdPlotter()

        self.calibration = CalibrationPlotter()

        self.importance = FeatureImportancePlotter()

    def generate_all(
        self,
        model,
        X_test,
        y_test,
        feature_names,
    ):

        logger.info("=" * 70)
        logger.info("Generating Visualization Package...")
        logger.info("=" * 70)

        self.roc.plot_and_save(
            model,
            X_test,
            y_test,
        )

        self.pr.plot_and_save(
            model,
            X_test,
            y_test,
        )

        self.confusion.plot_and_save(
            model,
            X_test,
            y_test,
            normalize=True,
        )

        self.threshold.plot_and_save(
            model,
            X_test,
            y_test,
        )

        self.calibration.plot_and_save(
            model,
            X_test,
            y_test,
        )

        self.importance.plot_and_save(
            model,
            X_test,
            y_test,
            feature_names,
        )

        logger.info("=" * 70)
        logger.info("Visualization Package Completed.")
        logger.info("=" * 70)