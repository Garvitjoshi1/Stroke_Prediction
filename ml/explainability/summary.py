from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import shap

from ml.explainability.shap_explainer import SHAPExplainer

logger = logging.getLogger(__name__)


class SummaryPlotter:

    def __init__(
        self,
        model=None,
        output_dir="artifacts/explainability",
    ):

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.shap_engine = SHAPExplainer(
            model=model,
            output_dir=output_dir,
        )

    def plot(
        self,
        X,
        feature_names,
        max_display=20,
        save=True,
        show=False,
    ):

        logger.info(
            "Generating SHAP Summary Plot..."
        )

        shap_values = self.shap_engine.explain(X)

        plt.figure(
            figsize=(10, 8)
        )

        shap.summary_plot(

            shap_values.values,

            X,

            feature_names=feature_names,

            max_display=max_display,

            show=False,

        )

        if save:

            filepath = (
                self.output_dir
                / "summary_plot.png"
            )

            plt.savefig(
                filepath,
                dpi=300,
                bbox_inches="tight",
            )

            logger.info(
                "Saved -> %s",
                filepath,
            )

        if show:

            plt.show()

        else:

            plt.close()

        return shap_values