from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import shap

from ml.explainability.shap_explainer import SHAPExplainer

logger = logging.getLogger(__name__)


class DependencePlotter:

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
        feature,
        interaction_feature="auto",
        save=True,
        show=False,
    ):

        logger.info(
            "Generating SHAP Dependence Plot..."
        )

        shap_values = self.shap_engine.explain(X)

        if isinstance(feature, int):

            feature_name = feature_names[feature]

        else:

            feature_name = feature

        plt.figure(
            figsize=(9, 7)
        )

        shap.dependence_plot(

            feature_name,

            shap_values.values,

            X,

            feature_names=feature_names,

            interaction_index=interaction_feature,

            show=False,

        )

        if save:

            filepath = (
                self.output_dir
                / f"dependence_{feature_name}.png"
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

        return filepath