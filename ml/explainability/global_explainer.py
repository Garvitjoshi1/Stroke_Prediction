from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap

from ml.explainability.shap_explainer import SHAPExplainer

logger = logging.getLogger(__name__)


class GlobalExplainer:

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

    def compute_importance(
        self,
        X,
        feature_names,
    ):

        shap_values = self.shap_engine.explain(X)

        importance = np.abs(
            shap_values.values
        ).mean(axis=0)

        order = np.argsort(
            importance
        )[::-1]

        return {

            "values": shap_values,

            "importance": importance,

            "sorted_importance": importance[order],

            "sorted_features": np.array(
                feature_names
            )[order],

            "order": order,

        }

    def plot(
        self,
        X,
        feature_names,
        top_k=20,
        save=True,
        show=False,
    ):

        logger.info(
            "Generating global SHAP importance..."
        )

        results = self.compute_importance(
            X,
            feature_names,
        )

        importance = results[
            "sorted_importance"
        ][:top_k]

        features = results[
            "sorted_features"
        ][:top_k]

        fig = plt.figure(
            figsize=(10, 8)
        )

        plt.barh(
            features[::-1],
            importance[::-1],
        )

        plt.xlabel(
            "Mean |SHAP Value|"
        )

        plt.title(
            "Global Feature Importance"
        )

        plt.tight_layout()

        if save:

            filepath = (
                self.output_dir
                / "global_feature_importance.png"
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

            plt.close(fig)

        return results