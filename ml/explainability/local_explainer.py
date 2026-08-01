from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ml.explainability.shap_explainer import SHAPExplainer
from ml.explainability.waterfall import WaterfallPlotter

logger = logging.getLogger(__name__)


class LocalExplainer:

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

        self.waterfall = WaterfallPlotter(
            model=model,
            output_dir=output_dir,
        )

    def explain(
        self,
        X,
        feature_names,
        sample_index: int = 0,
        generate_plot: bool = True,
    ):

        logger.info(
            "Generating local explanation..."
        )

        shap_values = self.shap_engine.explain(X)

        explanation = shap_values[sample_index]

        feature_values = X.iloc[sample_index]

        contributions = []

        for i, feature in enumerate(feature_names):

            contributions.append({

                "feature": feature,

                "value": float(feature_values.iloc[i]),

                "shap_value": float(
                    explanation.values[i]
                ),

                "abs_shap": abs(
                    float(
                        explanation.values[i]
                    )
                )

            })

        contributions = sorted(

            contributions,

            key=lambda x: x["abs_shap"],

            reverse=True,

        )

        if generate_plot:

            self.waterfall.plot(

                X,

                sample_index=sample_index,

            )

        result = {

            "base_value": float(

                np.squeeze(
                    explanation.base_values
                )

            ),

            "prediction": float(

                explanation.base_values +

                explanation.values.sum()

            ),

            "top_features": contributions,

        }

        logger.info(
            "Local explanation completed."
        )

        return result